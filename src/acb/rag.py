"""
Retrieval-augmented generation over the ACB Caribbean fee & service JSON files,
plus optionally scraped web content (see scrape_site.py -> data/web_content.json).

Flow:
    1. On startup, load_index() flattens data/antigua_fees.json,
       data/grenada_fees.json, data/business_services.json,
       data/grenada_business_services.json, and (if present)
       data/web_content.json into small self-contained text chunks.
    2. Each chunk is embedded once via the Gemini embedding API. Embeddings are
       cached to rag_cache.json, keyed by a hash of the source files, so a
       normal server restart doesn't re-embed everything.
    3. retrieve(query) embeds the incoming user message and returns the top-k
       chunks by cosine similarity, optionally filtered to one jurisdiction.

This is intentionally dependency-light (no numpy, no vector DB) because the
corpus is a few hundred short chunks -- brute-force cosine similarity over
that in pure Python is well under a millisecond and not worth the extra
moving parts. If this corpus grows into the thousands of chunks, swap the
linear scan in retrieve() for a proper vector index (e.g. FAISS, sqlite-vec)
but keep the chunking/caching structure the same.
"""

import hashlib
import json
import math
import os

import google.genai as genai
from google.genai import types

EMBEDDING_MODEL = "gemini-embedding-001"  # verify current model name against your SDK version
CACHE_PATH = "rag_cache.json"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

SOURCE_FILES = {
    os.path.join(DATA_DIR, "antigua_fees.json"): "fees",
    os.path.join(DATA_DIR, "grenada_fees.json"): "fees",
    os.path.join(DATA_DIR, "business_services.json"): "services",
    os.path.join(DATA_DIR, "grenada_business_services.json"): "services",
}

# Optional: output of scrape_site.py. Not in SOURCE_FILES above because it's
# fine for this file not to exist yet -- unlike the core fee/service JSON,
# which SHOULD crash loudly if missing since the bot has no data at all
# without them.
WEB_CONTENT_FILE = os.path.join(DATA_DIR, "web_content.json")
WEB_CHUNK_WORDS = 180  # ~ a paragraph or two; keeps chunks focused for retrieval

_client = None
_chunks: list[dict] = []          # [{id, type, jurisdiction, text, embedding}, ...]


# ---------------------------------------------------------------------------
# Chunking: JSON -> flat list of {id, type, jurisdiction, text}
# ---------------------------------------------------------------------------

def _format_fee_value(item: dict) -> str:
    if "fee" in item:
        return item["fee"]
    if "values" in item:
        return "; ".join(f"{k}: {v}" for k, v in item["values"].items())
    return ""


def _chunk_fees(fees: dict) -> list[dict]:
    chunks = []
    jurisdiction = fees["jurisdiction"]
    vat_applies = fees.get("vat_applies", False)
    for cat in fees["categories"]:
        for grp in cat.get("groups", []):
            notes = " ".join(grp.get("notes", []))
            for item in grp.get("items", []):
                text = (
                    f"[{jurisdiction} fee schedule] {cat['name']} > {grp['name']} "
                    f"— {item['item']}: {_format_fee_value(item)}."
                )
                if notes:
                    text += f" Note: {notes}"
                if vat_applies:
                    text += " (VAT may apply in addition to this fee unless stated otherwise.)"
                chunks.append({
                    "id": f"fee::{jurisdiction}::{cat['id']}::{grp['id']}::{item['item']}",
                    "type": "fee",
                    "jurisdiction": jurisdiction,
                    "text": text,
                })
    return chunks


def _chunk_services(svc: dict) -> list[dict]:
    chunks = []
    jurisdiction = svc["jurisdiction"]
    for s in svc["services"]:
        parts = [f"[{jurisdiction} service] {s['name']} ({s.get('category', '')})."]
        if s.get("summary"):
            parts.append(s["summary"])
        if s.get("details"):
            parts.append(s["details"])
        kf = s.get("key_facts")
        if kf:
            kf_text = "; ".join(
                f"{k.replace('_', ' ')}: {v if not isinstance(v, list) else ', '.join(v)}"
                for k, v in kf.items()
            )
            parts.append(f"Key facts: {kf_text}")
        if s.get("requirements"):
            parts.append("Requirements: " + "; ".join(s["requirements"]))
        if s.get("apply_url"):
            parts.append(f"Apply: {s['apply_url']}")
        # Guardrails are folded into the chunk text itself, so if this chunk
        # gets retrieved, the model sees the constraints right alongside the
        # facts -- e.g. "don't claim this account earns interest."
        guardrails = s.get("answer_guardrails")
        if guardrails:
            parts.append("IMPORTANT constraints when answering about this: " + " | ".join(guardrails))
        chunks.append({
            "id": f"service::{jurisdiction}::{s['id']}",
            "type": "service",
            "jurisdiction": jurisdiction,
            "text": " ".join(parts),
        })
    return chunks


def _chunk_web_content() -> list[dict]:
    """Loads and chunks data/web_content.json (scrape_site.py's output) if it
    exists. Each page is split into ~WEB_CHUNK_WORDS-word pieces rather than
    embedded as one giant chunk, since a whole page is too coarse for
    precise retrieval and too long/unfocused as context for the model."""
    if not os.path.exists(WEB_CONTENT_FILE):
        return []

    with open(WEB_CONTENT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    chunks = []
    for page in data.get("pages", []):
        words = page["text"].split()
        jurisdiction = page["jurisdiction"]
        for i in range(0, len(words), WEB_CHUNK_WORDS):
            piece = " ".join(words[i:i + WEB_CHUNK_WORDS])
            chunks.append({
                "id": f"web::{jurisdiction}::{page['url']}::{i}",
                "type": "web",
                "jurisdiction": jurisdiction,
                "text": f"[{jurisdiction} website — {page['title']}] {piece} (Source: {page['url']})",
            })
    return chunks


def _build_chunks() -> list[dict]:
    chunks = []
    for filename, kind in SOURCE_FILES.items():
        with open(filename, encoding="utf-8") as f:
            data = json.load(f)
        chunks += _chunk_fees(data) if kind == "fees" else _chunk_services(data)
    chunks += _chunk_web_content()
    return chunks


def _source_hash() -> str:
    """Hash of the source files' contents, so the cache invalidates itself
    whenever any of the JSON files change -- no manual cache-busting needed."""
    h = hashlib.sha256()
    for filename in sorted(SOURCE_FILES):
        with open(filename, "rb") as f:
            h.update(f.read())
    if os.path.exists(WEB_CONTENT_FILE):
        with open(WEB_CONTENT_FILE, "rb") as f:
            h.update(f.read())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Embedding + cache
# ---------------------------------------------------------------------------

def _embed_texts(client: genai.Client, texts: list[str]) -> list[list[float]]:
    """Embeds a list of texts, batching to stay under API request limits."""
    BATCH_SIZE = 20
    vectors = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=batch,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        vectors.extend(e.values for e in response.embeddings)
    return vectors


def load_index(client: genai.Client) -> None:
    """Call once at startup. Loads embeddings from cache if the source JSON
    hasn't changed since the cache was built; otherwise re-embeds and writes
    a fresh cache."""
    global _client, _chunks
    _client = client

    current_hash = _source_hash()
    chunks = _build_chunks()

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, encoding="utf-8") as f:
            cache = json.load(f)
        if cache.get("source_hash") == current_hash and len(cache.get("chunks", [])) == len(chunks):
            _chunks = cache["chunks"]
            print(f"RAG index: loaded {len(_chunks)} chunks from cache")
            return

    print(f"RAG index: (re)building embeddings for {len(chunks)} chunks...")
    embeddings = _embed_texts(client, [c["text"] for c in chunks])
    for c, vec in zip(chunks, embeddings):
        c["embedding"] = vec
    _chunks = chunks

    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({"source_hash": current_hash, "chunks": _chunks}, f)
    print(f"RAG index: built and cached {len(_chunks)} chunks")


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


JURISDICTIONS = {
    "antigua & barbuda": "Antigua & Barbuda",
    "antigua and barbuda": "Antigua & Barbuda",
    "antigua": "Antigua & Barbuda",
    "barbuda": "Antigua & Barbuda",
    "ag": "Antigua & Barbuda",
    "grenada": "Grenada",
    "gd": "Grenada",
}


def normalize_jurisdiction(raw: str | None) -> str | None:
    """Maps a jurisdiction value sent explicitly by the frontend (e.g. a
    country code or name tied to which site/subdomain the widget is embedded
    on) to the canonical jurisdiction string used everywhere else. Returns
    None for anything unrecognized rather than raising, since a bad/missing
    value should just fall back to keyword detection, not break the request."""
    if not raw:
        return None
    return JURISDICTIONS.get(raw.strip().lower())


def detect_jurisdiction(text: str) -> str | None:
    """Very simple keyword check -- good enough to filter retrieval when the
    customer names a country, without adding another model call. Returns
    None (search everything) if neither or both are mentioned."""
    t = text.lower()
    mentions_ag = "antigua" in t or "barbuda" in t
    mentions_gd = "grenada" in t
    if mentions_ag and not mentions_gd:
        return "Antigua & Barbuda"
    if mentions_gd and not mentions_ag:
        return "Grenada"
    return None


def retrieve(query: str, top_k: int = 6, jurisdiction: str | None = None) -> list[dict]:
    """Returns the top_k most relevant chunks for the query, each as
    {id, type, jurisdiction, text, score}. If jurisdiction is given, only
    chunks from that jurisdiction are considered."""
    if _client is None or not _chunks:
        return []

    query_vec = _embed_texts(_client, [query])[0]

    pool = _chunks if jurisdiction is None else [c for c in _chunks if c["jurisdiction"] == jurisdiction]
    scored = [
        {**c, "score": _cosine(query_vec, c["embedding"])}
        for c in pool
    ]
    scored.sort(key=lambda c: c["score"], reverse=True)
    return scored[:top_k]


def format_context(chunks: list[dict]) -> str:
    """Renders retrieved chunks as a block to drop into the system prompt."""
    if not chunks:
        return "(No matching reference information was found for this question.)"
    return "\n".join(f"- {c['text']}" for c in chunks)
