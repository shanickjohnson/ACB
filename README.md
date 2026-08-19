# ACB
2026 GEN AI CAMP

ACB Caribbean's digital banking assistant: a FastAPI backend that routes
customer messages through a LangGraph multi-agent pipeline (guardrails ->
intent router -> one of six domain agents -> escalation/output guardrail),
backed by a small RAG index over the bank's fee/service data and an
ElevenLabs voice layer for TTS/STT.

## Layout

```
ACB/
├── src/acb/
│   ├── config.py           # shared constants (e.g. CHAT_MODEL)
│   ├── i18n.py              # supported languages + all canned/translated messages
│   ├── guardrails.py        # PII/injection/jailbreak checks + guardrail graph nodes
│   ├── tools.py              # amortize(), CSV fast-path lookup, JSON reference loader
│   ├── rag.py                 # embeds + retrieves fee/service/web-content chunks
│   ├── scrape_site.py     # offline script: populates data/web_content.json
│   ├── agents/
│   │   ├── state.py         # ACBState (LangGraph shared state)
│   │   ├── router.py        # intent classification node
│   │   ├── domain_agents.py # the six domain agent nodes
│   │   ├── escalation.py    # human-handoff node
│   │   └── graph.py         # builds/compiles the graph, run_graph()
│   └── api/
│       ├── main.py          # FastAPI app: /chat, /languages, /calculate/*
│       ├── voice.py         # /tts, /stt (ElevenLabs)
│       └── schemas.py       # request bodies
├── data/                      # qa_data.csv, fee/service JSON, (generated) web_content.json
├── frontend/                  # index.html + static/ assets served at "/"
└── tests/                     # pytest — guardrails, router, tools
```

## Running locally

```bash
pip install -r requirements-dev.txt   # includes requirements.txt + pytest
```

Create a `.env` file in the repo root with:

```
GEMINI_API_KEY=...
ELEVENLABS_API_KEY=...          # optional — only needed for /tts and /stt
ELEVENLABS_VOICE_ID=...         # optional — defaults to the ACB assistant voice
```

Start the server from the repo root:

```bash
PYTHONPATH=src uvicorn acb.api.main:app --reload
```

The frontend is served at `/`; the FastAPI app reads `frontend/index.html`
and `data/*` relative to the repo root regardless of your working directory.

## Tests

```bash
pytest
```

`pytest.ini` sets `pythonpath = src` so tests import `acb` without any
extra setup. Guardrail regex, the router's fallback-to-`faq` behavior, and
`amortize()`'s edge cases are covered — these are pure functions with no
network calls.

## Refreshing the knowledge base

`src/acb/scrape_site.py` is a standalone script (not run at server
startup) that scrapes ACB Caribbean's public site into
`data/web_content.json`, which `rag.py` picks up alongside the fee/service
JSON files on the next embedding rebuild:

```bash
PYTHONPATH=src python -m acb.scrape_site
```
