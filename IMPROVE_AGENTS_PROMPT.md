# Prompt: Clean up ACB's agent architecture and file structure

Paste this into Claude Code (or any coding agent) at the root of the `ACB`
repo to drive the refactor. It's based on a read-through of the current
code, not guesswork — findings below cite the actual files.

---

## Context

ACB is a FastAPI + LangGraph banking assistant for ACB Caribbean. It has
two overlapping implementations living side by side:

1. **`app.py`** — the original monolith: `get_bot_reply()`, `ask_gemini()`,
   `translate_text()`, a hand-rolled `SESSIONS` dict, its own copies of the
   PII/injection/jailbreak regexes, and CSV loading.
2. **A LangGraph pipeline** — `graph.py`, `router.py`, `domain_agents.py`,
   `guardrails.py`, `escalation.py`, `agent_state.py`, `tools.py` — which is
   what `/chat` actually calls today via `run_graph()` (`app.py:304,310`).

The migration to (2) was never finished by deleting the leftovers of (1),
and the project has no package structure at all — every module sits at
repo root next to data files, the frontend, and generated artifacts.

## Confirmed problems (fix these, don't rediscover them)

**Dead code in `app.py`** — `get_bot_reply()` (line 402) is never called by
anything; `/chat` calls `run_graph()` directly (line 310). That makes the
following dead: `get_bot_reply`, `ask_gemini`, `translate_text`,
`CSV_REPLIES`/`load_csv_data`, `SESSIONS`/`get_or_create_session`/
`remember_turn`/`_prune_expired_sessions` (session state now lives in the
LangGraph SQLite checkpointer in `graph.py`), and the module-level
`PII_PATTERNS`/`INJECTION_PATTERNS`/`JAILBREAK_PATTERNS`/`SYSTEM_PROMPT`.
That's roughly 250+ lines to delete from `app.py`, keeping only: the
FastAPI app setup, `/chat`, `/tts`, `/stt`, the two `/calculate/*`
endpoints, and whatever those five endpoints actually need.

**Triplicated guardrail regex.** The exact same `PII_PATTERNS`,
`INJECTION_PATTERNS`, `JAILBREAK_PATTERNS` (and the functions built on them)
are defined independently in `app.py` and `guardrails.py`. Once the dead
code above is removed, `guardrails.py` is the only copy — keep it that way
and have everything import from there.

**Triplicated `amortize()`.** Identical implementations exist in `app.py`
and `tools.py`. `app.py`'s `/calculate/mortgage` and `/calculate/loan`
should import `amortize` from `tools.py` instead of keeping their own copy.

**Language/message tables scattered across 4 files.**
`SUPPORTED_LANGUAGES`, `REFUSAL_MESSAGE(S)`, `FALLBACK_MESSAGES` live in
`app.py`; `REFUSAL_MESSAGES` is re-declared in `guardrails.py`;
`HANDOFF_NOTE` is in `domain_agents.py`; `ESCALATION_MESSAGES` is in
`escalation.py`. These should be one module so a 5th language doesn't mean
editing four files and risking the copies drifting apart.

**Model name hardcoded 4 times.** `"gemini-3.6-flash"` is a literal string
in `app.py` (×2), `domain_agents.py`, and `router.py`. Pull it into one
constant (e.g. `CHAT_MODEL`) so upgrading the model is a one-line change.

**Repo hygiene:**
- There are *two* gitignore files: a working `.gitignore` and a stray
  tracked `gitignore` (no dot) — the second one does nothing and is just
  confusing. Delete it, or fold anything intentional in it into
  `.gitignore`.
- `.DS_Store` and `__pycache__/app.cpython-314.pyc` are tracked in git
  despite `.gitignore` supposedly covering them (they were committed
  before the ignore rule existed). Untrack them (`git rm --cached`) and
  confirm they stay out from then on.
- `rag_cache.json` and `acb_sessions.sqlite` are runtime-generated; the
  former is gitignored, but the SQLite checkpointer DB (`graph.py:79`)
  isn't in `.gitignore` at all — add it.

**No package structure.** Every `.py` module, every data file
(`qa_data.csv`, `*_fees.json`, `*_business_services.json`), the frontend
(`index.html`, `static/`), and the scraper (`scrape_site.py`) all sit
flat at repo root. There's no `tests/` directory at all.

## Target structure

```
ACB/
├── src/
│   └── acb/
│       ├── __init__.py
│       ├── config.py              # CHAT_MODEL, env var loading, shared constants
│       ├── i18n.py                # SUPPORTED_LANGUAGES, REFUSAL_MESSAGES,
│       │                          # FALLBACK_MESSAGES, HANDOFF_NOTE, ESCALATION_MESSAGES
│       ├── api/
│       │   ├── __init__.py
│       │   ├── main.py            # FastAPI app, CORS, /chat, /calculate/*
│       │   ├── schemas.py         # ChatMessage, TTSRequest, MortgageCalcRequest, LoanCalcRequest
│       │   └── voice.py           # /tts, /stt, elevenlabs_tts/stt, strip_markdown_for_speech
│       ├── agents/
│       │   ├── __init__.py
│       │   ├── state.py           # ACBState (was agent_state.py)
│       │   ├── graph.py           # build_graph/get_graph/run_graph
│       │   ├── router.py
│       │   ├── domain_agents.py
│       │   └── escalation.py
│       ├── guardrails.py          # single source of truth (was guardrails.py)
│       ├── tools.py               # amortize, csv_fast_path, load_json_reference, retrieve_faq_context
│       ├── rag.py
│       └── scrape_site.py
├── data/
│   ├── qa_data.csv
│   ├── antigua_fees.json
│   ├── grenada_fees.json
│   ├── business_services.json
│   └── grenada_business_services.json
├── frontend/
│   ├── index.html
│   └── static/acb-logo.png
├── tests/
│   ├── test_guardrails.py
│   ├── test_domain_agents.py
│   ├── test_router.py
│   └── test_tools.py
├── .gitignore
├── requirements.txt
└── README.md
```

Notes on the split:
- `agents/` holds only graph/LangGraph concerns (state, nodes, routing,
  compilation). `api/` holds only HTTP concerns (FastAPI routes, request
  schemas, the ElevenLabs voice glue). Neither should import internals
  from the other beyond `agents.graph.run_graph`, which is the one
  intended seam between them.
- Moving the JSON/CSV data files into `data/` means every `open(...)`
  call and `SOURCE_FILES` dict in `rag.py`/`tools.py`/`app.py` needs its
  path updated — do this as one pass, don't leave half the loaders
  pointing at the old root-level paths.
- `index.html`'s fetch calls hit `/chat`, `/tts`, etc. as relative URLs,
  so moving it into `frontend/` just needs `FileResponse` in the new
  `api/main.py` to point at the new path — no frontend code changes.

## Execution steps

1. **Delete dead code first, before moving anything.** Strip `app.py`
   down to FastAPI setup + the 5 live endpoints. Verify nothing else in
   the repo imports the functions/constants you're deleting (`grep -rn`
   for each name) before removing it.
2. **Deduplicate**: point `/calculate/*` at `tools.amortize`; delete the
   copy in `app.py`. Collapse the three guardrail regex copies into
   `guardrails.py`. Introduce `i18n.py` and migrate every language/message
   dict into it, updating every importer.
3. **Create the `src/acb` package** (add `__init__.py` files, an
   `acb.egg-info`-free `pyproject.toml` or just rely on running from `src/`
   with `PYTHONPATH=src` — pick whichever matches how this project is
   actually deployed; don't add packaging machinery nobody asked for).
4. **Move files into the target tree** module by module, fixing imports
   and file-path string literals (`"antigua_fees.json"` →
   `"data/antigua_fees.json"`, etc.) as you go. Do this incrementally and
   run the server after each move rather than moving everything and
   debugging imports at the end.
5. **Fix repo hygiene**: remove the stray `gitignore` file, `git rm
   --cached` the tracked `.DS_Store`/`.pyc`, add the SQLite checkpointer
   file to `.gitignore`.
6. **Add `tests/`** covering the guardrail regexes (PII/injection/
   jailbreak true/false cases), the router's fallback-to-`faq` behavior,
   and `amortize()`'s edge cases (zero rate, invalid inputs) — these are
   pure functions with no network calls, so they're cheap to test and
   currently have zero coverage.
7. **Update `README.md`** with the new layout and how to run the server
   (`uvicorn acb.api.main:app`, env vars needed, where data files live).

## Constraints

- Don't change agent *behavior* — prompts, routing logic, guardrail
  patterns, and the graph topology (`graph.py`'s node/edge wiring) stay
  functionally identical. This is a structure/dedup pass, not a rewrite
  of what the bot says or does.
- Don't introduce new dependencies (no ORMs, no DI frameworks, no
  settings-management libraries) to solve problems that a plain module of
  constants already solves.
- Keep commits small and re-testable: dead-code removal, dedup, and the
  file move are each independently revertible steps, not one giant diff.
