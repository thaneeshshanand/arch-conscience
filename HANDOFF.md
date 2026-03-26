# arch-conscience — conversation handoff

## What this project is

arch-conscience is an always-on AI agent that watches GitHub PRs for changes that silently contradict documented architectural decisions (ADRs). When a PR reintroduces a rejected pattern or violates an active architectural constraint, it sends a proactive alert to the responsible engineer via Telegram before the code merges.

**Live repo:** https://github.com/thaneeshshanand/arch-conscience
**Deployed:** Railway (Node.js version, running and working)
**LinkedIn post:** published, getting engagement

---

## Current state

### Node.js version (complete, deployed)
Lives at `~/LEARN/AI/arch-conscience`. Fully working end-to-end:
- `src/server.js` — FastAPI-equivalent HTTP server, receives GitHub webhooks
- `src/router.js` — extracts affected services, builds diff summary
- `src/corpus.js` — Qdrant wrapper (embed, upsert, hybrid query)
- `src/detect.js` — two-stage gap detection pipeline (Stage 1: gpt-4o-mini relevance filter → confidence gate → Stage 2: gpt-4o chain-of-thought detection)
- `src/notify.js` — Telegram alert dispatch
- `src/ingest.js` — ADR/Confluence/Jira corpus ingestion
- `src/config.js` — env var validation
- `src/gap-log.js` — JSONL corpus gap signal logger

Infrastructure: Railway (server) + Qdrant Cloud (vector store, 4 chunks from ADR-001)

### Python rewrite (in progress)
Lives at `~/LEARN/AI/arch-conscience-py`. New folder, clean slate.
`app/config.py` is written. Nothing else yet.

**Stack:**
- FastAPI + uvicorn
- qdrant-client
- openai + anthropic SDKs
- pydantic-settings
- httpx
- pytest + pytest-asyncio
- Python 3.11.9 via pyenv

**Key addition vs Node.js:** LLM provider abstraction layer (`app/llm/`)

---

## Architecture decisions and their reasoning

### Why standalone Node.js → Python rewrite
Originally built as OpenClaw skill. OpenClaw's webhook auth (Bearer token) is incompatible with GitHub's webhook auth (HMAC signature). After attempting to bridge them, concluded OpenClaw adds complexity without proportional value for this product. Standalone agent is the right architecture. Python chosen because the AI/ML ecosystem (Ragas for RAG evaluation, sentence-transformers, future research paper) is Python-first.

### Why no LangChain / LlamaIndex / Haystack
The two-stage pipeline, confidence gate, section-level ADR chunking, and hybrid retrieval with service filtering are all custom enough that framework abstractions fight you rather than help. When Stage 2 prompt misbehaves, you need to see the exact messages array — LangChain hides that. Also: this is the intellectual contribution of the project. Wrapping it in a framework obscures that in interviews and research.

### Why two-stage pipeline
Single monolithic "does this violate any ADR?" prompt hallucinates. Stage 1 (cheap model) filters retrieved chunks for architectural relevance before the expensive Stage 2 call runs. On a busy repo, most PRs have no relevant chunks and Stage 2 never fires. This keeps costs low and false positives down.

### Why section-level ADR chunking
ADRs chunked at section level (Context / Decision / Consequences / Rejected Alternatives) rather than by token count. The "Rejected Alternatives" section is the most valuable — it records why an approach was ruled out. If you chunk by token count, that rationale gets buried. Section-level chunking means the retriever can surface the exact rejection rationale directly.

### Why `COMPLETION_PROVIDER` and `EMBEDDING_PROVIDER` are separate config fields
Anthropic has no embedding model. If you switch completions to Claude, you still need OpenAI for embeddings. The two concerns are independent — best completion model and best embedding model are chosen separately.

### Why `validate_required()` is explicit rather than in `__init__`
Allows instantiating Settings in tests with partial config without immediately throwing. `validate_required()` is called once in `main.py` at server startup.

### Why hybrid search (dense + BM25)
Pure dense retrieval misses exact term matches. If an ADR says "must not use session cookies" and a PR introduces `Set-Cookie: session=...`, BM25 catches the lexical signal that semantic embeddings might soften. Both signals combined reduces false negatives on retrieval.

### Why `score_threshold: 0.35` in Qdrant query
Drops low-quality matches before Stage 1 even sees them. Saves Stage 1 LLM calls on obviously irrelevant chunks.

### Why the false positive guard in Stage 2 prompt
Alert fatigue kills the product faster than missed detections. Additive changes (new endpoint alongside existing ones) are never flagged. Only genuine contradictions (replacing, removing, or bypassing an existing constraint) fire alerts. "When in doubt, default to no gap."

### Why `rejected_alt_reintroduced` is a separate field
The differentiator from every existing tool. When an ADR explicitly documents a rejected alternative and the PR reimplements it, this field flips true and escalates severity to high automatically. ArchGuard and Eraserbot completely miss this case.

### Why corpus gap signal exists
When retrieval finds no relevant decisions for a changed service, the system logs this as a gap signal. This is the Self-Healing RAG connection — the system knows what it doesn't know. The gap log is input to a future pipeline that surfaces undocumented architectural decisions and prompts teams to write missing ADRs.

### Deployment decisions
- Railway chosen over Render (sleeps on free tier), Fly.io (requires Docker), VPS (overkill), AWS/GCP/Azure (too complex for current stage)
- Qdrant Cloud (free tier) over local Docker — always-on, no infrastructure management
- ngrok rejected as permanent solution (URL changes on restart), Railway gives stable URL

---

## LLM provider abstraction — design

```
app/llm/
├── __init__.py
├── base.py           # abstract LLMProvider with complete() and embed()
├── openai_provider.py
├── anthropic_provider.py
└── factory.py        # get_llm_provider() and get_embedding_provider()
```

`base.py` defines:
- `Message` dataclass (role, content)
- `CompletionResult` dataclass (content, model)
- `LLMProvider` abstract class with `complete()` and `embed()` abstract methods

`factory.py` reads `COMPLETION_PROVIDER` and `EMBEDDING_PROVIDER` from settings and returns the right implementation. Used via dependency injection in `detect.py` and `corpus.py`.

---

## Stage 2 system prompt

The full prompt is in `src/detect.js` (Node.js) — copy verbatim to Python `detect.py`. Key elements:
- Explicit reasoning protocol: Step 1 summarise → Step 2 filter active → Step 3 match → Step 4 false positive guard → Step 5 severity → Step 6 JSON output
- `temperature=0` mandatory (deterministic task)
- Structured JSON output only, no prose
- Hard rules: never flag additive changes, never flag superseded decisions, never reason from code quality, confidence < 0.7 → gap_detected = false

---

## Output schema (Stage 2)

```json
{
  "gap_detected": boolean,
  "confidence": float,
  "severity": "low|medium|high|null",
  "violated_adr_id": "string|null",
  "constraint_type": "string|null",
  "rejected_alt_reintroduced": boolean,
  "change_summary": "string",
  "reasoning": "string",
  "alert_headline": "string|null",
  "alert_body": "string|null",
  "corpus_gap_signal": boolean
}
```

---

## ADR format

```markdown
---
id: adr-001
title: Use JWT for stateless authentication
status: active
date: 2024-03-15
services: [auth-service, api-gateway]
constraint_type: security
author: thaneesh
---

## Context
## Decision
## Consequences
## Rejected Alternatives
```

Valid `constraint_type` values: `security`, `compliance`, `performance`, `scalability`, `data_model`, `operational`
Valid `status` values: `active`, `superseded`, `proposed` — only `active` checked at detection time

---

## Environment variables

```bash
COMPLETION_PROVIDER=openai          # openai | anthropic
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...        # only if using Anthropic
STAGE1_MODEL=gpt-4o-mini
STAGE2_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=...
QDRANT_URL=https://...cloud.qdrant.io
QDRANT_API_KEY=...
QDRANT_COLLECTION=arch_decisions
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
CONFIDENCE_THRESHOLD=0.7
STAGE1_THRESHOLD=0.5
ALERT_CHANNEL=telegram
SERVICE_MAP={"services/auth":"auth-service","services/payments":"payments-service"}
WEBHOOK_PORT=3456
```

---

## Next steps (in order)

1. `app/llm/base.py` — abstract interface
2. `app/llm/openai_provider.py` — OpenAI implementation
3. `app/llm/anthropic_provider.py` — Anthropic implementation
4. `app/llm/factory.py` — provider factory
5. `app/corpus.py` — Qdrant wrapper
6. `app/gap_log.py` — JSONL gap logger
7. `app/router.py` — GitHub payload → pipeline payload
8. `app/detect.py` — two-stage pipeline
9. `app/notify.py` — alert dispatch
10. `app/ingest.py` — ADR/Confluence/Jira ingestion
11. `app/main.py` — FastAPI webhook server
12. `tests/test_smoke.py` — config + Qdrant connection
13. `tests/test_e2e.py` — full pipeline simulation
14. Update `Procfile`, deploy to Railway, update GitHub webhook URL

---

## How to start a new conversation

Paste this document and say:

"I am rewriting arch-conscience from Node.js to Python. The Node.js version is complete and deployed. The Python version is in `~/LEARN/AI/arch-conscience-py`. `app/config.py` is complete. We are ready to write `app/llm/base.py` next. Please read the handoff document above and continue."