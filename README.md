# arch-conscience

> An always-on AI agent that watches your GitHub PRs for changes that silently contradict documented architectural decisions — and alerts the responsible engineer before the code merges.

---

## The problem

Engineering teams make architectural decisions for good reasons. They write them down in ADRs, Confluence pages, and Jira epics. Then six months later, a new engineer opens a PR that reintroduces the exact approach the team explicitly rejected — because they never knew the decision existed.

Most architecture tooling focuses on visualisation, linting, or dependency tracking. None of them solve the *intent* problem: detecting when a code change violates a past decision and the reasoning behind it.

arch-conscience doesn't generate diagrams or lint code. It does one thing: watch for PRs that break architectural promises and tell the right engineer why, before the code merges.

---

## How it works

```
GitHub PR opened
       │
       ▼
  FastAPI webhook server (app/main.py)
       │
       ▼
  Router — extracts affected services, builds diff summary
       │
       ▼
  Qdrant corpus — hybrid search over ADRs, Confluence, Jira
       │ filtered by affected services + status: active
       ▼
  Stage 1 — relevance filter (gpt-4o-mini, cheap)
       │ drops chunks below relevance threshold
       ▼
  Stage 2 — gap detection (gpt-4o)
       │ chain-of-thought: summarise → match decisions → find contradiction
       ▼
  Alert dispatched via Telegram
```

The RAG corpus stores architectural decisions at **section level** — Context, Decision, Consequences, and Rejected Alternatives are separate chunks. This means when an engineer reintroduces a pattern that was explicitly rejected two years ago, the system retrieves that specific rejection rationale and includes it in the alert.

When retrieval returns no relevant decisions for a changed service, the event is logged as a **corpus gap signal** — a self-healing hook that surfaces undocumented architectural decisions to the team.

---

## What makes it different

- **Detects intent violations** — not just structural drift, but contradictions against documented reasoning
- **Reasons over rejected alternatives** — catches when a PR reintroduces an approach the team explicitly ruled out
- **Always-on proactive alerts** — engineers hear about the problem before the code merges, not after
- **Corpus gap signals** — the system knows what it *doesn't* know, and logs blind spots for future ADR authoring
- **Section-level retrieval** — surfaces the exact ADR section (Context, Decision, Consequences, Rejected Alternatives) most relevant to the change

---

## Stack

- **Python 3.11** + **FastAPI** + **uvicorn**
- **LiteLLM** — unified LLM interface (model string routes to any provider)
- **Qdrant** — vector database with dense + BM25 sparse search
- **httpx** — async HTTP for GitHub API and Telegram
- **pydantic-settings** — typed configuration with `.env` support
- Deployed on **Railway** (server) + **Qdrant Cloud** (vector store)

---

## Requirements

- Python 3.11+
- Docker Desktop (for local Qdrant)
- OpenAI API key with access to `gpt-4o` and `text-embedding-3-large`
- GitHub personal access token with `repo` scope
- A Telegram bot (takes 2 minutes — see setup step 4)

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/thaneeshshanand/arch-conscience.git
cd arch-conscience
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Start Qdrant

```bash
docker run -d -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant
```

Confirm it's running at `http://localhost:6333/dashboard`.

### 3. Configure environment

```bash
cp .env.example .env
```

Fill in your API keys and tokens. See `.env.example` for all available options.

The `SERVICE_MAP` tells the system which files belong to which service, so it can filter the corpus to relevant architectural decisions:

```bash
SERVICE_MAP={"services/auth":"auth-service","services/payments":"payments-service"}
```

### 4. Create a Telegram bot

1. Open Telegram and message `@BotFather`
2. Send `/newbot` and follow the prompts — you'll receive a token
3. Paste as `TELEGRAM_BOT_TOKEN` in your `.env`
4. Send any message to your new bot, then visit:
   ```
   https://api.telegram.org/bot<your_token>/getUpdates
   ```
5. Copy the `chat.id` value — this is your `TELEGRAM_CHAT_ID`

### 5. Write your ADRs

Create markdown files in `/adrs`. Each ADR section becomes a separate searchable chunk:

```markdown
---
id: adr-001
title: Use JWT for stateless authentication
status: active
date: 2024-03-15
services: [auth-service, api-gateway]
constraint_type: security
author: yourname
---

## Context
Why this decision was needed.

## Decision
What was decided.

## Consequences
Tradeoffs accepted.

## Rejected Alternatives
What was ruled out and why. This is the most important section —
it's what the system uses to detect when a discarded approach
is being reintroduced.
```

Valid `constraint_type` values: `security`, `compliance`, `performance`, `scalability`, `data_model`, `operational`

Valid `status` values: `active`, `superseded`, `proposed` — only `active` decisions are checked.

### 6. Ingest the corpus

```bash
python -m scripts.run_ingest
```

Expected output:

```
Scanning local ADR files...
Found 1 ADR files
Upserted 4 chunks

Done — adr:4 confluence:0 jira:0
```

Re-run whenever you add or update ADRs.

### 7. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3456
```

Expose it with ngrok for local development:

```bash
ngrok http 3456
```

### 8. Register the GitHub webhook

In your repo: **Settings → Webhooks → Add webhook**

```
Payload URL:   https://your-url (ngrok for local, Railway for prod)
Content type:  application/json
Secret:        <your GITHUB_WEBHOOK_SECRET>
Events:        Pull requests
Active:        ✅
```

---

## Testing

### Automated tests

```bash
# Config and validation tests
pytest tests/test_smoke.py -v

# Full pipeline simulation (mocked LLM, no API calls)
pytest tests/test_e2e.py -v
```

### Manual simulation

Simulate a PR that reintroduces session cookies against ADR-001:

```bash
python -m scripts.simulate_pr
```

A passing run:

```
Retrieved 4 chunks:
  [0.475] adr-001 — rejected_alternatives
  [0.436] adr-001 — context
  [0.428] adr-001 — consequences
  [0.387] adr-001 — decision

Stage 1: 4/4 chunks passed
Stage 2: gap=True confidence=1.0 severity=high

gap_detected:              True
confidence:                1.0
severity:                  high
violated_adr_id:           adr-001
rejected_alt_reintroduced: True
✅ Alert dispatched — check your Telegram bot!
```

---

## What an alert looks like

```
🔴 Architectural gap detected (adr-001)

PR #42 may reintroduce session cookies — ADR-001 requires
stateless JWT auth for GDPR compliance.

ADR-001 mandates JWT for stateless authentication due to security
and scalability concerns. Session cookies were rejected as they
introduce a centralised session store and conflict with GDPR
requirements. This PR contradicts that decision by implementing
session cookies. See doc_id: adr-001.

PR: https://github.com/org/repo/pull/42
Author: @engineer
Severity: high
⚠️ This approach was explicitly rejected in the ADR.
```

---

## Project structure

```
arch-conscience/
├── adrs/                    Local ADR markdown files
├── app/
│   ├── config.py            Pydantic settings + startup validation
│   ├── llm/
│   │   ├── base.py          Data types (Message, CompletionResult)
│   │   └── provider.py      LiteLLM wrapper (complete, embed)
│   ├── corpus.py            Qdrant wrapper (upsert, query, stats)
│   ├── detect.py            Two-stage gap detection pipeline
│   ├── router.py            GitHub payload → pipeline payload
│   ├── notify.py            Telegram alert dispatch
│   ├── ingest.py            ADR / Confluence / Jira ingestion
│   ├── gap_log.py           Corpus gap signal logger (JSONL)
│   └── main.py              FastAPI webhook server
├── scripts/
│   ├── run_ingest.py        CLI: ingest ADRs into Qdrant
│   └── simulate_pr.py       CLI: simulate a PR through the pipeline
├── tests/
│   ├── test_smoke.py        Config + Qdrant connection tests
│   └── test_e2e.py          Full pipeline + webhook tests
├── pyproject.toml
├── requirements.txt
├── Procfile                 Railway deployment
└── runtime.txt              Python version for Railway
```

---

## Deployment

Deployed on Railway with Qdrant Cloud:

1. Push to GitHub
2. Create a Railway service from the repo
3. Set environment variables in Railway dashboard
4. Generate a public domain under Settings → Networking
5. Update the GitHub webhook URL to the Railway domain

---

## Architecture decisions

### Why two-stage pipeline
A single "does this violate any ADR?" prompt hallucinates. Stage 1 (cheap model) filters retrieved chunks for relevance before the expensive Stage 2 call. On a busy repo, most PRs have no relevant chunks and Stage 2 never fires. Keeps costs low and false positives down.

### Why section-level ADR chunking
ADRs chunked at section level rather than by token count. The "Rejected Alternatives" section records why an approach was ruled out. Section-level chunking means the retriever surfaces the exact rejection rationale directly.

### Why no LangChain / LlamaIndex
The pipeline, confidence gate, section-level chunking, and hybrid retrieval are all custom enough that framework abstractions fight you rather than help. When Stage 2 misbehaves, you need to see the exact messages array. This is also the intellectual contribution — wrapping it in a framework obscures that.

### Why LiteLLM for provider abstraction
LiteLLM handles provider routing, response normalization, and retries without hiding the messages array. The model string *is* the provider selector — no separate config needed. Switching from OpenAI to Anthropic is a one-line `.env` change.

### Why hybrid search (dense + BM25)
Pure dense retrieval misses exact term matches. If an ADR says "must not use session cookies" and a PR introduces `Set-Cookie: session=...`, BM25 catches the lexical signal that semantic embeddings might soften.

### Why the false positive guard
Alert fatigue kills the product faster than missed detections. Additive changes are never flagged. Only genuine contradictions fire alerts. "When in doubt, default to no gap."

---

## Self-healing corpus gap signal

When a PR touches a service with no relevant active decisions in the corpus, arch-conscience logs a gap signal to `gap.log`:

```json
{"type":"no_chunks_found","services":["payments-service"],"pr_url":"...","ts":"..."}
```

This signals a blind spot — the team is making architectural decisions without documenting them. The gap log feeds a future self-healing pipeline that surfaces these gaps and prompts the team to write the missing ADRs.

---

## Confluence and Jira ingestion

Add these to your `.env` to ingest from Confluence and Jira:

```bash
CONFLUENCE_BASE_URL=https://yourorg.atlassian.net
CONFLUENCE_TOKEN=your_atlassian_api_token
CONFLUENCE_SPACE_KEY=ENG

JIRA_BASE_URL=https://yourorg.atlassian.net
JIRA_TOKEN=your_atlassian_api_token
```

Confluence pages labelled `architecture-decision` and Jira epics labelled `arch-decision` are ingested automatically.

---

## Roadmap

- Heartbeat: periodic re-scan of recently merged PRs missed by webhook
- Status endpoint: corpus stats, recent alerts, gap log summary
- Slack alert channel
- ADR authoring suggestions driven by gap log (self-healing)
- Multi-repo support
- BM25 sparse reranking in query pipeline

---

## Background

Built as a portfolio project exploring agentic RAG systems and always-on AI agents. The self-healing corpus gap signal connects to ongoing research into automated detection and remediation of knowledge base degradation in production RAG pipelines.

---

## License

MIT — see [LICENSE](LICENSE)
