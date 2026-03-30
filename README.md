# arch-conscience

> An always-on AI agent that prevents architectural violations before they happen, by injecting your team's documented decisions into the coding agent's context at the moment code is being written.

---

## The problem

Engineering teams make architectural decisions for good reasons. They write them down in ADRs, Confluence pages, design docs, and rules files. Then six months later, an engineer (or their AI coding agent) opens a PR that reintroduces the exact approach the team explicitly rejected. Because neither the human nor the agent knew the decision existed.

In a world where AI generates nearly half of all production code, architectural drift isn't a documentation problem anymore. It's a systemic gap in the development loop. The coding agent has zero awareness of your team's decisions.

arch-conscience closes that gap.

---

## What it does

**Prevents violations at code-generation time.** AI coding agents (Claude Code, Cursor, Copilot) call arch-conscience via MCP before writing code. The agent's behavior adapts to the type of knowledge being violated — hard constraints trigger an outright refusal, decisions with rejected alternatives trigger a refusal with explanation, and principles flag a deviation but let the developer proceed.

**Catches violations at PR time.** A webhook pipeline watches every PR, retrieves relevant knowledge items, runs a two-stage detection pipeline with knowledge-type-aware reasoning, and alerts the responsible engineer via Telegram when a genuine contradiction is found.

**Ingests your existing documentation.** arch-conscience meets teams where they are. Point it at an ADR directory, a CLAUDE.md or .cursorrules file, a Confluence page, an RFC, or any other document where your team has written down an architectural thought. A two-pass LLM pipeline extracts the decisions, constraints, and principles automatically — no reformatting required.

**Drafts new ADRs from conversation.** When an engineer makes a significant architectural decision, the agent generates a structured ADR with Context, Decision, Consequences, and Rejected Alternatives sections, informed by existing decisions in the corpus.

**Knows what it doesn't know.** When a PR touches a service with no documented decisions, the system logs a corpus gap signal, surfacing blind spots in your architectural knowledge base.

---

## Knowledge types — how the agent reasons

arch-conscience distinguishes three kinds of architectural knowledge. Each drives different agent behavior:

| Type | Definition | Agent behavior | Detection severity |
|------|-----------|----------------|-------------------|
| **Decision** | The team chose X over Y | Refuse + explain why the alternative was rejected | By domain (security/compliance → high) |
| **Constraint** | X must always be true | Hard stop — refuse unconditionally | Always high |
| **Principle** | The team prefers X approach | Proceed, but flag the deviation | Low (confidence capped at 0.7) |

This taxonomy lets the agent escalate correctly without crying wolf. A coding style deviation is a low-confidence flag. A constraint violation is a hard stop, full stop.

---

## Demo: AI agent refuses to violate a constraint

> This demo requires a constraint to be ingested. The sample below uses a rule extracted from a CLAUDE.md: `"All external API calls must go through the API gateway. Services must not call third-party APIs directly."` Run `ingest_document` on your team's rules file to unlock this behavior.

```
You: Add a feature to call the Stripe API directly from the
     payments worker.

Claude Code: [calls get_architectural_context(service="payments-worker",
              approach="Direct Stripe API call from payments worker")]

     Hard stop. This violates an active constraint:

     "All external API calls must go through the API gateway.
      Services must not call third-party APIs directly."

     This is a compliance constraint — it was put in place for
     centralized rate limiting, audit logging, and secrets management.
     Bypassing the gateway means Stripe credentials and call volume
     would be invisible to monitoring.

     I won't generate this code. If the constraint should be revisited,
     use draft_adr to propose a change and route it through the team.
```

---

## Demo: AI agent refuses to reintroduce a rejected alternative

```
You: I need to add authentication to the auth-service. I'm thinking
     of using session cookies with a Redis-backed session store.

Claude Code: [calls get_architectural_context(service="auth-service",
              approach="Use session cookies with Redis")]

     I need to flag a conflict before proceeding. ADR-001 explicitly
     rejected session cookies for three reasons:

     1. Single point of failure — centralised session store
     2. Horizontal scaling — session state complicates auto-scaling
     3. GDPR data minimisation — retaining session data server-side

     The decided architecture is JWT (stateless). Want me to implement
     JWT-based auth instead, or draft an updated ADR if the original
     decision should be revisited?
```

---

## Demo: Principle deviation — flagged but allowed

```
You: Use a direct database query in this handler instead of going
     through the repository layer.

Claude Code: [calls get_architectural_context(service="api",
              approach="Direct DB query bypassing repository layer")]

     Note: This deviates from a standing principle — the team prefers
     all data access to go through the repository layer for testability
     and consistency. There are known exceptions for reporting queries.

     I'll write the direct query as requested, but flagging this so you
     can decide whether it warrants an exception note in the ADR or a
     TODO to refactor.
```

---

## Quick start: Connect your coding agent

Try it against the live demo corpus:

```bash
# Claude Code
claude mcp add --transport http arch-conscience \
  https://arch-conscience-production-e722.up.railway.app/mcp/

# Then ask Claude Code:
# "Add session cookie auth to auth-service"
# "Call the Stripe API directly from the payments worker"
```

To run your own instance, see [Full Setup](#full-setup) below.

---

## How it works

```
AI coding agent (Claude Code / Cursor / Copilot)
       │
       │ MCP: get_architectural_context("auth-service", "session cookies")
       ▼
  arch-conscience MCP server
       │
       ▼
  Qdrant corpus: section-level knowledge chunks
       │ filtered by service + status: active
       ▼
  Returns: decisions + rejected alternatives, constraints, principles
       │
       ▼
  Agent behavior adapts to knowledge type:
  constraint → hard stop
  decision   → refuse + explain rejected alternative
  principle  → flag deviation, allow proceed


GitHub PR opened
       │
       ▼
  Webhook server (FastAPI)
       │
       ▼
  Router: extracts affected services, builds diff summary
       │
       ▼
  Qdrant corpus: semantic search over knowledge base
       │
       ▼
  Stage 1: relevance filter (gpt-4o-mini, cheap)
       │ knowledge-type-aware scoring
       ▼
  Stage 2: gap detection (gpt-4o, chain-of-thought)
       │ classifies: Violating / Contradicting / Reintroducing / Deviating
       ▼
  Alert dispatched via Telegram
```

---

## MCP tools

arch-conscience exposes four tools via the Model Context Protocol:

### `get_architectural_context`

Call before writing or modifying code. Returns relevant knowledge items and conflict analysis. Behavior adapts based on inputs:

```python
# "What rules apply to this service?"
get_architectural_context(service="auth-service")

# "Does my plan conflict with anything?"
get_architectural_context(
    service="auth-service",
    approach="Use session cookies with Redis"
)

# "Any decisions about this anywhere?"
get_architectural_context(approach="Replace JWT with session cookies")

# "What's in the corpus?"
get_architectural_context()
```

When `approach` is provided, the response includes a verdict: `potential_conflict`, `review_recommended`, or `context_available`, plus per-type behavior guidance.

### `draft_adr`

Generates a structured ADR from natural conversation. Queries the corpus for related decisions to cross-reference:

```python
draft_adr(
    title="Use event sourcing for payment state",
    services="payments-service",
    context="Need audit compliance for 50k transactions/day...",
    approach="Append-only event store with materialized views",
    alternatives_considered="CRUD + audit log, third-party audit service...",
    constraint_type="compliance",
    author="yourname",
)
```

Returns complete ADR markdown with frontmatter, ready for team review. Once approved, save it to the `/adrs` directory and run `python scripts/run_ingest.py` to add it to the corpus.

### `ingest_document`

Ingests any document into the corpus. Auto-detects format and routes to the appropriate handler:

| Input | Handler | LLM calls |
|-------|---------|-----------|
| ADR with YAML frontmatter | Regex parser | 0 |
| Known rules file (CLAUDE.md, .cursorrules, AGENTS.md…) | Rules bridge | 1 |
| Everything else | Two-pass normalizer | 1 + N |

```python
# Ingest a Confluence page, design doc, RFC, or any document
ingest_document(
    content="<document text or HTML>",
    filename="payments-rfc.md",          # optional — used for format detection
    source_url="https://wiki.example.com/page/123",  # optional — provenance
    source_type="rfc",                   # optional — provenance label
)
```

The response includes extracted items, chunk count, and any conflicts with existing corpus items (with resolution suggestions).

### `update_item_status`

Updates the status of a knowledge item — all chunks matching the `doc_id` in one call. Used for conflict resolution and lifecycle management:

```python
# Supersede an older item when a newer one takes precedence
update_item_status(
    doc_id="adr-002",
    new_status="superseded",
    reason="Replaced by adr-007 which expands scope to all services",
)

# Retire a principle that no longer applies
update_item_status(doc_id="norm-design-doc-md-3", new_status="deprecated")
```

Valid statuses: `active`, `proposed`, `superseded`, `deprecated`.

---

## Format-agnostic ingestion

arch-conscience meets teams where they are. You don't need ADRs in a specific format — anything your team has ever written down is fair game.

### What gets ingested

The two-pass LLM normalizer processes any document through two stages:

1. **Pass 1 — Discovery.** The full document is sent to the LLM, which returns a manifest of all knowledge items found: titles, types, and which sections contain the relevant information.

2. **Pass 2 — Focused extraction.** Each discovered item gets a targeted extraction call with a type-specific prompt (decision vs. constraint vs. principle). Runs in parallel (up to 5 concurrent). Includes validation and one automatic retry.

### What comes out

Each extracted item produces section-level chunks: context, decision/constraint/principle, consequences, and (for decisions) rejected alternatives. Section-level chunking means the retriever surfaces the exact section most relevant to the incoming code change — including the rejected alternatives that catch "but what if we tried X again?" moments.

### Preprocessing

Before extraction, raw content passes through a preprocessor:
- HTML → clean markdown (tables preserved as markdown tables, images → `[Image: alt_text]` placeholders)
- Encoding normalization (BOM, CRLF)
- Headingless documents get synthetic section headings so Pass 1 can reference them

---

## Corpus health

After bulk ingestion, run the health check to surface any overlapping items that may need resolution:

```bash
python scripts/check_health.py
```

Output shows items with overlapping `(domain, services)` coverage, sorted by date, with resolution suggestions:

```
⚠️  Overlap area: domain=security, services=auth-service
   2 items govern this area:
   - [adr-005] Use OAuth2 for third-party auth (2025-08-01) [decision]
   - [adr-001] Use JWT for stateless authentication (2024-03-15) [decision]
   → Suggestion: 'adr-005' is newest. Consider superseding older items:
     update_item_status(doc_id='adr-001', new_status='superseded')

✅ No other overlaps found.
```

---

## Full setup

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

### 3. Configure environment

```bash
cp .env.example .env
```

Minimum required variables:

```bash
OPENAI_API_KEY=sk-...
GITHUB_TOKEN=ghp_...
GITHUB_WEBHOOK_SECRET=...
QDRANT_URL=http://localhost:6333
```

See `.env.example` for all options including Telegram alerts, Confluence/Jira ingestion, and model selection.

### 4. Write your ADRs

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
What was ruled out and why. This section drives "refused and explained" behavior in coding agents.
```

Valid `constraint_type` values: `security`, `compliance`, `performance`, `scalability`, `data_model`, `operational`

### 5. Ingest the corpus

```bash
# Local ADR files + Confluence/Jira if configured
python scripts/run_ingest.py
```

Or ingest any document via the MCP tool from your coding agent:

```
Call ingest_document with the content of your Confluence page, design doc, or RFC.
```

### 6. Check corpus health (after bulk ingestion)

```bash
python scripts/check_health.py
```

### 7. Connect your coding agent

**Claude Code (remote — uses the deployed server):**
```bash
claude mcp add --transport http arch-conscience \
  https://your-railway-url.up.railway.app/mcp/
```

**Claude Code (local — runs MCP server on your machine):**
```bash
claude mcp add-json arch-conscience '{
  "command": "/path/to/.venv/bin/python",
  "args": ["-m", "app.mcp_server"],
  "cwd": "/path/to/arch-conscience"
}'
```

### 8. Start the webhook server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3456
```

### 9. Register the GitHub webhook

In your repo: **Settings > Webhooks > Add webhook**

```
Payload URL:   https://your-url/
Content type:  application/json
Secret:        <your GITHUB_WEBHOOK_SECRET>
Events:        Pull requests
Active:        ✅
```

---

## Testing

```bash
# All tests
pytest

# By module
pytest tests/test_smoke.py -v          # config + Qdrant connection (10 tests)
pytest tests/test_e2e.py -v            # pipeline + webhook + notifications (15 tests)
pytest tests/test_mcp.py -v            # MCP tools + format detection (24 tests)
pytest tests/test_rules_bridge.py -v   # rules extraction (11 tests)
pytest tests/test_preprocess.py -v     # HTML conversion + preprocessing (19 tests)
pytest tests/test_extract.py -v        # two-pass extraction pipeline (16 tests)
```

### Manual simulation

```bash
# Simulate a PR that reintroduces session cookies against ADR-001
python scripts/simulate_pr.py

# Interactive MCP tool testing
mcp dev app/mcp_server.py
```

---

## What a PR alert looks like

```
🔴 Architectural gap detected (adr-001)

PR #42 may reintroduce session cookies — ADR-001 requires
stateless JWT auth for security.

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

## Stack

- **Python 3.11** + **FastAPI** + **uvicorn**
- **MCP Python SDK** — streamable HTTP transport + stdio for local use
- **LiteLLM 1.82.6** — unified LLM interface (pinned; 1.82.7/1.82.8 compromised in supply chain attack 2026-03-24)
- **Qdrant** — vector database with dense + BM25 sparse search configured
- **markdownify** — HTML→markdown conversion with table preservation
- **httpx** — async HTTP (GitHub API, Telegram)
- **pydantic-settings** — typed config with `.env` support
- Deployed on **Railway** + **Qdrant Cloud**

---

## Project structure

```
arch-conscience/
├── adrs/
│   └── adr-001.md               Sample ADR (JWT for stateless auth)
├── app/
│   ├── config.py                Pydantic settings + startup validation
│   ├── llm/
│   │   ├── base.py              Data types (Message, CompletionResult)
│   │   └── provider.py          LiteLLM wrapper (complete, embed)
│   ├── corpus.py                Qdrant wrapper (upsert, query, conflict
│   │                              resolution, update_payload, find_overlapping)
│   ├── detect.py                Two-stage knowledge-type-aware detection pipeline
│   ├── router.py                GitHub payload → PipelinePayload
│   ├── notify.py                Telegram alert dispatch
│   ├── ingest.py                ADR / Confluence / Jira ingestion
│   ├── gap_log.py               Corpus gap signal logger (JSONL)
│   ├── adr_drafter.py           LLM-powered ADR draft generator
│   ├── rules_bridge.py          Extract decisions from rules files
│   ├── extract.py               Two-pass LLM extraction pipeline
│   ├── preprocess.py            HTML→markdown, encoding, synthetic headings
│   ├── format_detect.py         Document format detection and routing
│   ├── mcp_server.py            MCP server (4 tools + 1 resource)
│   └── main.py                  FastAPI server (webhook + MCP mount)
├── scripts/
│   ├── run_ingest.py            CLI: ingest ADRs into Qdrant
│   ├── simulate_pr.py           CLI: simulate PR detection
│   ├── ingest_rules.py          CLI: extract decisions from rules files
│   └── check_health.py          CLI: corpus health check for overlaps
├── tests/
│   ├── conftest.py              Shared fixtures
│   ├── test_smoke.py            Config + Qdrant connection
│   ├── test_e2e.py              Full pipeline + webhook + resolution
│   ├── test_mcp.py              MCP tools + format detection
│   ├── test_rules_bridge.py     Rules file extraction
│   ├── test_preprocess.py       HTML conversion + preprocessing
│   └── test_extract.py          Two-pass extraction pipeline
├── pyproject.toml
├── requirements.txt
├── Procfile                     Railway: web: uvicorn app.main:app
└── runtime.txt                  Python 3.11.9
```

---

## Architecture decisions

### Why three knowledge types instead of one

Constraints, decisions, and principles warrant fundamentally different responses — a constraint violation is a hard stop, a decision contradiction should explain *why* the alternative was rejected, and a principle deviation is worth flagging but not blocking. Collapsing all three into a single "violated/not violated" signal loses that nuance and makes the agent either too aggressive or too permissive.

### Why MCP for agent integration

The Model Context Protocol is the standard for connecting AI coding agents to external tools. A single MCP server makes arch-conscience available to Claude Code, Cursor, Copilot, and any future agent that supports the protocol, without building separate integrations for each.

### Why two-stage detection pipeline

A single "does this violate any ADR?" prompt hallucinates. Stage 1 (cheap model) filters retrieved chunks for relevance before the expensive Stage 2 call. On a busy repo, most PRs have no relevant chunks and Stage 2 never fires.

### Why section-level chunking

ADRs and normalizer output are chunked by section, not by token count. The "Rejected Alternatives" section records why an approach was ruled out. Section-level chunking means the retriever surfaces the exact rejection rationale — directly enabling "I see you're reintroducing X, which was explicitly rejected because Y" responses.

### Why two-pass extraction instead of one LLM call

Pass 1 produces a manifest (titles, types, relevant sections) without extracting full content. Pass 2 uses that manifest as a targeting guide, with type-specific prompts per item. This avoids the "kitchen sink" problem where a single extraction call produces a large undifferentiated blob. Pass 2 runs in parallel under a semaphore (5 concurrent at a time), so a 20-item document processes in four batches rather than sequentially.

### Why the false positive guard

Alert fatigue kills adoption faster than missed detections. Additive changes are never flagged. The Stage 2 prompt includes an explicit "false positive guard" step — does the change override an existing constraint, or does it add something new alongside existing patterns? If additive, no gap.

### Why no LangChain / LlamaIndex

The pipeline, confidence gate, section-level chunking, and query-time conflict resolution are all custom enough that framework abstractions fight you rather than help. When Stage 2 misbehaves, you need to see the exact messages array.

---

## Confluence and Jira ingestion

Confluence pages must carry the `architecture-decision` label and Jira epics must carry the `arch-decision` label to be picked up during ingestion. Add these to your `.env`:

```bash
CONFLUENCE_BASE_URL=https://yourorg.atlassian.net
CONFLUENCE_TOKEN=your_atlassian_api_token
CONFLUENCE_SPACE_KEY=ENG          # space to scan

JIRA_BASE_URL=https://yourorg.atlassian.net
JIRA_TOKEN=your_atlassian_api_token
```

Then run `python scripts/run_ingest.py` — it will pick up ADRs, Confluence pages, and Jira epics in one pass. For one-off documents (design docs, RFCs, Notion exports), use `ingest_document` via the MCP tool instead.

---

## Roadmap

- **Notion integration** — ingest from Notion workspaces
- **Large document splitting** — split on top-level headings for docs exceeding context window
- **Auto-ingest** — re-ingest corpus when source documents change
- **Multi-tenancy** — per-tenant corpus isolation, API key auth on MCP server
- **GitHub App** — one install covers the whole org, multi-repo support
- **Slack channel routing** — per-team alert routing using the `owners` field
- **Post-merge conformance scanning** — detect gradual architectural drift
- **Decision lifecycle management** — surface items that are consistently violated, suggest superseding

---

## Background

Built as a portfolio project exploring agentic RAG systems, AI-native developer tools, and the Model Context Protocol. The evolution from "ADR enforcement tool" to "architectural knowledge layer" reflects a broader realization: the addressable market for "teams that write ADRs" is tiny. The market for "engineering teams that have ever written down an architectural thought anywhere" is every engineering team.

---

## License

MIT. See [LICENSE](LICENSE)