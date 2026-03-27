# arch-conscience

> An always-on AI agent that prevents architectural violations before they happen, by injecting your team's documented decisions into the coding agent's context at the moment code is being written.

---

## The problem

Engineering teams make architectural decisions for good reasons. They write them down in ADRs, Confluence pages, and Jira epics. Then six months later, an engineer (or their AI coding agent) opens a PR that reintroduces the exact approach the team explicitly rejected. Because neither the human nor the agent knew the decision existed.

In a world where AI generates nearly half of all production code, architectural drift isn't a documentation problem anymore. It's a systemic gap in the development loop. The coding agent has zero awareness of your team's decisions.

arch-conscience closes that gap.

---

## What it does

**Prevents violations at code-generation time.** AI coding agents (Claude Code, Cursor, Copilot) call arch-conscience via MCP before writing code. The agent sees your team's constraints, rejected alternatives, and active decisions, and refuses to generate code that violates them.

**Catches violations at PR time.** A webhook pipeline watches every PR, retrieves relevant ADRs, runs a two-stage detection pipeline, and alerts the responsible engineer via Telegram when a genuine contradiction is found.

**Drafts new ADRs from conversation.** When an engineer makes a significant architectural decision, the agent generates a structured ADR with Context, Decision, Consequences, and Rejected Alternatives sections, informed by existing decisions in the corpus.

**Knows what it doesn't know.** When a PR touches a service with no documented decisions, the system logs a corpus gap signal, surfacing blind spots in your architectural knowledge base.

---

## Demo: AI agent refuses to violate an ADR

```
You: I need to add authentication to the auth-service. I'm thinking of 
     using session cookies with a Redis-backed session store.

Claude Code: [calls get_architectural_context]

     I need to flag a conflict before proceeding. ADR-001 explicitly
     rejects session cookies for three reasons:

     1. Single point of failure: centralised session store
     2. Horizontal scaling: session state complicates scaling
     3. GDPR data residency: session data replication across regions

     The decided architecture is JWT (stateless). Want me to implement
     JWT-based auth instead, or draft an updated ADR if the original
     decision should be revisited?
```

The agent checked your team's decisions, found the conflict, and refused to write the violating code. All before a single line was generated.

---

## Quick start: Connect your coding agent

If you just want to try it against the live demo corpus:

```bash
# Claude Code
claude mcp add --transport http arch-conscience https://arch-conscience-production-e722.up.railway.app/mcp/

# Then in Claude Code, try:
# "Add session cookie auth to auth-service"
```

To run your own instance with your team's ADRs, see [Full Setup](#full-setup) below.

---

## How it works

```
AI coding agent (Claude Code / Cursor / Copilot)
       │
       │ MCP: get_architectural_context("auth-service")
       ▼
  arch-conscience MCP server
       │
       ▼
  Qdrant corpus: section-level ADR chunks
       │ filtered by service + status: active
       ▼
  Returns: decisions, rejected alternatives, constraints
       │
       ▼
  Agent generates compliant code (or refuses + explains why)


GitHub PR opened
       │
       ▼
  Webhook server (FastAPI)
       │
       ▼
  Router: extracts affected services, builds diff summary
       │
       ▼
  Qdrant corpus: hybrid search over ADRs
       │
       ▼
  Stage 1: relevance filter (gpt-4o-mini, cheap)
       │
       ▼
  Stage 2: gap detection (gpt-4o, chain-of-thought)
       │
       ▼
  Alert dispatched via Telegram
```

---

## MCP tools

arch-conscience exposes two tools via the Model Context Protocol:

### `get_architectural_context`

Call before writing or modifying code. Returns relevant ADR sections and conflict analysis.

```python
# "What rules apply to this service?"
get_architectural_context(service="auth-service")

# "Does my plan conflict with any decision?"
get_architectural_context(service="auth-service", approach="Use session cookies with Redis")

# "Any decisions about this anywhere?"
get_architectural_context(approach="Replace JWT with session cookies")

# "What's in the corpus?"
get_architectural_context()
```

When `approach` is provided, the response includes a verdict: `potential_conflict`, `review_recommended`, or `context_available`.

### `draft_adr`

Generates a structured ADR from natural conversation. Consults existing decisions in the corpus to cross-reference related ADRs.

```python
draft_adr(
    title="Use event sourcing for payment state",
    services="payments-service",
    context="Need audit compliance for 50k transactions/day...",
    approach="Append-only event store with materialized views",
    alternatives_considered="CRUD + audit log, third-party audit service...",
    constraint_type="compliance",
)
```

Returns a complete ADR in markdown with frontmatter, ready for team review.

---

## What makes it different

- **Prevents violations before code is written**, not just after. MCP integration means the agent knows your constraints at generation time.
- **Reasons over rejected alternatives.** Catches when a PR or agent reintroduces an approach the team explicitly ruled out.
- **Section-level retrieval.** ADRs are chunked by section (Context, Decision, Consequences, Rejected Alternatives), not by token count. The retriever surfaces the exact section most relevant to the change.
- **Two-stage detection pipeline.** Cheap model filters for relevance, expensive model reasons about contradictions. Keeps costs low and false positives down.
- **Corpus gap signals.** The system knows what it doesn't know, logging blind spots where services have no documented decisions.
- **ADR drafting.** Generates structured ADRs from conversation, informed by existing decisions in the corpus.

---

## Stack

- **Python 3.11** + **FastAPI** + **uvicorn**
- **MCP Python SDK** for Model Context Protocol (streamable HTTP + stdio)
- **LiteLLM** for unified LLM interface (model string routes to any provider)
- **Qdrant** for vector database with dense + BM25 sparse search
- **httpx** for async HTTP (GitHub API and Telegram)
- **pydantic-settings** for typed configuration with `.env` support
- Deployed on **Railway** (server) + **Qdrant Cloud** (vector store)

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

Fill in your API keys and tokens. See `.env.example` for all available options.

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
What was ruled out and why. This is the most important section.
```

Valid `constraint_type` values: `security`, `compliance`, `performance`, `scalability`, `data_model`, `operational`

### 5. Ingest the corpus

```bash
python -m scripts.run_ingest
```

### 6. Connect your coding agent

**Claude Code (remote, uses the deployed server):**
```bash
claude mcp add --transport http arch-conscience https://your-railway-url.up.railway.app/mcp/
```

**Claude Code (local, runs MCP server on your machine):**
```bash
claude mcp add-json arch-conscience '{"command":"/path/to/arch-conscience/.venv/bin/python","args":["-m","app.mcp_server"],"cwd":"/path/to/arch-conscience"}'
```

### 7. Start the webhook server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 3456
```

### 8. Register the GitHub webhook

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
# Config and validation tests
pytest tests/test_smoke.py -v

# Full pipeline + webhook + MCP tests
pytest tests/test_e2e.py -v
pytest tests/test_mcp.py -v
```

### Manual simulation

```bash
# Simulate a PR that violates ADR-001
python -m scripts.simulate_pr

# Test MCP tools interactively
mcp dev app/mcp_server.py
```

---

## What an alert looks like

When a PR contradicts a documented decision:

```
🔴 Architectural gap detected (adr-001)

PR #42 may reintroduce session cookies. ADR-001 requires
stateless JWT auth for GDPR compliance.

ADR-001 mandates JWT for stateless authentication due to security
and scalability concerns. Session cookies were rejected as they
introduce a centralised session store and conflict with GDPR
requirements. This PR contradicts that decision.

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
│   ├── router.py            GitHub payload to pipeline payload
│   ├── notify.py            Telegram alert dispatch
│   ├── ingest.py            ADR / Confluence / Jira ingestion
│   ├── gap_log.py           Corpus gap signal logger (JSONL)
│   ├── adr_drafter.py       LLM-powered ADR draft generator
│   ├── mcp_server.py        MCP server (get_architectural_context, draft_adr)
│   └── main.py              FastAPI server (webhook + MCP mount)
├── scripts/
│   ├── run_ingest.py        CLI: ingest ADRs into Qdrant
│   └── simulate_pr.py       CLI: simulate a PR through the pipeline
├── tests/
│   ├── test_smoke.py        Config + Qdrant connection tests
│   ├── test_e2e.py          Full pipeline + webhook tests
│   └── test_mcp.py          MCP tool tests
├── pyproject.toml
├── requirements.txt
├── Procfile                 Railway deployment
└── runtime.txt              Python version for Railway
```

---

## Architecture decisions

### Why MCP for agent integration
The Model Context Protocol is the standard for connecting AI coding agents to external tools. A single MCP server makes arch-conscience available to Claude Code, Cursor, Copilot, and any future agent that supports the protocol, without building separate integrations for each.

### Why two-stage detection pipeline
A single "does this violate any ADR?" prompt hallucinates. Stage 1 (cheap model) filters retrieved chunks for relevance before the expensive Stage 2 call. On a busy repo, most PRs have no relevant chunks and Stage 2 never fires.

### Why section-level ADR chunking
ADRs chunked at section level rather than by token count. The "Rejected Alternatives" section records why an approach was ruled out. Section-level chunking means the retriever surfaces the exact rejection rationale directly.

### Why no LangChain / LlamaIndex
The pipeline, confidence gate, section-level chunking, and hybrid retrieval are all custom enough that framework abstractions fight you rather than help. When Stage 2 misbehaves, you need to see the exact messages array.

### Why LiteLLM for provider abstraction
LiteLLM handles provider routing, response normalization, and retries without hiding the messages array. Switching from OpenAI to Anthropic is a one-line `.env` change.

### Why the false positive guard
Alert fatigue kills adoption faster than missed detections. Additive changes are never flagged. Only genuine contradictions fire alerts.

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

---

## Roadmap

- Auto-ingest: automatically re-ingest corpus when ADR files change
- ADR delivery: draft_adr opens a GitHub PR with the ADR file for team review
- GitHub App: one install covers the whole org, multi-repo support
- Post-merge conformance scanning: detect gradual architectural drift
- Decision lifecycle: detect when ADRs are being consistently violated and suggest superseding them
- Multi-source context for ADR drafting: gather context from Jira, Slack, PR discussions, and codebase analysis

---

## Background

Built as a portfolio project exploring agentic RAG systems, AI-native developer tools, and the Model Context Protocol. The self-healing corpus gap signal connects to ongoing research into automated detection and remediation of knowledge base degradation in production RAG pipelines.

---

## License

MIT. See [LICENSE](LICENSE)