# Try arch-conscience in 2 minutes

No installation. No Python. No database. Just your coding agent.

---

## Prerequisites

- [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed (any language/stack)

---

## Path 1: Live demo (2 minutes, no setup)

Use the pre-loaded demo corpus to see how arch-conscience works. No data leaves your machine — you're querying decisions we've already ingested.

### Connect to the live server

```bash
claude mcp add --transport http arch-conscience \
  https://arch-conscience-production-e722.up.railway.app/mcp/
```

### Provoke a violation

The demo corpus contains ADR-001: "Use JWT for stateless authentication." Session cookies were explicitly rejected. Try this:

```
I'm working on auth-service. Add session cookie authentication with a Redis session store.
```

**Watch what happens.** Claude Code calls `get_architectural_context`, finds the conflict with ADR-001, and flags it *before writing any code* — explaining that session cookies were rejected due to single point of failure, horizontal scaling issues, and GDPR conflicts. It then offers the decided approach (JWT) as the alternative.

### Try another behavior

The demo corpus also contains a constraint extracted from a rules file: all external API calls must go through the API gateway. Try this:

```
I'm working on payments-service. Add a function that calls the Stripe API directly.
```

Notice how both responses flag the conflict *before writing any code* — but the content is different. The first response explains *why* the alternative was rejected (scaling, GDPR, sticky sessions) and points to the decided approach (JWT). The second cites a hard constraint and explains the correct pattern (route through the gateway). The agent retrieves the specific knowledge that applies and adapts its explanation accordingly.

---

## Path 2: Your own codebase (15 minutes, data stays local)

Evaluate against a real application with real architectural decisions. Everything runs on your machine — nothing is sent to a third-party server.

This walkthrough uses the [conduit realworld app](https://github.com/TonyMckes/conduit-realworld-example-app) (React + Express + Sequelize + PostgreSQL) as an example. You can substitute your own codebase and rules.

### 1. Clone the test app

```bash
git clone https://github.com/TonyMckes/conduit-realworld-example-app.git
cd conduit-realworld-example-app
```

### 2. Set up arch-conscience

```bash
git clone https://github.com/thaneeshshanand/arch-conscience.git
cd arch-conscience
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 3. Start Qdrant

```bash
docker run -d -p 6333:6333 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  --name qdrant qdrant/qdrant
```

### 4. Configure

```bash
cp .env.example .env
```

Edit `.env` with your keys:

```bash
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
GITHUB_TOKEN=ghp-placeholder
GITHUB_WEBHOOK_SECRET=placeholder
```

> `GITHUB_TOKEN` and `GITHUB_WEBHOOK_SECRET` are required by config validation. Use placeholder values if you're only evaluating MCP — they're not used until you set up the webhook.

### 5. Write your architectural rules

Save this as `CLAUDE.md` somewhere **outside** the conduit repo (e.g. `/tmp/conduit-rules/CLAUDE.md`). Keeping it outside the repo is important — it forces Claude Code to discover the rules through the MCP tool rather than reading them from the project directory.

```markdown
# Conduit — Architecture Rules

## Backend Stack
Express.js 5 with Sequelize ORM and PostgreSQL. We evaluated Prisma
but rejected it — Sequelize has better raw query support and the team
has established migration workflows. We also considered NestJS but
rejected it as over-engineered for this application size.

## Authentication
JWT-based auth with bcrypt password hashing. We evaluated Passport.js
but rejected it — unnecessary abstraction for a single auth strategy.
Never implement session-based auth — the API is stateless.

## Data Access
Controllers call Sequelize models directly — no service layer. We
evaluated adding a service layer but rejected it as unnecessary
indirection for this application's complexity.

## Constraints
- PostgreSQL is the only supported database. Do not add SQLite, MySQL,
  or in-memory database support.
- All schema changes must go through Sequelize migrations. Never modify
  the database by hand or via model sync. Migrations must be
  backward-compatible.
- Authentication middleware must run before any route that accesses
  user data. Routes must not parse tokens themselves.

## Principles
- Keep controllers thin — validate input, call model methods, format
  responses. Complex query logic belongs in Sequelize scopes or model
  class methods.
- Prefer Sequelize query methods over raw SQL. Raw queries are
  acceptable only for complex aggregations that can't be expressed
  efficiently with Sequelize's query builder.
- Error handling goes through the centralized error handler middleware.
  Controllers should throw errors, not catch and format them.
```

> **Use your own rules instead.** If your team has a CLAUDE.md, .cursorrules, design doc, or any document with architectural decisions — use that with your own codebase. The system handles any format.

### 6. Ingest the rules

```bash
cd /path/to/arch-conscience
python scripts/ingest_rules.py --file test-files/conduit-app-CLAUDE.md
```

You'll see output like:

```
Extracted 13 items, indexed 28 chunks.
By type: 4 constraint(s), 5 decision(s), 4 principle(s)
```

All three knowledge types extracted from a single rules file.

### 7. Connect Claude Code from the conduit repo

```bash
cd /path/to/conduit-realworld-example-app
claude mcp add-json arch-conscience \
  "{\"command\": \"/absolute/path/to/arch-conscience/.venv/bin/python\", \"args\": [\"-m\", \"app.mcp_server\"], \"cwd\": \"/absolute/path/to/arch-conscience\"}"
```

> Replace `/absolute/path/to/arch-conscience` with the actual path where you cloned arch-conscience.

### 8. Test three knowledge types

Start Claude Code inside the conduit repo:

```bash
claude
```

Try these one at a time. Each triggers a different behavior:

**Decision — rejected alternative:**
```
Before writing any code, check get_architectural_context. Then replace Sequelize with Prisma for the User and Article models.
```

Expected: Claude Code finds that Prisma was explicitly evaluated and rejected. It flags the conflict, explains why Prisma was rejected, and offers to draft an ADR if the team wants to revisit the decision.

**Constraint — hard rule:**
```
Before writing any code, check get_architectural_context. Then add SQLite support as an alternative to PostgreSQL for local development.
```

Expected: "Hard stop." SQLite is called out by name as forbidden. Claude Code suggests the compliant alternative (Docker with Postgres) instead, and only offers to override via a formal ADR process.

**Principle — flag but proceed:**
```
Before writing any code, check get_architectural_context. Then add a raw SQL query in the articles controller to fetch trending articles with a complex join.
```

Expected: Claude Code reads the exception clause ("raw queries acceptable for complex aggregations"), agrees the trending query qualifies, but catches that putting it in the controller violates the thin-controller principle. It offers to write the code the right way — raw SQL inside a model method, called from the controller.

### What you just saw

Three visibly different agent behaviors from the same system:

| Knowledge type | What happened | Agent behavior |
|---|---|---|
| **Decision** | Prisma was rejected | Flagged conflict, explained rejection reasons, offered ADR |
| **Constraint** | SQLite is forbidden | Hard stop, offered compliant alternative |
| **Principle** | Raw SQL has exceptions | Acknowledged the exception, caught the *real* violation, wrote compliant code |

The agent didn't just pattern-match on keywords. It read the exception clauses, cross-referenced multiple rules, and adapted its response based on knowledge type.

---

## Bonus: Draft an ADR

Once you've seen enforcement in action (either path), try having Claude Code document a new decision:

```
We just decided to use event sourcing for the payments service
because we need a full audit trail for compliance. We considered
CRUD with an audit log table but rejected it because it can't
reconstruct historical state. Draft an ADR for this.
```

Claude Code calls `draft_adr`, queries the corpus for related decisions, and generates a structured ADR with Context, Decision, Consequences, and Rejected Alternatives — ready for team review.

---

## What just happened

1. Your rules file was parsed by an LLM pipeline that extracted structured decisions, constraints, and principles
2. Each item was indexed as section-level chunks in your local Qdrant instance
3. When you asked for violating code in a fresh session, Claude Code called the MCP tool to check the corpus *before* generating anything
4. The agent's behavior adapted based on knowledge type — citing rejection reasons, explaining constraints, or flagging deviations

This is what happens on every code generation when arch-conscience is connected. It also runs at PR time, catching violations that slip through and alerting your team via Slack or Telegram.

---

## How it works (30-second version)

```
You: "Write code that does X"
         │
Claude Code calls get_architectural_context()
         │
         ▼
arch-conscience searches your team's knowledge base
         │
         ▼
Returns: relevant decisions, constraints, principles
         │
         ▼
Claude Code adapts its behavior:
  constraint violated  → flags + explains correct pattern
  rejected alt found   → flags + explains why it was rejected
  principle deviation  → flags it, writes the code
```

---

## Next steps

**Connect to your repo:** Register a GitHub webhook to catch violations at PR time. See the [full setup guide](README.md#full-setup).

**Questions or feedback:** Open an issue on [GitHub](https://github.com/thaneeshshanand/arch-conscience) or reach out on [LinkedIn](https://www.linkedin.com/in/thaneeshshanand/).