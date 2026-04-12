# Conduit — RealWorld Example App

## Project Overview
Medium.com clone (Conduit) — monorepo with React frontend and Express.js backend.
Two workspaces: `backend/` and `frontend/`.

## Architecture

### Backend Stack
Express.js 5 with Sequelize ORM and PostgreSQL. This stack was chosen for its maturity and the team's existing expertise. We evaluated Prisma as an alternative ORM but rejected it — Sequelize has better support for raw queries we need for complex article feeds, and the team already has Sequelize migration workflows established. We also considered NestJS but rejected it as over-engineered for this application size.

### Authentication
JWT-based authentication implemented in `backend/helper/jwt.js` with bcrypt password hashing in `backend/helper/bcrypt.js`. Tokens are verified via the `backend/middleware/authentication.js` middleware. We evaluated Passport.js but rejected it — the abstraction layer adds complexity without benefit for a single auth strategy. Never store plain-text passwords. Never implement session-based auth — the API is stateless and consumed by multiple clients.

### Data Access
All database access goes through Sequelize models in `backend/models/`. Controllers in `backend/controllers/` call model methods directly — there is no service layer by design. We evaluated adding a service layer but rejected it as unnecessary indirection for this application's complexity level. Do not introduce a service layer without updating this decision.

### Database Migrations
All schema changes must go through Sequelize migrations in `backend/migrations/`. Never modify the database schema by hand or via model sync. Migrations must be backward-compatible — the deploy pipeline runs migrations before the new code is live, so old code must work with the new schema.

### API Design
RESTful API following the RealWorld spec. All endpoints are defined in `backend/routes/`. Route files handle URL mapping only — business logic belongs in controllers. We considered GraphQL but rejected it for this project — the RealWorld spec defines a REST API, and GraphQL would add client complexity without clear benefit for our data access patterns.

### Frontend
React with Vite + SWC. Context API for state management via `frontend/src/context/`. We evaluated Redux but rejected it — Context API is sufficient for this application's state complexity, and Redux adds boilerplate without proportional benefit. Do not add Redux or other state management libraries without revisiting this decision.

## Constraints
- All API responses must follow the RealWorld API spec format. No custom response shapes.
- Authentication middleware must run before any route that accesses user data. The middleware sets `req.user` — routes must not parse tokens themselves.
- PostgreSQL is the only supported database. Sequelize is configured for Postgres via `backend/config/config.js`. Do not add SQLite, MySQL, or in-memory database support.
- The frontend must not call the database directly or import anything from `backend/`. The frontend communicates with the backend exclusively through the REST API.
- Passwords must be hashed with bcrypt before storage. The cost factor is set in `backend/helper/bcrypt.js`. Never reduce the bcrypt cost factor below 10.

## Principles
- Keep controllers thin — they should validate input, call model methods, and format responses. Complex query logic belongs in Sequelize scopes or model class methods.
- One component per file in the frontend. Each component gets its own directory with an index.js barrel export.
- Prefer Sequelize query methods over raw SQL. Raw queries are acceptable only for complex aggregations that can't be expressed efficiently with Sequelize's query builder.
- Error handling goes through the centralized `backend/middleware/errorHandler.js`. Controllers should throw errors, not catch and format them.
- Use environment variables for all configuration. No hardcoded connection strings, secrets, or environment-specific values in source code.