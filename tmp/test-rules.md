# Project Overview
E-commerce platform built with Next.js 14, TypeScript, PostgreSQL.

# Code Style
- TypeScript strict mode, no `any`
- Functional components with arrow functions
- Named exports, no default exports
- 2-space indentation

# Architecture
- Server Components by default, Client Components only when interactivity is needed
- Use JWT for authentication, never session cookies
- PostgreSQL for all transactional data, Redis for caching only
- Event-driven communication between services via RabbitMQ, no direct HTTP calls between services
- All external API calls go through the API gateway, services never call third-party APIs directly

# Testing
- Vitest for unit tests, Playwright for E2E
- 80% coverage on business logic

# Commands
- pnpm dev - start dev server
- pnpm test - run tests

# Known Constraints
- The payments module uses a legacy SOAP client for the bank integration. Do not refactor this until the bank migrates to REST (Q3 2026).
- Auth middleware must run before rate limiting. Reversing this order breaks token validation for rate-limited endpoints.
- The /api/webhook endpoint intentionally skips CSRF validation because GitHub webhooks use HMAC signatures instead.