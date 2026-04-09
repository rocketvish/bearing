# Task Management API

A simple REST API for task management, built with Express and SQLite.

## Architecture

- `src/index.js` — Express server entry point
- `src/db.js` — SQLite database setup and schema
- `src/routes/` — Route handlers
- `src/middleware/` — Auth and validation middleware
- `src/tests/` — Test files

## Conventions

- Use better-sqlite3 (synchronous API)
- JWT for authentication (jsonwebtoken package)
- bcryptjs for password hashing
- Node's built-in test runner (node --test)
- Return JSON for all API responses
- Use HTTP status codes correctly (201 for creation, 401 for unauthorized, etc.)
