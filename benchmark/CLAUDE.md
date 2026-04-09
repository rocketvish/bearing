# Task Management API

A REST API for task management with weekly report generation, built with Express and SQLite.

## Architecture

- `src/index.js` — Express server entry point
- `src/db.js` — SQLite database setup and schema
- `src/routes/` — Route handlers (auth, tasks, reports)
- `src/middleware/` — Auth and validation middleware
- `src/utils/` — Utility modules (markdown converter)
- `src/reports.js` — Report generation logic

## Conventions

- Use sql.js (SQLite compiled to JavaScript, async initialization)
- JWT for authentication (jsonwebtoken package)
- bcryptjs for password hashing
- Node's built-in test runner (node:test)
- Return JSON for all API responses
- Use HTTP status codes correctly (201 for creation, 401 for unauthorized, 204 for delete, etc.)
- No external markdown libraries — use regex-based converter
