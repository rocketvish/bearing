# Bearing Benchmark

Synthetic project for evaluating Bearing's context compression strategies.

## Setup

```bash
cd benchmark
npm install
git init
git add .
git commit -m "benchmark baseline"
```

## Run evaluation

```bash
# From the benchmark directory:
bearing eval .
```

This runs the same 8 tasks under 4 context format conditions (prose, structured, embedding, embedding+llm), measures metrics, judges quality with Claude Opus, and outputs a comparison report to `eval/results.md`.

## Prerequisites

- Claude Code CLI in PATH
- Ollama running with `nomic-embed-text` and `gemma4:26b` pulled
- Git repo initialized (eval runner resets codebase between conditions)

## Task structure

```
task-001: Database schema          (independent)
task-002: User auth                (depends: 001)
task-003: Task CRUD                (depends: 001)
task-004: Input validation         (depends: 002, 003)
task-005: Auth middleware           (depends: 002, 003)
task-006: Auth tests               (depends: 002, 004)
task-007: Task CRUD tests          (depends: 003, 004, 005)
task-008: API documentation        (depends: 005, 006, 007)
```

Tasks 4-8 accumulate predecessor context where some is relevant and some isn't. This is where embedding-based scoring should demonstrate value by selectively dropping irrelevant context.
