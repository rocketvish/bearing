# Bearing

Planning and execution shouldn't share a context window.

Bearing separates the two. You have a strategic conversation with an AI planner about *what* to build. Bearing runs the tasks in fresh, isolated sessions. Results come back as files. No copy-pasting between windows.

Works with Claude Code, Codex, or any CLI agent.

## Install

```bash
git clone https://github.com/rocketvish/bearing.git
cd bearing
uv tool install -e .
```

Requires Python 3.11+ and at least one CLI agent (Claude Code, Codex, etc.) in PATH. No other dependencies.

## Usage

```bash
cd your-project
bearing start .
```

This opens an interactive Claude Code session (Opus) that knows how to write Bearing task files. Talk through what you want to build. When you converge on a plan, the planner writes `tasks.json`. Then, from within the same session:

```
!bearing run .
```

Tasks execute one by one in separate processes. Each gets a clean context window. Results write to `status.md`. Check progress anytime:

```
!bearing summary .
```

One window, files as the interface.

## How It Works

```
You (planner session)     Bearing         CLI Agent (executor)
---------------------     -------         --------------------
Discuss approach     ->   writes tasks.json
                          reads task       ->   claude -p / codex "prompt..."
Read status.md       <-   writes status.md <-   returns result
Adjust plan          ->   picks up next    ->   next task...
```

Four files, all in your project directory:

- **tasks.json** -- the task queue. Bearing reads this, you (or the planner) write it.
- **status.md** -- Bearing writes this. Results, costs, errors.
- **plan.md** -- your scratch space. Bearing ignores it.
- **CLAUDE.md** -- unchanged. Still your project context.

## Multi-CLI Support

Each task specifies which CLI agent runs it:

```json
{
  "config": {
    "cli": "claude",
    "model": "opus"
  }
}
```

Use Claude with Opus for architectural decisions, Sonnet for routine implementation, Codex for a different perspective, or mix them in the same task queue. The planner decides which tool fits each task.

Supported CLIs: `claude` (full flag support), `codex` (basic), or any custom command.

## Context Focusing

The highest-leverage feature. Instead of the executor reading your entire codebase and hoping attention lands in the right place, tell it exactly what matters:

```json
{
  "relevant_files": ["src/hooks/useAuth.js", "src/components/Login.jsx"],
  "ignore_patterns": ["node_modules", "dist", "*.test.js"]
}
```

The executor sees focused directives before the task prompt:
```
FOCUS: Read these files first, they are most relevant: src/hooks/useAuth.js, src/components/Login.jsx
SKIP: Do not read or modify these: node_modules, dist, *.test.js
```

This reduces token consumption and concentrates the model's attention on what actually matters. When a task completes, its relevant_files automatically propagate to dependent tasks.

## Auto-Context

When task-001 completes, Bearing injects a summary into every dependent task:

```
[task-001: Add auth hook] Created useAuth hook at src/hooks/useAuth.js...
```

Both the text summary and the file relevance list propagate. No manual copy-paste. No duplication on re-runs.

## Task Format

```json
{
  "id": "task-001",
  "name": "Add user settings page",
  "prompt": "Read the current codebase first -- especially the files in FOCUS...",
  "config": {
    "cli": "claude",
    "model": "sonnet",
    "effort": "high",
    "budget_usd": 3.00,
    "max_turns": 20,
    "permission_mode": "auto",
    "worktree": null,
    "fast_mode": false
  },
  "depends_on": [],
  "checkpoint": "pause",
  "on_failure": "retry_once",
  "context": "",
  "relevant_files": ["src/components/", "src/routes/"],
  "ignore_patterns": ["node_modules", "dist"]
}
```

## Commands

```
bearing start .       Open a planner session (Opus)
bearing init .        Create starter files
bearing run .         Execute queued tasks
bearing summary .     One-line progress check
bearing status .      Full status
bearing watch .       Live-tail task changes
bearing validate .    Check tasks.json syntax
```

## What This Is Not

Not Conductor (the Mac app for parallel agents). Not Gas Town (20-agent swarms). Not gstack (role-switching within one session).

Those tools answer "how do I run more agents?" Bearing answers a different question: "how do I think clearly about what to build while AI builds it?"

This is just four Python files, no UI or dependencies. Uses standard MIT license.
