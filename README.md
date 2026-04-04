# Bearing

Planning and execution shouldn't share a context window.

Bearing separates the two. You have a strategic conversation with Claude Code about *what* to build. Bearing runs the tasks in fresh, isolated sessions. Results come back as files. No copy-pasting between windows.

## Install

```bash
git clone https://github.com/rocketvish/bearing.git
cd bearing
uv tool install -e .
```

Requires Python 3.11+, Claude Code CLI in PATH, and auto mode enabled (`claude --enable-auto-mode`). No other dependencies.

## Usage

```bash
cd your-project
bearing start .
```

This opens an interactive Claude Code session (Opus) that knows how to write Bearing task files. Talk through what you want to build. When you converge on a plan, the planner writes `tasks.json`. Then, from within the same session:

```
!bearing run .
```

Tasks execute one by one in separate `claude -p` processes. Each gets a clean context window. Results write to `status.md`. Check progress anytime:

```
!bearing summary .
```

That's the whole workflow. One window, files as the interface.

## How It Works

```
You (planner session)     Bearing         Claude Code (executor)
---------------------     -------         ----------------------
Discuss approach     ->   writes tasks.json
                          reads task       ->   claude -p "prompt..."
Read status.md       <-   writes status.md <-   returns result
Adjust plan          ->   picks up next    ->   claude -p "prompt..."
```

Four files, all in your project directory:

- **tasks.json** -- the task queue. Bearing reads this, you (or the planner) write it.
- **status.md** -- Bearing writes this. Results, costs, errors.
- **plan.md** -- your scratch space. Bearing ignores it.
- **CLAUDE.md** -- unchanged. Still your project context.

## Task Format

```json
{
  "id": "task-001",
  "name": "Add user settings page",
  "prompt": "Read the current codebase first -- especially src/components/...",
  "config": {
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
  "context": ""
}
```

`budget_usd` acts as a token ceiling even on subscription plans. `checkpoint: "pause"` stops execution for your review. `checkpoint: "auto"` continues to the next task. `on_failure: "retry_once"` appends the error to the prompt and tries again.

## Auto-Context

When task-001 completes, Bearing injects a summary into every task that depends on it:

```
[task-001: Add settings page] Created SettingsPage component with theme toggle...
```

No manual copy-paste. No duplication on re-runs.

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

Four Python files. No UI. No dependencies. MIT licensed.
