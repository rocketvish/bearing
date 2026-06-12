# Bearing

Bearing is a small task runner for directing AI coding agents with a human in the planning loop. It keeps the planning conversation separate from execution: you debate the work, translate intent into precise tasks, and then Bearing runs each task in its own fresh agent process.

It is meant for codebase work where a single long agent session starts to get noisy. The task queue is both a project-management tool and a context-management tool: it records what should happen, what depends on what, which files matter, and what context should carry forward.

Bearing works with Claude Code, Codex, and custom CLI commands. It also includes an experimental API-backed agent that can use OpenAI Responses or Anthropic Messages directly.

For the longer motivation behind the workflow, see [Why I stopped copy-pasting between AI coding sessions](https://rocketvish.substack.com/p/why-i-stopped-copy-pasting-between).

## Install

```bash
git clone https://github.com/rocketvish/bearing.git
cd bearing
uv tool install -e .
```

You need Python 3.11+ and at least one CLI agent in `PATH`, unless you are only using the standalone API agent.

## Basic Flow

From the project you want to work on:

```bash
bearing start .
```

Use the planner to talk through the work, challenge the approach, and decide where human judgment matters. The planner writes those decisions into `tasks.json`. When the plan is ready, run:

```bash
bearing run .
```

Bearing reads the queue, runs ready tasks one at a time, and writes progress to `status.md`. You can check the current state without starting a run:

```bash
bearing summary .
bearing status .
```

## Files

Bearing uses ordinary project files as its interface:

- `tasks.json` is the task queue.
- `status.md` is generated from task results.
- `plan.md` is optional scratch space for planning.
- `CLAUDE.md` or `AGENTS.md` remain normal agent context files; Bearing does not own them.

## Task Queue

A task describes what to run, which files matter, and how it relates to other tasks:

```json
{
  "id": "task-001",
  "name": "Add user settings page",
  "prompt": "Add a settings page and tests. Read the focused files first.",
  "config": {
    "cli": "claude",
    "model": "sonnet",
    "effort": "high",
    "budget_usd": 3.0,
    "max_turns": 20,
    "permission_mode": "auto"
  },
  "depends_on": [],
  "checkpoint": "pause",
  "on_failure": "retry_once",
  "context": "",
  "relevant_files": ["src/components/", "src/routes/"],
  "ignore_patterns": ["node_modules", "dist"]
}
```

The planner can write this by hand, or you can create it yourself. `bearing validate .` checks that the file is shaped correctly.

## Context Focusing

Tasks can name files that are likely to matter:

```json
{
  "relevant_files": ["src/hooks/useAuth.js", "src/components/Login.jsx"],
  "ignore_patterns": ["node_modules", "dist", "*.test.js"]
}
```

Bearing turns those fields into instructions at the top of the executor prompt. This does not stop the agent from reading other files, but it gives the run a better starting point and makes the intended scope clear.

When a task completes, its summary and relevant files are propagated to dependent tasks. Later tasks get the parts of the prior work that should matter, without inheriting the entire conversation, test output, and error history.

## Commands

```bash
bearing start .        Open a planner session
bearing init .         Create starter files
bearing run .          Execute queued tasks
bearing summary .      Show a short progress summary
bearing status .       Show full task status
bearing watch .        Watch task status updates
bearing validate .     Validate tasks.json
bearing eval .         Run context-format evals
bearing eval-compare . Compare isolated tasks with a single accumulated session
bearing eval-agent .   Run standalone-agent compression/retrieval evals
```

## Agent Choices

Normal tasks run through CLI agents. A task can choose a built-in CLI or a custom command:

```json
{
  "config": {
    "cli": "codex",
    "model": "gpt-5.5"
  }
}
```

The standalone API agent in `agent.py` is separate from the CLI runner. It exists to test context-management ideas without depending on a particular CLI product. It exposes the same three local tools to models:

- `read_file(path)`
- `write_file(path, content)`
- `run_command(command)`

OpenAI Responses is the default backend. Anthropic Messages is still supported through the same normalized loop:

```python
from agent import run_agent

result = run_agent(
    task_prompt="Build a REST API with tests",
    project_dir="./my-project",
    provider="openai",  # or "anthropic"
    compression_mode="retrieval",
    use_caching=True,
    reasoning_effort="high",
)
```

For `bearing eval-agent`, set `BEARING_AGENT_PROVIDER` and `BEARING_AGENT_MODEL` to compare backends.

Review `tasks.json` before running it. Bearing executes the configured CLI command in your project directory, and the standalone API agent has a `run_command` tool for tests and inspection. The API agent confines file reads and writes to the project root and blocks a small set of destructive command patterns, but it is still a code-execution tool.

## Design Choices

Bearing tries to keep the project model simple:

- Planning produces files instead of hidden session state.
- Execution happens in isolated runs so one task's context does not silently become another task's baggage.
- Context is passed forward deliberately through task summaries, dependencies, and focused file lists. This is a rough form of human-guided compression: the planner helps decide what context is still important instead of leaving that entirely to an automatic summary.
- Human review remains part of the loop. Tasks can pause at checkpoints, and the planner can adjust the queue based on `status.md`.
- Provider-specific API shapes stay behind adapters where possible. OpenAI Responses has richer agent primitives than the Anthropic Messages API, but Bearing's internal loop should not require one provider's object model.

The tradeoff is that Bearing is more explicit than a single long agent session. You have to maintain a queue, name dependencies, and decide where context should flow. That cost is the point when the work is large enough that your original intent starts competing with everything else the agent has seen.

## Evaluation

The eval scripts are for comparing context-management strategies, not for proving a universal ranking between models. Current evals include:

- `bearing eval`: context format comparison.
- `bearing eval-compare`: isolated task sessions versus one accumulated mega-prompt.
- `bearing eval-agent`: standalone API agent with raw history, compression, caching, retrieval, and combinations of those modes.

Results are noisy because agent runs are noisy. The reports are most useful for spotting broad patterns, checking whether compression or retrieval changes the token curve, and finding cases where a strategy loses task detail.
