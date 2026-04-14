# Bearing - Task Orchestrator for Claude Code

## Architecture

- `bearing.py` -- CLI entry point + orchestration loop (`run_orchestrator`, `propagate_context`)
- `executor.py` -- Prompt assembly (`assemble_prompt`) + CLI execution (`run_task`)
- `relevance.py` -- Embedding-based relevance scoring + compression via Ollama
- `eval_runner.py` -- Evaluation framework: runs benchmark under 4 context format conditions
- `eval_compare.py` -- Eval compare: Bearing (8 sessions) vs single session (1 mega-prompt)
- `agent.py` -- Tool-use agent via Anthropic API (urllib, no SDK), with prompt caching + extended thinking
- `compressor.py` -- Mid-conversation history compression (API or Ollama backends)
- `eval_agent.py` -- Eval: agent with/without compression/caching vs claude -p (5 conditions)
- `test_sanity.py` -- Quick sanity test for caching + thinking features
- `tasks_schema.py` -- Dataclasses for Task, TaskQueue, TaskResult
- `status_writer.py` -- Generates status.md from task queue state

## Eval Framework

Four context format conditions: `prose`, `structured`, `embedding`, `embedding+llm`.

### Eval Compare (`bearing eval-compare`)

Compares Bearing's 8-session approach against a single mega-prompt session. This tests
the core value proposition: task isolation prevents context accumulation and compaction.
The single session gets the same task prompts and FOCUS directives but must build
everything in one accumulated context. Key metrics: total tokens, compaction count,
per-task quality.

### Key Learnings

**Embedding similarity baseline is high for same-domain text.** nomic-embed-text cosine
similarity for chunks within the same project bottoms out around 0.5-0.6, not 0.2-0.3.
Default thresholds are set at keep=0.75, drop=0.55 to account for this. Benchmarks that
need chunks to actually drop require genuinely unrelated domains (e.g., different languages
or problem spaces), not just "parallel feature tracks" in the same codebase.

**Injected context is <0.1% of total tokens.** Claude Code `claude -p` sessions consume
100K-300K tokens reading files, writing code, and running tests. Our injected context is
~50-300 tokens. Cost differences between conditions are dominated by execution
non-determinism, not context size. The value of context compression is in compaction
avoidance across long shared sessions.

**Single-run eval results are noisy.** Per-task turns and cost vary 2-4x across conditions
due to agentic execution randomness. Multi-trial runs are recommended.

**Judge must see task-relevant files.** The quality judge receives source files filtered and
prioritized by task relevance. Files mentioned in the task prompt appear first to survive
truncation.

### Eval Agent (`bearing eval-agent`)

Tests mid-conversation compression hypothesis: full history re-sent every turn is the
real waste, not context accumulation itself. Runs all 8 tasks as a single mega-prompt
agent session (same methodology as eval-compare) to ensure enough context accumulates
(30K+ tokens) to trigger compression. Uses the Anthropic API directly via urllib with
three tools (read_file, write_file, run_command).

Five conditions: `agent-raw` (no compression, no caching), `agent-compressed` (API
compression at 12K threshold), `agent-cached` (prompt caching, no compression),
`agent-compressed-cached` (both compression + caching), `claude-p` (loads cached
results from eval-compare if available, otherwise runs fresh). Key output: per-turn
input token table showing accumulation curves, cache hit rates, and compression
sawtooth. Per-task quality judging using the same judge infrastructure as the other
evals. Report includes per-condition cost breakdowns with cached pricing.

### Agent Features

**Prompt caching** (`use_caching=True`): Formats system prompt as a content block
array with `cache_control: {type: "ephemeral"}` and adds the `anthropic-beta:
prompt-caching-2024-07-31` header. Cache reads cost 0.1x input price; cache creation
costs 1.25x. After compression events, the message prefix cache is invalidated (system
prompt cache survives). The `calculate_cost()` function handles all pricing tiers.

**Extended thinking** (`use_thinking=True`): Adds `thinking: {type: "enabled",
budget_tokens: 10000}` to API requests. Bumps `max_tokens` to 16000 to accommodate
thinking budget + text output. Thinking and redacted_thinking blocks are preserved in
conversation history (API requires them for continuity) but not acted upon. Thinking
blocks are silently skipped during compression serialization.

**Token tracking**: Per-turn metrics include `cache_read_tokens`,
`cache_creation_tokens`, and `thinking_tokens`. The API's `input_tokens` field is the
uncached portion; `total_input = input_tokens + cache_read + cache_creation`. Cost
calculation uses the uncached portion at full price, cache reads at 0.1x, cache
creation at 1.25x, and thinking at output price.

## Commands

```
bearing start <dir>         # Launch interactive planner session
bearing run <dir>           # Execute task queue
bearing eval <dir>          # Run 4-condition evaluation
bearing eval-compare <dir>  # Bearing vs single session comparison
bearing eval-agent <dir>    # Agent compression/caching eval (5 conditions)
bearing status <dir>        # Show status
bearing summary <dir>       # Quick check for planner
bearing watch <dir>         # Live updates
```

## Development

- Python 3.12+, no external dependencies (stdlib only + Anthropic HTTP API + Ollama HTTP API)
- Lint: `ruff check .`
- Format: `ruff format .`
- Tests: `python -m pytest` (if test files exist)
