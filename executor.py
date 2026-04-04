"""
Bearing - Task executor

Runs tasks via `claude -p` with the appropriate flags,
captures output, and parses results.

Output format verified against Claude Code v2.1.x (April 2026).
"""

import subprocess
import shutil
import time
import json
from tasks_schema import Task, TaskResult, TaskStatus


def check_claude_installed() -> bool:
    """Verify claude CLI is available."""
    return shutil.which("claude") is not None


def build_command(task: Task, project_dir: str) -> list[str]:
    """
    Build the `claude -p` command with all flags for a given task.

    Combines the task prompt with any cross-task context so Claude
    Code gets everything it needs in a single invocation.
    """
    cfg = task.config

    # Build the full prompt: context + task prompt
    full_prompt_parts = []
    if task.context:
        full_prompt_parts.append(
            f"CONTEXT FROM PREVIOUS TASKS:\n{task.context}\n"
        )
    full_prompt_parts.append(task.prompt)
    full_prompt = "\n".join(full_prompt_parts)

    cmd = [
        "claude",
        "-p", full_prompt,
        "--model", cfg.model,
        "--output-format", "json",
        "--max-budget-usd", str(cfg.budget_usd),
        "--max-turns", str(cfg.max_turns),
    ]

    if cfg.permission_mode == "auto":
        cmd.extend(["--permission-mode", "auto"])

    if cfg.effort:
        cmd.extend(["--effort", cfg.effort])

    if cfg.worktree:
        cmd.extend(["-w", cfg.worktree])

    return cmd


def parse_claude_output(raw_output: str) -> dict:
    """Parse Claude Code's JSON output."""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {
            "result": raw_output,
            "is_error": False,
        }


def extract_cost(data: dict) -> float:
    """
    Extract cost from Claude Code's JSON output.
    Verified field: total_cost_usd (top-level)
    """
    if "total_cost_usd" in data:
        try:
            return float(data["total_cost_usd"])
        except (ValueError, TypeError):
            pass

    # Fallback: sum per-model costs from modelUsage
    if isinstance(data.get("modelUsage"), dict):
        total = 0.0
        for model_data in data["modelUsage"].values():
            if isinstance(model_data, dict) and "costUSD" in model_data:
                try:
                    total += float(model_data["costUSD"])
                except (ValueError, TypeError):
                    pass
        if total > 0:
            return total

    return 0.0


def extract_tokens(data: dict) -> tuple[int, int, int]:
    """
    Extract token counts from Claude Code's JSON output.
    Returns (input_tokens, output_tokens, cache_read_tokens).

    Verified format:
      usage.input_tokens           — direct (uncached) input
      usage.cache_creation_input_tokens — freshly cached
      usage.cache_read_input_tokens    — read from cache (0.1x cost)
      usage.output_tokens          — model output

    We sum all input types for total input_tokens, and track
    cache_read_tokens separately since they're 0.1x cost and
    shouldn't be weighted equally for usage estimates.
    """
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0

    if isinstance(data.get("usage"), dict):
        usage = data["usage"]
        direct = int(usage.get("input_tokens", 0))
        cache_create = int(usage.get("cache_creation_input_tokens", 0))
        cache_read = int(usage.get("cache_read_input_tokens", 0))
        input_tokens = direct + cache_create + cache_read
        output_tokens = int(usage.get("output_tokens", 0))
        cache_read_tokens = cache_read

    # Fallback: modelUsage (camelCase)
    elif isinstance(data.get("modelUsage"), dict):
        for model_data in data["modelUsage"].values():
            if isinstance(model_data, dict):
                direct = int(model_data.get("inputTokens", 0))
                cache_create = int(model_data.get("cacheCreationInputTokens", 0))
                cache_read = int(model_data.get("cacheReadInputTokens", 0))
                input_tokens += direct + cache_create + cache_read
                output_tokens += int(model_data.get("outputTokens", 0))
                cache_read_tokens += cache_read

    return input_tokens, output_tokens, cache_read_tokens


def extract_turns(data: dict) -> int:
    """Extract number of turns used."""
    return int(data.get("num_turns", 0))


def extract_summary(data: dict) -> str:
    """
    Extract a human-readable summary from Claude's output.
    Verified field: result (top-level string)
    """
    if isinstance(data, dict):
        result = data.get("result", "")
        if isinstance(result, str):
            return result[:2000]
        if isinstance(result, list):
            texts = [
                block.get("text", "")
                for block in result
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(texts)[:2000]
    return str(data)[:2000]


def extract_error(data: dict) -> str:
    """Extract error message from Claude's JSON output."""
    errors = data.get("errors", [])
    if errors:
        return "; ".join(str(e) for e in errors)
    subtype = data.get("subtype", "")
    if subtype and subtype != "success":
        return subtype
    return ""


def parse_result(parsed: dict, default_status: TaskStatus) -> TaskResult:
    """Build a TaskResult from parsed Claude Code JSON output."""
    cost = extract_cost(parsed)
    input_tokens, output_tokens, cache_read_tokens = extract_tokens(parsed)
    turns = extract_turns(parsed)
    summary = extract_summary(parsed)
    error = extract_error(parsed)

    # Determine status
    is_error = parsed.get("is_error", False)
    status = TaskStatus.FAILED if is_error else default_status

    return TaskResult(
        status=status,
        cost_usd=cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
        turns_used=turns,
        summary=summary,
        error=error,
    )


def run_task(task: Task, project_dir: str) -> TaskResult:
    """
    Execute a single task via `claude -p`.

    Returns a TaskResult with status, cost, summary, and any errors.
    """
    cmd = build_command(task, project_dir)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=1800,  # 30 min safety net
        )

        # Claude Code puts useful JSON in stdout even on non-zero exit
        if result.stdout:
            parsed = parse_claude_output(result.stdout)
            if isinstance(parsed, dict) and "type" in parsed:
                default_status = (
                    TaskStatus.COMPLETED if result.returncode == 0
                    else TaskStatus.FAILED
                )
                return parse_result(parsed, default_status)

        # No parseable JSON output
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return TaskResult(
                status=TaskStatus.FAILED,
                summary=f"Claude Code exited with code {result.returncode}",
                error=error_msg,
            )

        # Empty successful output (shouldn't happen)
        return TaskResult(
            status=TaskStatus.COMPLETED,
            summary="(no output)",
        )

    except subprocess.TimeoutExpired:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary="Task timed out after 30 minutes",
            error="TIMEOUT",
        )
    except FileNotFoundError:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary="claude CLI not found in PATH",
            error="CLAUDE_NOT_FOUND",
        )
    except Exception as e:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Unexpected error: {type(e).__name__}",
            error=str(e),
        )
