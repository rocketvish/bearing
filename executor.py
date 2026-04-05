"""
Bearing - Task executor

Runs tasks via CLI agents (Claude Code, Codex, or custom).
Assembles focused prompts with relevance directives, then
captures and parses results.

Supports:
  - claude: Full flag support (model, effort, budget, turns, auto mode)
  - codex: Basic prompt passthrough
  - Custom: Any CLI that accepts a prompt argument
"""

import subprocess
import shutil
import time
import json
from tasks_schema import Task, TaskResult, TaskStatus


# --- Prompt Assembly (CLI-agnostic) ---

def assemble_prompt(task: Task) -> str:
    """
    Build the full prompt from task fields.

    This is where context focusing happens. Instead of just passing
    the raw prompt, we prepend:
      1. Relevance directives (which files to read, which to skip)
      2. Context from previous tasks
      3. The actual task prompt

    The executor sees a focused, curated view of the project
    rather than having to discover relevance from the full codebase.
    """
    parts = []

    # Relevance directives — tell the agent what matters
    if task.relevant_files or task.ignore_patterns:
        focus_lines = []
        if task.relevant_files:
            file_list = ", ".join(task.relevant_files)
            focus_lines.append(
                f"FOCUS: Read these files first, they are most relevant: {file_list}"
            )
        if task.ignore_patterns:
            skip_list = ", ".join(task.ignore_patterns)
            focus_lines.append(
                f"SKIP: Do not read or modify these: {skip_list}"
            )
        parts.append("\n".join(focus_lines))

    # Context from completed dependencies
    if task.context:
        parts.append(f"CONTEXT FROM PREVIOUS TASKS:\n{task.context}")

    # The actual task prompt
    parts.append(task.prompt)

    return "\n\n".join(parts)


# --- CLI Command Builders ---

def build_claude_command(task: Task, prompt: str) -> list[str]:
    """Build a `claude -p` command with full flag support."""
    cfg = task.config
    cmd = [
        "claude",
        "-p", prompt,
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


def build_codex_command(task: Task, prompt: str) -> list[str]:
    """Build a `codex` command. Flags will evolve as Codex CLI matures."""
    cfg = task.config
    cmd = [
        "codex",
        prompt,
    ]

    # Codex model selection (if supported)
    if cfg.model:
        cmd.extend(["--model", cfg.model])

    return cmd


def build_custom_command(task: Task, prompt: str) -> list[str]:
    """Build a command for any custom CLI agent."""
    return [task.config.cli, prompt]


def build_command(task: Task) -> list[str]:
    """Route to the correct CLI command builder."""
    prompt = assemble_prompt(task)
    cli = task.config.cli.lower()

    if cli == "claude":
        return build_claude_command(task, prompt)
    elif cli == "codex":
        return build_codex_command(task, prompt)
    else:
        return build_custom_command(task, prompt)


def check_cli_installed(cli: str = "claude") -> bool:
    """Verify the CLI tool is available."""
    return shutil.which(cli) is not None


# --- Output Parsing ---

def parse_claude_output(raw_output: str) -> dict:
    """Parse Claude Code's JSON output."""
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"result": raw_output, "is_error": False}


def parse_codex_output(raw_output: str) -> dict:
    """
    Parse Codex CLI output.
    Format may differ from Claude — adapt as Codex CLI evolves.
    Falls back to treating raw output as result text.
    """
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        return {"result": raw_output, "is_error": False}


def parse_output(raw_output: str, cli: str) -> dict:
    """Route to the correct output parser."""
    if cli == "claude":
        return parse_claude_output(raw_output)
    elif cli == "codex":
        return parse_codex_output(raw_output)
    else:
        # Generic: try JSON, fall back to raw text
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            return {"result": raw_output, "is_error": False}


# --- Field Extractors (Claude format, verified) ---

def extract_cost(data: dict) -> float:
    """Extract cost. Verified field: total_cost_usd (top-level)."""
    if "total_cost_usd" in data:
        try:
            return float(data["total_cost_usd"])
        except (ValueError, TypeError):
            pass

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
    Extract token counts: (input, output, cache_read).
    All three input types (direct, cache create, cache read) sum to total input.
    Cache reads tracked separately since they're 0.1x cost.
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
    return int(data.get("num_turns", 0))


def extract_summary(data: dict) -> str:
    """Extract result text. Verified field: result (top-level string)."""
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
    errors = data.get("errors", [])
    if errors:
        return "; ".join(str(e) for e in errors)
    subtype = data.get("subtype", "")
    if subtype and subtype != "success":
        return subtype
    return ""


def parse_result(parsed: dict, default_status: TaskStatus) -> TaskResult:
    """Build a TaskResult from parsed CLI output."""
    cost = extract_cost(parsed)
    input_tokens, output_tokens, cache_read_tokens = extract_tokens(parsed)
    turns = extract_turns(parsed)
    summary = extract_summary(parsed)
    error = extract_error(parsed)

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


# --- Task Runner ---

def run_task(task: Task, project_dir: str) -> TaskResult:
    """
    Execute a single task via the configured CLI.
    Returns a TaskResult with status, cost, summary, and any errors.
    """
    cmd = build_command(task)
    cli = task.config.cli.lower()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=1800,  # 30 min safety net
        )

        # Parse output (even on non-zero exit — CLI puts useful data in stdout)
        if result.stdout:
            parsed = parse_output(result.stdout, cli)
            if isinstance(parsed, dict) and ("type" in parsed or "result" in parsed):
                default_status = (
                    TaskStatus.COMPLETED if result.returncode == 0
                    else TaskStatus.FAILED
                )
                return parse_result(parsed, default_status)

        # No parseable output
        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return TaskResult(
                status=TaskStatus.FAILED,
                summary=f"{cli} exited with code {result.returncode}",
                error=error_msg,
            )

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
            summary=f"{cli} CLI not found in PATH",
            error=f"{cli.upper()}_NOT_FOUND",
        )
    except Exception as e:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Unexpected error: {type(e).__name__}",
            error=str(e),
        )
