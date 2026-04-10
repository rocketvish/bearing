"""
Bearing - Task executor

Runs tasks via CLI agents (Claude Code, Codex, or custom).
Assembles focused prompts with relevance directives and
structured context compression, then captures and parses results.

Supports:
  - claude: Full flag support (model, effort, budget, turns, auto mode)
  - codex: Non-interactive exec mode with --json output
  - Custom: Any CLI that accepts a prompt argument
"""

import subprocess
import shutil
import json
import re
from tasks_schema import Task, TaskResult, TaskStatus


# --- Context Compression ---


def parse_context_entries(context_text: str) -> list[dict]:
    """
    Parse prose context entries into structured dicts.

    Input format (written by propagate_context):
      [task-001: Add auth hook | files: src/hooks/useAuth.js] Created useAuth...
      [task-002: Settings page] Created SettingsPage...

    Returns list of:
      {"id": "task-001", "name": "Add auth hook", "files": [...], "did": "Created..."}
    """
    if not context_text.strip():
        return []

    entries = []
    for line in context_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue

        # Parse: [task-id: name | files: f1, f2] summary
        # Or:    [task-id: name] summary (no files)
        match = re.match(
            r"\[([^:]+):\s*([^|\]]+?)(?:\s*\|\s*files:\s*([^\]]*))?\]\s*(.*)",
            line,
        )
        if match:
            task_id = match.group(1).strip()
            name = match.group(2).strip()
            files_str = match.group(3)
            summary = match.group(4).strip()
            files = (
                [f.strip() for f in files_str.split(",") if f.strip()]
                if files_str
                else []
            )
            entries.append(
                {
                    "id": task_id,
                    "name": name,
                    "files": files,
                    "did": summary,
                }
            )
        else:
            # Unparseable line — keep as raw text
            entries.append({"id": "?", "name": "", "files": [], "did": line})

    return entries


def compress_context(context_text: str) -> str:
    """
    Convert prose context to structured JSON.

    Prose (stored in tasks.json, human-readable):
      [task-001: Add auth hook | files: src/hooks/useAuth.js] Created useAuth...

    Structured (sent to executor, token-efficient):
      PRIOR:[{"id":"task-001","did":"Created useAuth...","files":["src/hooks/useAuth.js"]}]
    """
    entries = parse_context_entries(context_text)
    if not entries:
        return ""

    compressed = []
    for e in entries:
        item = {"id": e["id"], "did": e["did"]}
        if e["files"]:
            item["files"] = e["files"]
        compressed.append(item)

    return "PRIOR:" + json.dumps(compressed, separators=(",", ":"))


# --- Prompt Assembly ---


def assemble_prompt(
    task: Task,
    context_format: str = "structured",
    threshold_keep: float = 0.6,
    threshold_drop: float = 0.35,
    embedding_model: str = "nomic-embed-text",
    compression_model: str = "gemma4:26b",
) -> tuple[str, dict]:
    """
    Build the full prompt from task fields.

    Returns (prompt_string, metrics_dict).

    Four context formats:
      prose          — v1 behavior, everything as natural language
      structured     — JSON hooks, FOCUS/SKIP as lists
      embedding      — relevance scoring + template compression for mid-tier
      embedding+llm  — relevance scoring + LLM compression for mid-tier

    Prompt structure:
      1. Relevance directives (FOCUS/SKIP)
      2. Context from previous tasks (format-dependent)
      3. The actual task prompt (always prose)
    """
    parts = []
    metrics = {
        "context_original": len(task.context),
        "context_compressed": 0,
        "format": context_format,
        "chunks_kept": 0,
        "chunks_compressed": 0,
        "chunks_dropped": 0,
        "scoring_latency_ms": 0,
        "compression_latency_ms": 0,
        "scores": [],
        "ollama_available": True,
    }

    use_compact_directives = context_format != "prose"

    # Relevance directives
    if task.relevant_files or task.ignore_patterns:
        if use_compact_directives:
            focus_lines = []
            if task.relevant_files:
                focus_lines.append(
                    "FOCUS:" + json.dumps(task.relevant_files, separators=(",", ":"))
                )
            if task.ignore_patterns:
                focus_lines.append(
                    "SKIP:" + json.dumps(task.ignore_patterns, separators=(",", ":"))
                )
            parts.append("\n".join(focus_lines))
        else:
            focus_lines = []
            if task.relevant_files:
                file_list = ", ".join(task.relevant_files)
                focus_lines.append(
                    f"FOCUS: Read these files first, they are most relevant: {file_list}"
                )
            if task.ignore_patterns:
                skip_list = ", ".join(task.ignore_patterns)
                focus_lines.append(f"SKIP: Do not read or modify these: {skip_list}")
            parts.append("\n".join(focus_lines))

    # Context from completed dependencies
    if task.context:
        if context_format == "prose":
            parts.append(f"CONTEXT FROM PREVIOUS TASKS:\n{task.context}")
            metrics["context_compressed"] = len(task.context)

        elif context_format == "structured":
            compressed = compress_context(task.context)
            if compressed:
                parts.append(compressed)
                metrics["context_compressed"] = len(compressed)

        elif context_format in ("embedding", "embedding+llm"):
            from relevance import score_and_compress

            # Split context into per-task chunks
            chunks = [
                line.strip()
                for line in task.context.strip().split("\n")
                if line.strip()
            ]

            scored_chunks, rel_metrics = score_and_compress(
                context_chunks=chunks,
                task_prompt=task.prompt,
                threshold_keep=threshold_keep,
                threshold_drop=threshold_drop,
                embedding_model=embedding_model,
                compression_model=compression_model,
                use_llm_compression=(context_format == "embedding+llm"),
            )

            # Merge relevance metrics
            metrics["chunks_kept"] = rel_metrics["chunks_kept"]
            metrics["chunks_compressed"] = rel_metrics["chunks_compressed"]
            metrics["chunks_dropped"] = rel_metrics["chunks_dropped"]
            metrics["scoring_latency_ms"] = rel_metrics["scoring_latency_ms"]
            metrics["compression_latency_ms"] = rel_metrics["compression_latency_ms"]
            metrics["scores"] = rel_metrics["scores"]
            metrics["ollama_available"] = rel_metrics["ollama_available"]

            if scored_chunks:
                context_text = "PRIOR:\n" + "\n".join(scored_chunks)
                parts.append(context_text)
                metrics["context_compressed"] = len(context_text)

    # The actual task prompt (always prose)
    parts.append(task.prompt)

    return "\n\n".join(parts), metrics


# --- CLI Command Builders ---


def build_claude_command(task: Task, prompt: str) -> list[str]:
    """Build a `claude -p` command with full flag support."""
    cfg = task.config
    cmd = [
        "claude",
        "-p",
        prompt,
        "--model",
        cfg.model,
        "--output-format",
        "json",
        "--max-budget-usd",
        str(cfg.budget_usd),
        "--max-turns",
        str(cfg.max_turns),
    ]

    if cfg.permission_mode == "auto":
        cmd.extend(["--permission-mode", "auto"])
    elif cfg.permission_mode == "dangerously_skip":
        cmd.append("--dangerously-skip-permissions")

    if cfg.effort:
        cmd.extend(["--effort", cfg.effort])

    if cfg.worktree:
        cmd.extend(["-w", cfg.worktree])

    return cmd


def build_codex_command(task: Task, prompt: str) -> list[str]:
    """Build a `codex exec` command for non-interactive mode."""
    cfg = task.config
    cmd = [
        "codex",
        "exec",
        "--json",
        prompt,
    ]

    if cfg.model:
        cmd.extend(["--model", cfg.model])

    return cmd


def build_custom_command(task: Task, prompt: str) -> list[str]:
    """Build a command for any custom CLI agent."""
    return [task.config.cli, prompt]


def build_command(task: Task, prompt: str) -> list[str]:
    """Route to the correct CLI command builder with pre-assembled prompt."""
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
    Parse Codex CLI JSONL output.
    codex exec --json outputs newline-delimited JSON events.
    We take the last text event as the result.
    """
    try:
        # Try single JSON object first
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    # Try JSONL — take last meaningful event
    lines = raw_output.strip().split("\n")
    last_text = ""
    for line in lines:
        try:
            event = json.loads(line)
            if isinstance(event, dict):
                # Codex events have a "message" or "content" field
                if "message" in event:
                    last_text = event["message"]
                elif "content" in event:
                    last_text = event["content"]
        except json.JSONDecodeError:
            continue

    if last_text:
        return {"result": last_text, "is_error": False}
    return {"result": raw_output, "is_error": False}


def parse_output(raw_output: str, cli: str) -> dict:
    """Route to the correct output parser."""
    if cli == "claude":
        return parse_claude_output(raw_output)
    elif cli == "codex":
        return parse_codex_output(raw_output)
    else:
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


def run_task(
    task: Task, project_dir: str, prompt: str = None, context_format: str = "structured"
) -> tuple[TaskResult, dict]:
    """
    Execute a single task via the configured CLI.

    If prompt is provided, uses it directly. Otherwise assembles from task fields.
    Returns (TaskResult, metrics_dict).
    """
    if prompt is None:
        prompt, metrics = assemble_prompt(task, context_format)
    else:
        metrics = {
            "context_original": 0,
            "context_compressed": 0,
            "format": context_format,
        }

    cmd = build_command(task, prompt)
    cli = task.config.cli.lower()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=1800,  # 30 min safety net
        )

        if result.stdout:
            parsed = parse_output(result.stdout, cli)
            if isinstance(parsed, dict) and ("type" in parsed or "result" in parsed):
                default_status = (
                    TaskStatus.COMPLETED
                    if result.returncode == 0
                    else TaskStatus.FAILED
                )
                return parse_result(parsed, default_status), metrics

        if result.returncode != 0:
            error_msg = result.stderr.strip() if result.stderr else "Unknown error"
            return TaskResult(
                status=TaskStatus.FAILED,
                summary=f"{cli} exited with code {result.returncode}",
                error=error_msg,
            ), metrics

        return TaskResult(
            status=TaskStatus.COMPLETED,
            summary="(no output)",
        ), metrics

    except subprocess.TimeoutExpired:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary="Task timed out after 30 minutes",
            error="TIMEOUT",
        ), metrics
    except FileNotFoundError:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"{cli} CLI not found in PATH",
            error=f"{cli.upper()}_NOT_FOUND",
        ), metrics
    except Exception as e:
        return TaskResult(
            status=TaskStatus.FAILED,
            summary=f"Unexpected error: {type(e).__name__}",
            error=str(e),
        ), metrics
