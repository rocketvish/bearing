"""
Bearing - Minimal code agent with tool use

A tool-use agent loop using the Anthropic API directly via urllib.
Supports mid-conversation context compression to reduce token
accumulation across turns.

Tools:
    read_file(path)      — Read a file relative to the working directory
    write_file(path, content) — Write a file, creating parent dirs
    run_command(command)  — Run a shell command with 30s timeout
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

SYSTEM_PROMPT = (
    "You are a coding agent. You have access to three tools: read_file, "
    "write_file, and run_command. Complete the given task by reading existing "
    "files, writing new files, and running commands as needed. Work efficiently "
    "— read only the files you need, and don't re-read files you've already "
    "seen unless they've changed. When you're done, respond with a brief "
    "summary of what you built. "
    "NEVER run commands that start servers or long-running processes "
    "(npm start, node server.js, node index.js, node src/index.js). "
    "These will timeout and waste turns. Only run commands that exit on "
    "their own like npm test, node -e, or ls."
)

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path "
            "relative to the working directory"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path to read",
                }
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file at the given path relative to the "
            "working directory. Creates parent directories if needed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path to write",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return stdout+stderr. "
            "Has a 30-second timeout. Output truncated to 5000 chars."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                }
            },
            "required": ["command"],
        },
    },
]

# Sonnet pricing (per MTok)
COST_INPUT_PER_MTOK = 3.0
COST_OUTPUT_PER_MTOK = 15.0
COST_CACHE_READ_PER_MTOK = 0.30
COST_CACHE_CREATION_PER_MTOK = 3.75


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    thinking: int = 0,
) -> float:
    """
    Calculate cost with cached and thinking token pricing.
    input_tokens is the uncached portion from the API.
    """
    return (
        input_tokens * COST_INPUT_PER_MTOK / 1_000_000
        + cache_read * COST_CACHE_READ_PER_MTOK / 1_000_000
        + cache_creation * COST_CACHE_CREATION_PER_MTOK / 1_000_000
        + (output_tokens + thinking) * COST_OUTPUT_PER_MTOK / 1_000_000
    )


def load_api_key(project_dir: str = ".") -> str | None:
    """
    Load Anthropic API key from environment or .env file.
    Returns None if not found.
    """
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key

    env_path = os.path.join(project_dir, ".env")
    if os.path.exists(env_path):
        # utf-8-sig strips BOM from PowerShell's "UTF8" encoding; fall back to UTF-16
        for encoding in ("utf-8-sig", "utf-16"):
            try:
                with open(env_path, "r", encoding=encoding) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("ANTHROPIC_API_KEY="):
                            val = line.split("=", 1)[1].strip()
                            # Strip quotes if present
                            if (
                                len(val) >= 2
                                and val[0] in ('"', "'")
                                and val[-1] == val[0]
                            ):
                                val = val[1:-1]
                            if val:
                                return val
                break  # File read successfully, no need to try next encoding
            except UnicodeDecodeError:
                continue
            except OSError:
                break
        print(f"Warning: .env file found at {env_path} but no ANTHROPIC_API_KEY in it")
    else:
        print(f"Note: No .env file at {env_path}")

    return None


def _call_api(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str = SYSTEM_PROMPT,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    use_caching: bool = False,
    use_thinking: bool = False,
) -> dict:
    """
    Send a request to the Anthropic Messages API.
    Returns the parsed response dict.
    Raises RuntimeError on API errors.
    """
    # Format system prompt: array with cache_control when caching, plain string otherwise
    if use_caching:
        system_value = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_value = system

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system_value,
        "messages": messages,
    }
    if tools:
        if use_caching:
            # Cache breakpoint on last tool: caches system + tools prefix
            cached_tools = list(tools)
            cached_tools[-1] = {
                **cached_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }
            body["tools"] = cached_tools
        else:
            body["tools"] = tools
    if use_thinking:
        body["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        # max_tokens must accommodate thinking budget + text/tool output
        if body["max_tokens"] < 16000:
            body["max_tokens"] = 16000

    # Cache breakpoint on last message: caches entire conversation prefix.
    # This is the key optimization — on turn N, system + tools + all messages
    # from turns 1..N-1 hit the cache. Only the new tool results are uncached.
    if use_caching and messages:
        cached_messages = list(messages)
        last_msg = {**cached_messages[-1]}
        content = last_msg.get("content", "")
        if isinstance(content, str):
            last_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        elif isinstance(content, list) and content:
            cached_content = list(content)
            cached_content[-1] = {
                **cached_content[-1],
                "cache_control": {"type": "ephemeral"},
            }
            last_msg["content"] = cached_content
        cached_messages[-1] = last_msg
        body["messages"] = cached_messages

    payload = json.dumps(body).encode("utf-8")

    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_API_VERSION,
    }
    if use_caching:
        headers["anthropic-beta"] = "prompt-caching-2024-07-31"

    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers=headers,
        method="POST",
    )

    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            # Request object is consumed after use, rebuild on retry
            if attempt > 0:
                req = urllib.request.Request(
                    ANTHROPIC_API_URL,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 529) and attempt < max_retries:
                retry_after = e.headers.get("retry-after")
                wait = int(retry_after) if retry_after else 60
                reason = "Rate limited" if e.code == 429 else "Server overloaded"
                print(
                    f"  {reason} ({e.code}), waiting {wait}s... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Anthropic API error {e.code}: {error_body[:500]}"
            ) from None


# --- Tool Execution ---


def _execute_tool(name: str, input_data: dict, project_dir: str) -> str:
    """Execute a tool call and return the result string."""
    if name == "read_file":
        return _tool_read_file(input_data.get("path", ""), project_dir)
    elif name == "write_file":
        return _tool_write_file(
            input_data.get("path", ""),
            input_data.get("content", ""),
            project_dir,
        )
    elif name == "run_command":
        return _tool_run_command(input_data.get("command", ""), project_dir)
    else:
        return f"Error: unknown tool '{name}'"


def _tool_read_file(path: str, project_dir: str) -> str:
    """Read a file relative to project_dir."""
    if not path:
        return "Error: no path provided"
    full_path = os.path.join(project_dir, path)
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"


def _tool_write_file(path: str, content: str, project_dir: str) -> str:
    """Write content to a file relative to project_dir."""
    if not path:
        return "Error: no path provided"
    full_path = os.path.join(project_dir, path)
    try:
        parent = os.path.dirname(full_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"OK: wrote {len(content.encode('utf-8'))} bytes to {path}"
    except OSError as e:
        return f"Error writing {path}: {e}"


_BLOCKED_MSG = (
    "Blocked: server-starting commands are not allowed. "
    "Use npm test or node -e instead."
)


def _tool_run_command(command: str, project_dir: str) -> str:
    """Run a shell command with 30s timeout, truncate output to 5000 chars."""
    if not command:
        return "Error: no command provided"
    # Block server-starting commands
    cmd_lower = command.lower()
    if "index.js" in cmd_lower and "test" not in cmd_lower:
        return _BLOCKED_MSG
    if "npm start" in cmd_lower or "node server.js" in cmd_lower:
        return _BLOCKED_MSG
    if "node" in cmd_lower and "require" in cmd_lower:
        return (
            "Blocked: node require() commands can hang. "
            "Use npm test to verify code works."
        )
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
        output = (result.stdout or "") + (result.stderr or "")
        if len(output) > 5000:
            output = output[:5000] + "\n... [truncated]"
        if not output.strip():
            output = f"(exit code {result.returncode}, no output)"
        return output
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 30 seconds"
    except OSError as e:
        return f"Error running command: {e}"


# --- Agent Loop ---


def run_agent(
    task_prompt: str,
    project_dir: str,
    model: str = "claude-sonnet-4-20250514",
    max_turns: int = 30,
    compression_mode: str = "none",
    compression_threshold: int = 30000,
    compression_model: str = "gemma4:26b",
    retrieval_top_k: int = 5,
    use_caching: bool = False,
    use_thinking: bool = False,
    transcript_path: str | None = None,
) -> dict:
    """
    Run a tool-use agent loop to complete a coding task.

    Args:
        task_prompt: The task for the agent to complete
        project_dir: Working directory for file operations
        model: Anthropic model ID
        max_turns: Maximum conversation turns
        compression_mode: "none", "api", "ollama", or "retrieval"
        compression_threshold: Compress when total input tokens exceeds this
        compression_model: Ollama model for "ollama" compression mode
        retrieval_top_k: Number of turns to retrieve in "retrieval" mode
        use_caching: Enable prompt caching (cache system prompt)
        use_thinking: Enable extended thinking (budget: 10K tokens)
        transcript_path: If set, write full conversation as JSONL to this path
            (one message dict per line, system prompt first)

    Returns:
        Dict with status, summary, token counts, cost, timing, etc.
    """
    project_dir = os.path.abspath(project_dir)
    api_key = load_api_key(project_dir)
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or .env file")
        return {
            "status": "error",
            "summary": "No API key",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "total_thinking_tokens": 0,
            "turns_used": 0,
            "per_turn_input_tokens": [],
            "per_turn_output_tokens": [],
            "per_turn_cache_read_tokens": [],
            "per_turn_cache_creation_tokens": [],
            "per_turn_thinking_tokens": [],
            "compressions": [],
            "retrievals": [],
            "cost_usd": 0.0,
            "wall_time_s": 0.0,
        }

    messages = [{"role": "user", "content": task_prompt}]
    per_turn_input = []
    per_turn_output = []
    per_turn_cache_read = []
    per_turn_cache_creation = []
    per_turn_thinking = []
    compressions = []
    retrievals = []
    final_text = ""
    status = "completed"
    just_compressed = False

    # Initialize retriever if using retrieval mode
    retriever = None
    if compression_mode == "retrieval":
        from retriever import TurnRetriever

        retriever = TurnRetriever()

    t_start = time.time()

    for turn in range(max_turns):
        # Call the API
        try:
            response = _call_api(
                messages=messages,
                model=model,
                api_key=api_key,
                tools=TOOL_DEFINITIONS,
                use_caching=use_caching,
                use_thinking=use_thinking,
            )
        except RuntimeError as e:
            print(f"  API error on turn {turn + 1}: {e}")
            status = "error"
            final_text = str(e)
            break

        # Track token usage
        usage = response.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)

        # Total input includes all categories (uncached + cache_read + cache_creation)
        total_input_this_turn = input_tokens + cache_read + cache_creation

        per_turn_input.append(total_input_this_turn)
        per_turn_output.append(output_tokens)
        per_turn_cache_read.append(cache_read)
        per_turn_cache_creation.append(cache_creation)
        per_turn_thinking.append(thinking_tokens)

        # Print full usage on first two turns for debugging cache behavior
        if turn < 2:
            print(f"  [DEBUG] Turn {turn + 1} usage: {usage}")
            if turn == 0 and use_caching:
                print(
                    "  [DEBUG] Caching enabled: system=content_block_array, "
                    "beta_header=prompt-caching-2024-07-31, "
                    "cache_breakpoints=system+last_tool+last_message"
                )

        # Build status line
        parts = [f"  Turn {turn + 1}: {total_input_this_turn:,} in"]
        if use_caching:
            parts[0] += f" ({cache_read:,} cached)"
        parts.append(f" / {output_tokens:,} out")
        if use_thinking and thinking_tokens > 0:
            parts.append(f" ({thinking_tokens:,} thinking)")
        note = ""
        if just_compressed:
            note = " [cache reset]"
            just_compressed = False
        if retriever is not None:
            note += f" [stored: {len(retriever.turns)}]"
        parts.append(f"  (stop: {response.get('stop_reason', '?')}){note}")
        print("".join(parts))

        # Process response content — preserve thinking blocks for conversation history
        content = response.get("content", [])
        tool_calls = []
        for block in content:
            if block.get("type") in ("thinking", "redacted_thinking"):
                pass  # Internal reasoning — track via usage, don't act
            elif block.get("type") == "text":
                final_text = block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        # If no tool calls, agent is done — record the final assistant turn
        if response.get("stop_reason") == "end_turn" or not tool_calls:
            messages.append({"role": "assistant", "content": content})
            break

        # Execute tools and build tool_result messages
        # Preserve full content (including thinking blocks) — API requires them
        assistant_msg = {"role": "assistant", "content": content}
        messages.append(assistant_msg)

        tool_results = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_input = tc.get("input", {})
            tool_id = tc.get("id", "")

            print(f"    -> {tool_name}({_summarize_input(tool_name, tool_input)})")
            result_text = _execute_tool(tool_name, tool_input, project_dir)

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_text,
                }
            )

        messages.append({"role": "user", "content": tool_results})

        # Store turn in retriever (before potential retrieval replaces messages)
        if retriever is not None:
            retriever.store_turn(turn + 1, assistant_msg, tool_results)

        # Check if we should compress/retrieve (use total input for threshold)
        if (
            compression_mode == "retrieval"
            and total_input_this_turn > compression_threshold
        ):
            print(
                f"  Retrieving top-{retrieval_top_k} turns "
                f"(total_input={total_input_this_turn:,} > "
                f"threshold={compression_threshold:,})..."
            )
            new_messages, ret_metrics = retriever.build_retrieved_history(
                original_task_prompt=task_prompt, top_k=retrieval_top_k
            )
            retrievals.append(
                {
                    "turn": turn + 1,
                    "tokens_before": total_input_this_turn,
                    "total_stored": ret_metrics["total_stored"],
                    "retrieved": ret_metrics["retrieved"],
                    "scores": ret_metrics["scores"],
                    "ollama_available": ret_metrics["ollama_available"],
                }
            )
            messages = new_messages
            just_compressed = True
            print(
                f"  Retrieved {ret_metrics['retrieved']}/{ret_metrics['total_stored']}"
                f" turns (ollama={'yes' if ret_metrics['ollama_available'] else 'no'})"
            )
        elif (
            compression_mode != "none"
            and compression_mode != "retrieval"
            and total_input_this_turn > compression_threshold
        ):
            print(
                f"  Compressing history (total_input={total_input_this_turn:,} > "
                f"threshold={compression_threshold:,})..."
            )
            from compressor import compress_history

            new_messages, comp_metrics = compress_history(
                messages=messages,
                original_task_prompt=task_prompt,
                mode=compression_mode,
                api_key=api_key,
                ollama_model=compression_model,
            )
            compressions.append(
                {
                    "turn": turn + 1,
                    "tokens_before": total_input_this_turn,
                    "tokens_after": comp_metrics.get("tokens_after", 0),
                    "compression_tokens": comp_metrics.get("compression_tokens", 0),
                }
            )
            messages = new_messages
            just_compressed = True  # Next turn will note cache invalidation
            print(
                f"  Compressed: {total_input_this_turn:,} -> "
                f"~{comp_metrics.get('tokens_after', 0):,} tokens"
            )
    else:
        status = "max_turns"

    wall_time = time.time() - t_start
    total_input = sum(per_turn_input)
    total_output = sum(per_turn_output)
    total_cache_read = sum(per_turn_cache_read)
    total_cache_creation = sum(per_turn_cache_creation)
    total_thinking = sum(per_turn_thinking)

    # Cost: input_tokens from API is uncached; derive from totals
    total_uncached = total_input - total_cache_read - total_cache_creation
    cost = calculate_cost(
        total_uncached,
        total_output,
        total_cache_read,
        total_cache_creation,
        total_thinking,
    )

    if transcript_path:
        try:
            parent = os.path.dirname(transcript_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"role": "system", "content": SYSTEM_PROMPT}) + "\n")
                for msg in messages:
                    f.write(json.dumps(msg) + "\n")
        except OSError as e:
            print(f"  Warning: failed to write transcript {transcript_path}: {e}")

    return {
        "status": status,
        "summary": final_text[:2000],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "total_cache_read_tokens": total_cache_read,
        "total_cache_creation_tokens": total_cache_creation,
        "total_thinking_tokens": total_thinking,
        "turns_used": len(per_turn_input),
        "per_turn_input_tokens": per_turn_input,
        "per_turn_output_tokens": per_turn_output,
        "per_turn_cache_read_tokens": per_turn_cache_read,
        "per_turn_cache_creation_tokens": per_turn_cache_creation,
        "per_turn_thinking_tokens": per_turn_thinking,
        "compressions": compressions,
        "retrievals": retrievals,
        "cost_usd": round(cost, 4),
        "wall_time_s": round(wall_time, 1),
    }


def _summarize_input(tool_name: str, tool_input: dict) -> str:
    """Short summary of tool input for logging."""
    if tool_name == "read_file":
        return tool_input.get("path", "?")
    elif tool_name == "write_file":
        path = tool_input.get("path", "?")
        size = len(tool_input.get("content", ""))
        return f"{path}, {size} chars"
    elif tool_name == "run_command":
        cmd = tool_input.get("command", "?")
        return cmd[:80]
    return str(tool_input)[:80]
