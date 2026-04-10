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
    "summary of what you built."
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

# Sonnet pricing: $3/MTok input, $15/MTok output
COST_INPUT_PER_MTOK = 3.0
COST_OUTPUT_PER_MTOK = 15.0


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
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("ANTHROPIC_API_KEY="):
                        val = line.split("=", 1)[1].strip()
                        # Strip quotes if present
                        if len(val) >= 2 and val[0] in ('"', "'") and val[-1] == val[0]:
                            val = val[1:-1]
                        if val:
                            return val
        except OSError:
            pass

    return None


def _call_api(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str = SYSTEM_PROMPT,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
) -> dict:
    """
    Send a request to the Anthropic Messages API.
    Returns the parsed response dict.
    Raises RuntimeError on API errors.
    """
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages,
    }
    if tools:
        body["tools"] = tools

    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        ANTHROPIC_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
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


def _tool_run_command(command: str, project_dir: str) -> str:
    """Run a shell command with 30s timeout, truncate output to 5000 chars."""
    if not command:
        return "Error: no command provided"
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_dir,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
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
) -> dict:
    """
    Run a tool-use agent loop to complete a coding task.

    Args:
        task_prompt: The task for the agent to complete
        project_dir: Working directory for file operations
        model: Anthropic model ID
        max_turns: Maximum conversation turns
        compression_mode: "none", "api", or "ollama"
        compression_threshold: Compress when input_tokens exceeds this
        compression_model: Ollama model for "ollama" compression mode

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
            "turns_used": 0,
            "per_turn_input_tokens": [],
            "per_turn_output_tokens": [],
            "compressions": [],
            "cost_usd": 0.0,
            "wall_time_s": 0.0,
        }

    messages = [{"role": "user", "content": task_prompt}]
    per_turn_input = []
    per_turn_output = []
    compressions = []
    final_text = ""
    status = "completed"

    t_start = time.time()

    for turn in range(max_turns):
        # Call the API
        try:
            response = _call_api(
                messages=messages,
                model=model,
                api_key=api_key,
                tools=TOOL_DEFINITIONS,
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
        per_turn_input.append(input_tokens)
        per_turn_output.append(output_tokens)

        print(
            f"  Turn {turn + 1}: {input_tokens:,} in / {output_tokens:,} out"
            f"  (stop: {response.get('stop_reason', '?')})"
        )

        # Process response content
        content = response.get("content", [])
        tool_calls = []
        for block in content:
            if block.get("type") == "text":
                final_text = block.get("text", "")
            elif block.get("type") == "tool_use":
                tool_calls.append(block)

        # If no tool calls, agent is done
        if response.get("stop_reason") == "end_turn" or not tool_calls:
            break

        # Execute tools and build tool_result messages
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

        # Check if we should compress
        if compression_mode != "none" and input_tokens > compression_threshold:
            print(
                f"  Compressing history (input_tokens={input_tokens:,} > "
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
                    "tokens_before": input_tokens,
                    "tokens_after": comp_metrics.get("tokens_after", 0),
                    "compression_tokens": comp_metrics.get("compression_tokens", 0),
                }
            )
            messages = new_messages
            print(
                f"  Compressed: {input_tokens:,} -> "
                f"~{comp_metrics.get('tokens_after', 0):,} tokens"
            )
    else:
        status = "max_turns"

    wall_time = time.time() - t_start
    total_input = sum(per_turn_input)
    total_output = sum(per_turn_output)
    cost = (
        total_input * COST_INPUT_PER_MTOK / 1_000_000
        + total_output * COST_OUTPUT_PER_MTOK / 1_000_000
    )

    return {
        "status": status,
        "summary": final_text[:2000],
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "turns_used": len(per_turn_input),
        "per_turn_input_tokens": per_turn_input,
        "per_turn_output_tokens": per_turn_output,
        "compressions": compressions,
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
