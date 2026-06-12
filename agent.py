"""
Bearing - Minimal code agent with tool use

A provider-agnostic tool-use agent loop using API calls directly via urllib.
Supports OpenAI Responses and Anthropic Messages backends, plus
mid-conversation context compression to reduce token accumulation across turns.

Tools:
    read_file(path)      - Read a file relative to the working directory
    write_file(path, content) - Write a file, creating parent dirs
    run_command(command)  - Run a shell command with 30s timeout
"""

import json
import os
import subprocess
import time
import urllib.error
import urllib.request

OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"

DEFAULT_PROVIDER = "openai"
DEFAULT_MODELS = {
    "openai": "gpt-5.5",
    "anthropic": "claude-sonnet-4-20250514",
}
DEFAULT_MODEL = DEFAULT_MODELS[DEFAULT_PROVIDER]

SYSTEM_PROMPT = (
    "You are a coding agent. You have access to three tools: read_file, "
    "write_file, and run_command. Complete the given task by reading existing "
    "files, writing new files, and running commands as needed. Work efficiently "
    "- read only the files you need, and don't re-read files you've already "
    "seen unless they've changed. When you're done, respond with a brief "
    "summary of what you built. "
    "NEVER run commands that start servers or long-running processes "
    "(npm start, node server.js, node index.js, node src/index.js). "
    "These will timeout and waste turns. Only run commands that exit on "
    "their own like npm test, node -e, or ls."
)

BASE_TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": (
            "Read the contents of a file at the given path "
            "relative to the working directory"
        ),
        "schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative file path to read",
                }
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "write_file",
        "description": (
            "Write content to a file at the given path relative to the "
            "working directory. Creates parent directories if needed."
        ),
        "schema": {
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
            "additionalProperties": False,
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return stdout+stderr. "
            "Has a 30-second timeout. Output truncated to 5000 chars."
        ),
        "schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to execute",
                }
            },
            "required": ["command"],
            "additionalProperties": False,
        },
    },
]

MODEL_PRICING = {
    # Prices per 1M tokens. OpenAI output tokens include reasoning tokens.
    ("openai", "gpt-5.5"): {
        "input": 5.0,
        "cached_input": 0.50,
        "cache_creation": 0.0,
        "output": 30.0,
        "reasoning_billed_separately": False,
    },
    ("openai", "gpt-5.4-mini"): {
        "input": 0.75,
        "cached_input": 0.075,
        "cache_creation": 0.0,
        "output": 4.50,
        "reasoning_billed_separately": False,
    },
    ("anthropic", "claude-sonnet-4-20250514"): {
        "input": 3.0,
        "cached_input": 0.30,
        "cache_creation": 3.75,
        "output": 15.0,
        "reasoning_billed_separately": True,
    },
}

# Backward-compatible constants used by eval reporting. These reflect the
# default OpenAI model; provider-specific runs should use calculate_cost().
COST_INPUT_PER_MTOK = MODEL_PRICING[(DEFAULT_PROVIDER, DEFAULT_MODEL)]["input"]
COST_OUTPUT_PER_MTOK = MODEL_PRICING[(DEFAULT_PROVIDER, DEFAULT_MODEL)]["output"]
COST_CACHE_READ_PER_MTOK = MODEL_PRICING[(DEFAULT_PROVIDER, DEFAULT_MODEL)][
    "cached_input"
]
COST_CACHE_CREATION_PER_MTOK = MODEL_PRICING[(DEFAULT_PROVIDER, DEFAULT_MODEL)][
    "cache_creation"
]


def _normalize_provider(provider: str | None) -> str:
    value = (provider or DEFAULT_PROVIDER).lower()
    aliases = {
        "responses": "openai",
        "openai-responses": "openai",
        "anthropic-messages": "anthropic",
    }
    value = aliases.get(value, value)
    if value not in DEFAULT_MODELS:
        raise ValueError(
            f"unsupported provider '{provider}'. "
            f"Expected one of: {', '.join(DEFAULT_MODELS)}"
        )
    return value


def _format_tools(provider: str) -> list[dict]:
    """Format neutral tool specs for the target provider."""
    tools = []
    for tool in BASE_TOOL_DEFINITIONS:
        if provider == "openai":
            tools.append(
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["schema"],
                    "strict": True,
                }
            )
        elif provider == "anthropic":
            schema = {
                key: value
                for key, value in tool["schema"].items()
                if key != "additionalProperties"
            }
            tools.append(
                {
                    "name": tool["name"],
                    "description": tool["description"],
                    "input_schema": schema,
                }
            )
    return tools


TOOL_DEFINITIONS = _format_tools(DEFAULT_PROVIDER)


def _pricing_for(provider: str, model: str) -> dict:
    return MODEL_PRICING.get(
        (provider, model),
        MODEL_PRICING.get((provider, DEFAULT_MODELS[provider])),
    )


def pricing_for(provider: str = DEFAULT_PROVIDER, model: str | None = None) -> dict:
    """Return the pricing table entry used for approximate cost calculations."""
    provider = _normalize_provider(provider)
    return dict(_pricing_for(provider, model or DEFAULT_MODELS[provider]))


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    cache_read: int = 0,
    cache_creation: int = 0,
    thinking: int = 0,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> float:
    """
    Calculate approximate cost with provider-specific cache/reasoning pricing.
    input_tokens is the uncached portion derived from API usage totals.
    """
    provider = _normalize_provider(provider)
    model = model or DEFAULT_MODELS[provider]
    pricing = _pricing_for(provider, model)
    billed_output = output_tokens
    if pricing.get("reasoning_billed_separately", False):
        billed_output += thinking
    return (
        input_tokens * pricing["input"] / 1_000_000
        + cache_read * pricing["cached_input"] / 1_000_000
        + cache_creation * pricing["cache_creation"] / 1_000_000
        + billed_output * pricing["output"] / 1_000_000
    )


def _read_env_file_key(project_dir: str, env_name: str) -> str | None:
    env_path = os.path.join(project_dir, ".env")
    if not os.path.exists(env_path):
        print(f"Note: No .env file at {env_path}")
        return None

    for encoding in ("utf-8-sig", "utf-16"):
        try:
            with open(env_path, "r", encoding=encoding) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{env_name}="):
                        val = line.split("=", 1)[1].strip()
                        if (
                            len(val) >= 2
                            and val[0] in ('"', "'")
                            and val[-1] == val[0]
                        ):
                            val = val[1:-1]
                        if val:
                            return val
            break
        except UnicodeDecodeError:
            continue
        except OSError:
            break
    print(f"Warning: .env file found at {env_path} but no {env_name} in it")
    return None


def load_api_key(project_dir: str = ".", provider: str = DEFAULT_PROVIDER) -> str | None:
    """
    Load provider API key from environment or .env file.
    Returns None if not found.
    """
    provider = _normalize_provider(provider)
    env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ.get(env_name)
    if key:
        return key
    return _read_env_file_key(project_dir, env_name)


def _call_api(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str = SYSTEM_PROMPT,
    tools: list[dict] | None = None,
    max_tokens: int = 4096,
    use_caching: bool = False,
    use_thinking: bool = False,
    reasoning_effort: str | None = None,
    provider: str = DEFAULT_PROVIDER,
) -> dict:
    """
    Send a request to the configured model API.
    Returns the parsed response dict.
    Raises RuntimeError on API errors.
    """
    provider = _normalize_provider(provider)
    if provider == "openai":
        return _call_openai_api(
            messages=messages,
            model=model,
            api_key=api_key,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            use_caching=use_caching,
            use_thinking=use_thinking,
            reasoning_effort=reasoning_effort,
        )
    if provider == "anthropic":
        return _call_anthropic_api(
            messages=messages,
            model=model,
            api_key=api_key,
            system=system,
            tools=tools,
            max_tokens=max_tokens,
            use_caching=use_caching,
            use_thinking=use_thinking,
            reasoning_effort=reasoning_effort,
        )
    raise AssertionError(f"unhandled provider: {provider}")


def _open_urlopen_json(
    url: str,
    payload: bytes,
    headers: dict,
    error_prefix: str,
    timeout: int = 300,
) -> dict:
    req = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    max_retries = 3
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                req = urllib.request.Request(
                    url, data=payload, headers=headers, method="POST"
                )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504, 529) and attempt < max_retries:
                retry_after = e.headers.get("retry-after")
                wait = int(retry_after) if retry_after else min(60, 2**attempt * 5)
                print(
                    f"  API transient error ({e.code}), waiting {wait}s... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{error_prefix} {e.code}: {error_body[:500]}") from None
        except urllib.error.URLError as e:
            if attempt < max_retries:
                wait = min(60, 2**attempt * 5)
                print(
                    f"  API connection error ({e.reason}), waiting {wait}s... "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                time.sleep(wait)
                continue
            raise RuntimeError(f"{error_prefix} connection error: {e.reason}") from None
    raise RuntimeError(f"{error_prefix}: retry loop exhausted")


def _call_openai_api(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    tools: list[dict] | None,
    max_tokens: int,
    use_caching: bool,
    use_thinking: bool,
    reasoning_effort: str | None,
) -> dict:
    body = {
        "model": model,
        "instructions": system,
        "input": messages,
        "max_output_tokens": max_tokens,
        "store": False,
    }
    if tools:
        body["tools"] = tools
        body["tool_choice"] = "auto"
    if use_caching:
        body["prompt_cache_retention"] = "24h"
    effort = reasoning_effort or ("high" if use_thinking else None)
    if effort:
        body["reasoning"] = {"effort": effort}

    return _open_urlopen_json(
        OPENAI_RESPONSES_API_URL,
        json.dumps(body).encode("utf-8"),
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        "OpenAI API error",
    )


def _call_anthropic_api(
    messages: list[dict],
    model: str,
    api_key: str,
    system: str,
    tools: list[dict] | None,
    max_tokens: int,
    use_caching: bool,
    use_thinking: bool,
    reasoning_effort: str | None,
) -> dict:
    system_value: str | list[dict]
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
            cached_tools = list(tools)
            cached_tools[-1] = {
                **cached_tools[-1],
                "cache_control": {"type": "ephemeral"},
            }
            body["tools"] = cached_tools
        else:
            body["tools"] = tools
    if use_thinking or reasoning_effort:
        body["thinking"] = {"type": "enabled", "budget_tokens": 10000}
        if body["max_tokens"] < 16000:
            body["max_tokens"] = 16000

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

    return _open_urlopen_json(
        ANTHROPIC_API_URL,
        json.dumps(body).encode("utf-8"),
        {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            **({"anthropic-beta": "prompt-caching-2024-07-31"} if use_caching else {}),
        },
        "Anthropic API error",
    )


def _usage_counts(usage: dict, provider: str) -> tuple[int, int, int, int, int]:
    """
    Normalize provider usage fields.
    Returns total_input, output, cache_read, cache_creation, reasoning_tokens.
    """
    if provider == "anthropic":
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        thinking_tokens = usage.get("thinking_tokens", 0)
        return (
            input_tokens + cache_read + cache_creation,
            output_tokens,
            cache_read,
            cache_creation,
            thinking_tokens,
        )

    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    input_details = usage.get("input_tokens_details") or usage.get(
        "prompt_tokens_details", {}
    )
    output_details = usage.get("output_tokens_details") or usage.get(
        "completion_tokens_details", {}
    )
    cache_read = input_details.get("cached_tokens", 0)
    reasoning_tokens = output_details.get(
        "reasoning_tokens", usage.get("reasoning_tokens", 0)
    )
    return input_tokens, output_tokens, cache_read, 0, reasoning_tokens


def _extract_openai_text(response: dict) -> str:
    text = response.get("output_text", "")
    if text:
        return text

    parts = []
    for item in response.get("output", []):
        if item.get("type") == "message":
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    parts.append(block.get("text", ""))
    return "".join(parts)


def _extract_anthropic_text(response: dict) -> str:
    parts = []
    for block in response.get("content", []):
        if block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts)


def _parse_tool_arguments(raw_arguments: object) -> dict:
    """Parse function arguments, tolerating malformed model output."""
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if not isinstance(raw_arguments, str) or not raw_arguments.strip():
        return {}
    try:
        value = json.loads(raw_arguments)
    except json.JSONDecodeError:
        return {"_raw_arguments": raw_arguments}
    return value if isinstance(value, dict) else {"_value": value}


def _normalize_response(
    response: dict,
    provider: str,
) -> tuple[list[dict], list[dict], str, str]:
    """
    Normalize provider response into appendable message items, tool calls,
    final text, and a status label.
    """
    if provider == "anthropic":
        content = response.get("content", [])
        tool_calls = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "name": block.get("name", ""),
                        "input": block.get("input", {}),
                        "call_id": block.get("id", ""),
                    }
                )
        return (
            [{"role": "assistant", "content": content}],
            tool_calls,
            _extract_anthropic_text(response),
            response.get("stop_reason", "?"),
        )

    output_items = response.get("output", [])
    tool_calls = []
    for item in output_items:
        if item.get("type") != "function_call":
            continue
        tool_calls.append(
            {
                "name": item.get("name", ""),
                "input": _parse_tool_arguments(item.get("arguments", "{}")),
                "call_id": item.get("call_id", ""),
            }
        )
    return output_items, tool_calls, _extract_openai_text(response), response.get(
        "status", "?"
    )


def _build_tool_output_item(provider: str, call_id: str, result_text: str) -> dict:
    if provider == "anthropic":
        return {
            "type": "tool_result",
            "tool_use_id": call_id,
            "content": result_text,
        }
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": result_text,
    }


# --- Tool Execution ---


def _execute_tool(name: str, input_data: dict, project_dir: str) -> str:
    """Execute a tool call and return the result string."""
    if "_raw_arguments" in input_data:
        return f"Error: invalid JSON arguments: {input_data['_raw_arguments']}"
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


def _resolve_project_path(path: str, project_dir: str) -> tuple[str | None, str | None]:
    """Resolve a model-supplied relative path and keep it inside project_dir."""
    if not path:
        return None, "Error: no path provided"
    if os.path.isabs(path):
        return None, "Error: absolute paths are not allowed"

    project_root = os.path.realpath(project_dir)
    full_path = os.path.realpath(os.path.join(project_root, path))
    try:
        common = os.path.commonpath(
            [os.path.normcase(project_root), os.path.normcase(full_path)]
        )
    except ValueError:
        return None, f"Error: path escapes project directory: {path}"

    if common != os.path.normcase(project_root):
        return None, f"Error: path escapes project directory: {path}"
    return full_path, None


def _tool_read_file(path: str, project_dir: str) -> str:
    """Read a file relative to project_dir."""
    full_path, error = _resolve_project_path(path, project_dir)
    if error:
        return error
    try:
        with open(full_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"


def _tool_write_file(path: str, content: str, project_dir: str) -> str:
    """Write content to a file relative to project_dir."""
    full_path, error = _resolve_project_path(path, project_dir)
    if error:
        return error
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

_DESTRUCTIVE_COMMAND_PATTERNS = [
    " rm -rf",
    " rm -fr",
    "rm -rf ",
    "rm -fr ",
    "del /s",
    "rmdir /s",
    "remove-item",
    "git reset",
    "git clean",
    "mkfs",
    "format.com",
    ":(){",
]


def _tool_run_command(command: str, project_dir: str) -> str:
    """Run a shell command with 30s timeout, truncate output to 5000 chars."""
    if not command:
        return "Error: no command provided"
    cmd_lower = command.lower()
    padded_cmd = f" {cmd_lower} "
    if any(pattern in padded_cmd for pattern in _DESTRUCTIVE_COMMAND_PATTERNS):
        return (
            "Blocked: destructive commands are not allowed through run_command. "
            "Use focused test, lint, or inspection commands instead."
        )
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


def _empty_result(status: str, summary: str) -> dict:
    return {
        "status": status,
        "summary": summary,
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


# --- Agent Loop ---


def run_agent(
    task_prompt: str,
    project_dir: str,
    model: str | None = None,
    max_turns: int = 30,
    compression_mode: str = "none",
    compression_threshold: int = 30000,
    compression_model: str = "gemma4:26b",
    retrieval_top_k: int = 5,
    use_caching: bool = False,
    use_thinking: bool = False,
    reasoning_effort: str | None = None,
    provider: str = DEFAULT_PROVIDER,
    transcript_path: str | None = None,
) -> dict:
    """
    Run a tool-use agent loop to complete a coding task.

    Args:
        task_prompt: The task for the agent to complete
        project_dir: Working directory for file operations
        model: Provider model ID. Defaults by provider if omitted.
        max_turns: Maximum conversation turns
        compression_mode: "none", "api", "ollama", or "retrieval"
        compression_threshold: Compress when total input tokens exceeds this
        compression_model: Ollama model for "ollama" compression mode
        retrieval_top_k: Number of turns to retrieve in "retrieval" mode
        use_caching: Enable provider-specific prompt caching
        use_thinking: Backward-compatible alias for high reasoning/thinking effort
        reasoning_effort: Optional OpenAI effort ("low", "medium", "high")
        provider: "openai" or "anthropic"
        transcript_path: If set, write full input history as JSONL to this path
            (one item dict per line, system prompt first)

    Returns:
        Dict with status, summary, token counts, cost, timing, etc.
    """
    project_dir = os.path.abspath(project_dir)
    try:
        provider = _normalize_provider(provider)
    except ValueError as e:
        print(f"Error: {e}")
        return _empty_result("error", str(e))

    if (
        provider == "openai"
        and reasoning_effort is not None
        and reasoning_effort not in {"low", "medium", "high"}
    ):
        print("Error: reasoning_effort must be one of: low, medium, high")
        return _empty_result("error", "Invalid reasoning_effort")

    model = model or os.environ.get("BEARING_AGENT_MODEL") or DEFAULT_MODELS[provider]
    api_key = load_api_key(project_dir, provider)
    if not api_key:
        env_name = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        print(f"Error: {env_name} not found in environment or .env file")
        return _empty_result("error", "No API key")

    messages = [{"role": "user", "content": task_prompt}]
    tools = _format_tools(provider)
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

    retriever = None
    if compression_mode == "retrieval":
        from retriever import TurnRetriever

        retriever = TurnRetriever()

    t_start = time.time()

    for turn in range(max_turns):
        try:
            response = _call_api(
                messages=messages,
                model=model,
                api_key=api_key,
                tools=tools,
                use_caching=use_caching,
                use_thinking=use_thinking,
                reasoning_effort=reasoning_effort,
                provider=provider,
            )
        except RuntimeError as e:
            print(f"  API error on turn {turn + 1}: {e}")
            status = "error"
            final_text = str(e)
            break

        usage = response.get("usage", {})
        (
            total_input_this_turn,
            output_tokens,
            cache_read,
            cache_creation,
            thinking_tokens,
        ) = _usage_counts(usage, provider)

        per_turn_input.append(total_input_this_turn)
        per_turn_output.append(output_tokens)
        per_turn_cache_read.append(cache_read)
        per_turn_cache_creation.append(cache_creation)
        per_turn_thinking.append(thinking_tokens)

        if turn < 2:
            print(f"  [DEBUG] Turn {turn + 1} usage: {usage}")
            if turn == 0 and use_caching:
                if provider == "openai":
                    print(
                        "  [DEBUG] Caching enabled: OpenAI automatic prompt cache, "
                        "prompt_cache_retention=24h"
                    )
                else:
                    print(
                        "  [DEBUG] Caching enabled: Anthropic cache_control "
                        "breakpoints"
                    )

        append_items, tool_calls, response_text, status_label = _normalize_response(
            response, provider
        )
        final_text = response_text or final_text

        parts = [f"  Turn {turn + 1}: {total_input_this_turn:,} in"]
        if use_caching:
            parts[0] += f" ({cache_read:,} cached)"
        parts.append(f" / {output_tokens:,} out")
        if (use_thinking or reasoning_effort) and thinking_tokens > 0:
            parts.append(f" ({thinking_tokens:,} reasoning)")
        note = ""
        if just_compressed:
            note = " [cache reset]"
            just_compressed = False
        if retriever is not None:
            note += f" [stored: {len(retriever.turns)}]"
        parts.append(f"  (status: {status_label}){note}")
        print("".join(parts))

        if not tool_calls:
            messages.extend(append_items)
            break

        messages.extend(append_items)

        raw_tool_outputs = []
        for tc in tool_calls:
            tool_name = tc.get("name", "")
            tool_input = tc.get("input", {})
            call_id = tc.get("call_id", "")

            print(f"    -> {tool_name}({_summarize_input(tool_name, tool_input)})")
            result_text = _execute_tool(tool_name, tool_input, project_dir)
            raw_tool_outputs.append(
                _build_tool_output_item(provider, call_id, result_text)
            )

        if provider == "anthropic":
            output_append_items = [{"role": "user", "content": raw_tool_outputs}]
        else:
            output_append_items = raw_tool_outputs
        messages.extend(output_append_items)

        if retriever is not None:
            assistant_for_retriever = (
                append_items[0] if provider == "anthropic" else append_items
            )
            retriever.store_turn(
                turn + 1,
                assistant_for_retriever,
                raw_tool_outputs,
            )

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
                api_provider=provider,
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
            just_compressed = True
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

    total_uncached = total_input - total_cache_read - total_cache_creation
    cost = calculate_cost(
        total_uncached,
        total_output,
        total_cache_read,
        total_cache_creation,
        total_thinking,
        provider=provider,
        model=model,
    )

    if transcript_path:
        try:
            parent = os.path.dirname(transcript_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "role": "system",
                            "provider": provider,
                            "model": model,
                            "content": SYSTEM_PROMPT,
                        }
                    )
                    + "\n"
                )
                for msg in messages:
                    f.write(json.dumps(msg) + "\n")
        except OSError as e:
            print(f"  Warning: failed to write transcript {transcript_path}: {e}")

    return {
        "status": status,
        "summary": final_text[:2000],
        "provider": provider,
        "model": model,
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
    if "_raw_arguments" in tool_input:
        return "invalid JSON"
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
