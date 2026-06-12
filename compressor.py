"""
Bearing - Mid-conversation history compression

Compresses a tool-use conversation into a concise summary, then
replaces the full message history with a single message containing
the original task prompt plus the summary. This resets input token
count from ~30-50K back to ~2-3K.

Two backends:
    api    — Compress via the configured model API. Costs money, high quality.
    ollama — Compress via local Ollama model (Gemma 4). Free, good quality.
             Falls back to api mode if Ollama is unavailable.
"""

import json
import urllib.error
import urllib.request

OPENAI_RESPONSES_API_URL = "https://api.openai.com/v1/responses"
OPENAI_COMPRESSION_MODEL = "gpt-5.4-mini"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_COMPRESSION_MODEL = "claude-sonnet-4-20250514"

OLLAMA_BASE = "http://localhost:11434"

COMPRESSION_PROMPT = (
    "Here is a conversation between a user and a coding agent. "
    "Summarize what has been accomplished. Include: "
    "(1) every file that was read and its key contents, "
    "(2) every file that was written or modified with the essential code/logic, "
    "(3) every command run and whether it succeeded or failed, "
    "(4) any errors encountered and how they were resolved, "
    "(5) what remains to be done. "
    "Be thorough but concise — preserve all details needed to continue the work."
)


def _serialize_messages(messages: list[dict]) -> str:
    """
    Serialize conversation messages into a readable text format
    for the compression prompt.
    """
    parts = []
    for msg in messages:
        item_type = msg.get("type", "")

        # Responses API output/input items are flat in the input list.
        if item_type == "message":
            role = msg.get("role", "assistant")
            for block in msg.get("content", []):
                if isinstance(block, dict) and block.get("type") in (
                    "output_text",
                    "text",
                    "input_text",
                ):
                    parts.append(f"[{role}]: {block.get('text', '')}")
            continue
        if item_type == "function_call":
            name = msg.get("name", "?")
            inp = msg.get("arguments", "{}")
            parts.append(f"[assistant tool_call]: {name}({inp})")
            continue
        if item_type == "function_call_output":
            result_content = msg.get("output", "")
            if isinstance(result_content, str):
                if len(result_content) > 1000:
                    result_content = result_content[:1000] + "... [truncated]"
                parts.append(f"[tool_result]: {result_content}")
            continue

        role = msg.get("role", "?")
        content = msg.get("content", "")

        if isinstance(content, str):
            parts.append(f"[{role}]: {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        parts.append(f"[{role}]: {block.get('text', '')}")
                    elif btype == "tool_use":
                        name = block.get("name", "?")
                        inp = json.dumps(block.get("input", {}), separators=(",", ":"))
                        parts.append(f"[{role} tool_call]: {name}({inp})")
                    elif btype == "tool_result":
                        result_content = block.get("content", "")
                        if isinstance(result_content, str):
                            # Truncate long tool results in the serialization
                            if len(result_content) > 1000:
                                result_content = (
                                    result_content[:1000] + "... [truncated]"
                                )
                            parts.append(f"[tool_result]: {result_content}")
                elif isinstance(block, str):
                    parts.append(f"[{role}]: {block}")

    return "\n\n".join(parts)


def _normalize_provider(provider: str | None) -> str:
    value = (provider or "openai").lower()
    aliases = {
        "responses": "openai",
        "openai-responses": "openai",
        "anthropic-messages": "anthropic",
    }
    value = aliases.get(value, value)
    if value not in {"openai", "anthropic"}:
        raise ValueError("api_provider must be 'openai' or 'anthropic'")
    return value


def _compress_via_openai(
    conversation_text: str,
    api_key: str,
    model: str = OPENAI_COMPRESSION_MODEL,
) -> tuple[str, dict]:
    """
    Compress conversation using the OpenAI Responses API.
    Returns (summary_text, metrics_dict).
    """
    user_message = f"{COMPRESSION_PROMPT}\n\nCONVERSATION:\n{conversation_text}"

    body = {
        "model": model,
        "max_output_tokens": 2048,
        "input": [{"role": "user", "content": user_message}],
        "store": False,
    }
    payload = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        OPENAI_RESPONSES_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Compression API error {e.code}: {error_body[:500]}"
        ) from None

    # Extract summary text
    summary = data.get("output_text", "")
    if not summary:
        for item in data.get("output", []):
            if item.get("type") != "message":
                continue
            for block in item.get("content", []):
                if block.get("type") in ("output_text", "text"):
                    summary += block.get("text", "")

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    metrics = {
        "compression_input_tokens": input_tokens,
        "compression_output_tokens": output_tokens,
        "compression_tokens": input_tokens + output_tokens,
    }

    return summary, metrics


def _compress_via_anthropic(
    conversation_text: str,
    api_key: str,
    model: str = ANTHROPIC_COMPRESSION_MODEL,
) -> tuple[str, dict]:
    """
    Compress conversation using Anthropic Messages API.
    Returns (summary_text, metrics_dict).
    """
    user_message = f"{COMPRESSION_PROMPT}\n\nCONVERSATION:\n{conversation_text}"

    body = {
        "model": model,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": user_message}],
    }
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
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Compression API error {e.code}: {error_body[:500]}"
        ) from None

    summary = ""
    for block in data.get("content", []):
        if block.get("type") == "text":
            summary += block.get("text", "")

    usage = data.get("usage", {})
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    metrics = {
        "compression_input_tokens": input_tokens,
        "compression_output_tokens": output_tokens,
        "compression_tokens": input_tokens + output_tokens,
    }

    return summary, metrics


def _compress_via_api(
    conversation_text: str,
    api_key: str,
    provider: str = "openai",
    model: str | None = None,
) -> tuple[str, dict]:
    provider = _normalize_provider(provider)
    if provider == "anthropic":
        return _compress_via_anthropic(
            conversation_text,
            api_key,
            model or ANTHROPIC_COMPRESSION_MODEL,
        )
    return _compress_via_openai(
        conversation_text,
        api_key,
        model or OPENAI_COMPRESSION_MODEL,
    )


def _compress_via_ollama(
    conversation_text: str,
    model: str = "gemma4:26b",
) -> str | None:
    """
    Compress conversation using local Ollama.
    Returns summary text, or None if Ollama is unavailable.
    """
    prompt = f"{COMPRESSION_PROMPT}\n\nCONVERSATION:\n{conversation_text}"

    try:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def compress_history(
    messages: list[dict],
    original_task_prompt: str,
    mode: str = "api",
    api_key: str | None = None,
    ollama_model: str = "gemma4:26b",
    api_provider: str = "openai",
    api_model: str | None = None,
) -> tuple[list[dict], dict]:
    """
    Compress conversation history into a fresh single-message conversation.

    Args:
        messages: Full conversation messages array
        original_task_prompt: The original task prompt to preserve
        mode: "api" (configured provider) or "ollama" (local Gemma)
        api_key: Required for "api" mode
        ollama_model: Model name for "ollama" mode
        api_provider: "openai" or "anthropic" for API compression
        api_model: Optional provider model override for API compression

    Returns:
        (new_messages, metrics)
        new_messages: Fresh conversation with compressed history
        metrics: tokens_before, tokens_after, compression_tokens
    """
    # Estimate tokens_before from message size (rough: 1 token ≈ 4 chars)
    serialized = _serialize_messages(messages)
    tokens_before_est = len(serialized) // 4

    metrics = {
        "tokens_before": tokens_before_est,
        "tokens_after": 0,
        "compression_tokens": 0,
    }

    summary = None
    comp_metrics = {}

    if mode == "ollama":
        summary = _compress_via_ollama(serialized, model=ollama_model)
        if summary is None:
            print("  Warning: Ollama unavailable, falling back to API compression")
            mode = "api"

    if mode == "api":
        if not api_key:
            print("  Error: No API key for compression. Skipping.")
            return messages, metrics
        summary, comp_metrics = _compress_via_api(
            serialized,
            api_key,
            provider=api_provider,
            model=api_model,
        )
        metrics["compression_tokens"] = comp_metrics.get("compression_tokens", 0)

    if not summary:
        return messages, metrics

    # Build compressed conversation
    compressed_content = (
        f"{original_task_prompt}\n\n"
        f"PROGRESS SO FAR:\n{summary}\n\n"
        f"Continue with the task. Do not re-read files you have already "
        f"read unless they have changed."
    )

    new_messages = [{"role": "user", "content": compressed_content}]

    # Estimate tokens_after
    metrics["tokens_after"] = len(compressed_content) // 4

    return new_messages, metrics
