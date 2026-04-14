"""
Bearing - Embedding-based selective history retrieval

An alternative to lossy summarization for mid-conversation compression.
Instead of summarizing and discarding the full history, this module embeds
each conversation turn and retrieves only the top-K most relevant turns
when context needs to be reduced.

Requires Ollama running locally with an embedding model:
    ollama pull nomic-embed-text

Falls back gracefully (returns full history) if Ollama is unavailable.
"""

import json
import math
import urllib.error
import urllib.request

OLLAMA_BASE = "http://localhost:11434"


def _ollama_embed(
    texts: list[str], model: str = "nomic-embed-text"
) -> list[list[float]] | None:
    """
    Get embedding vectors from Ollama.
    Returns None if Ollama is unavailable or model not pulled.
    """
    try:
        payload = json.dumps({"model": model, "input": texts}).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data.get("embeddings")
    except urllib.error.URLError:
        return None
    except Exception:
        return None


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _serialize_turn(assistant_message: dict, tool_results: list[dict]) -> str:
    """
    Serialize a single turn (assistant response + tool results) into a
    text string suitable for embedding.
    """
    parts = []

    # Extract text and tool calls from assistant message
    content = assistant_message.get("content", [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    parts.append(block.get("text", ""))
                elif btype == "tool_use":
                    name = block.get("name", "?")
                    inp = block.get("input", {})
                    # Include tool name and key input for embedding
                    if name == "read_file":
                        parts.append(f"read_file({inp.get('path', '')})")
                    elif name == "write_file":
                        path = inp.get("path", "")
                        content_preview = inp.get("content", "")[:200]
                        parts.append(f"write_file({path}): {content_preview}")
                    elif name == "run_command":
                        parts.append(f"run_command: {inp.get('command', '')}")
                    else:
                        parts.append(f"{name}({json.dumps(inp)[:200]})")
    elif isinstance(content, str):
        parts.append(content)

    # Include tool results (truncated)
    for tr in tool_results:
        result_content = tr.get("content", "")
        if isinstance(result_content, str) and result_content:
            parts.append(result_content[:500])

    return "\n".join(parts)


class TurnRetriever:
    """
    Stores conversation turns with their embeddings and retrieves
    the most relevant ones based on cosine similarity to a query.
    """

    def __init__(self, ollama_model: str = "nomic-embed-text"):
        self.ollama_model = ollama_model
        self.turns: list[dict] = []
        # Each entry: {
        #   "turn_number": int,
        #   "assistant_message": dict,
        #   "tool_results": list[dict],
        #   "text": str,        # serialized for embedding
        #   "embedding": list[float] | None,
        # }

    def store_turn(
        self,
        turn_number: int,
        assistant_message: dict,
        tool_results: list[dict],
    ):
        """
        Store a completed turn with its embedding.
        Embedding is computed immediately; if Ollama is unavailable,
        the embedding is stored as None (turn is still kept).
        """
        text = _serialize_turn(assistant_message, tool_results)

        embedding = None
        vectors = _ollama_embed([text], self.ollama_model)
        if vectors and len(vectors) > 0:
            embedding = vectors[0]

        self.turns.append(
            {
                "turn_number": turn_number,
                "assistant_message": assistant_message,
                "tool_results": tool_results,
                "text": text,
                "embedding": embedding,
            }
        )

    def retrieve(self, query: str, top_k: int = 5) -> tuple[list[dict], dict]:
        """
        Retrieve the top-K most relevant stored turns for the given query.

        Returns:
            (selected_turns, metrics)
            selected_turns: list of turn dicts, sorted by turn_number (chronological)
            metrics: {
                "total_stored": int,
                "retrieved": int,
                "scores": list of (turn_number, score) for ALL turns,
                "ollama_available": bool,
            }
        """
        metrics = {
            "total_stored": len(self.turns),
            "retrieved": 0,
            "scores": [],
            "ollama_available": True,
        }

        if not self.turns:
            return [], metrics

        # Check if any turns have embeddings
        has_embeddings = any(t["embedding"] is not None for t in self.turns)
        if not has_embeddings:
            # No embeddings available — return all turns (fallback)
            metrics["ollama_available"] = False
            metrics["retrieved"] = len(self.turns)
            return list(self.turns), metrics

        # Embed the query
        query_vectors = _ollama_embed([query], self.ollama_model)
        if query_vectors is None:
            # Ollama went away mid-session — return all turns
            metrics["ollama_available"] = False
            metrics["retrieved"] = len(self.turns)
            return list(self.turns), metrics

        query_embedding = query_vectors[0]

        # Score all turns
        scored = []
        for turn in self.turns:
            if turn["embedding"] is not None:
                score = _cosine_similarity(turn["embedding"], query_embedding)
            else:
                score = 0.0  # No embedding — low priority
            scored.append((turn, score))
            metrics["scores"].append((turn["turn_number"], round(score, 4)))

        # Sort by score descending, take top-K
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [t for t, _ in scored[:top_k]]

        # Re-sort selected by turn_number for chronological order
        selected.sort(key=lambda t: t["turn_number"])

        metrics["retrieved"] = len(selected)
        return selected, metrics

    def build_retrieved_history(
        self, original_task_prompt: str, top_k: int = 5
    ) -> tuple[list[dict], dict]:
        """
        Build a new message list from retrieved turns, suitable for
        replacing the full conversation history.

        The new history starts with the original task prompt, then
        includes assistant/tool messages from the top-K retrieved turns.

        Returns:
            (new_messages, metrics) — same metrics as retrieve()
        """
        selected, metrics = self.retrieve(original_task_prompt, top_k=top_k)

        if not selected:
            # Nothing stored yet — return just the task prompt
            return [{"role": "user", "content": original_task_prompt}], metrics

        # Build message sequence: task prompt + selected turns
        new_messages = [{"role": "user", "content": original_task_prompt}]

        for turn in selected:
            # Add assistant message
            new_messages.append(turn["assistant_message"])
            # Add tool results as user message
            if turn["tool_results"]:
                new_messages.append({"role": "user", "content": turn["tool_results"]})

        return new_messages, metrics
