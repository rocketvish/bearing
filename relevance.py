"""
Bearing - Relevance scoring and context compression

Scores context chunks against the current task prompt using local
embedding models via Ollama. High-relevance chunks are kept verbatim,
mid-relevance chunks are compressed (via template or LLM), and
low-relevance chunks are dropped entirely.

Requires Ollama running locally with embedding and (optionally)
compression models pulled:
    ollama pull nomic-embed-text
    ollama pull gemma4:26b

Falls back gracefully if Ollama is unavailable.
"""

import json
import math
import time
import urllib.request
import urllib.error
from typing import Optional

OLLAMA_BASE = "http://localhost:11434"


# --- Ollama API ---


def ollama_embed(
    texts: list[str], model: str = "nomic-embed-text"
) -> Optional[list[list[float]]]:
    """
    Get embedding vectors for a list of texts from Ollama.
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


def ollama_generate(prompt: str, model: str = "gemma4:26b") -> Optional[str]:
    """
    Generate text from Ollama. Returns None if unavailable.
    """
    try:
        payload = json.dumps(
            {
                "model": model,
                "prompt": prompt,
                "stream": False,
            }
        ).encode()
        req = urllib.request.Request(
            f"{OLLAMA_BASE}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read())
            return data.get("response", "")
    except urllib.error.URLError:
        return None
    except Exception:
        return None


# --- Embedding Cache ---

_embedding_cache: dict[str, list[float]] = {}


def clear_cache():
    """Clear the embedding cache. Call at the start of each bearing run."""
    _embedding_cache.clear()


def get_embeddings(
    texts: list[str], model: str = "nomic-embed-text"
) -> Optional[list[list[float]]]:
    """
    Get embeddings with caching. Cached texts are not re-sent to Ollama.
    Returns None if Ollama is unavailable.
    """
    uncached_texts = []
    uncached_indices = []

    for i, text in enumerate(texts):
        if text not in _embedding_cache:
            uncached_texts.append(text)
            uncached_indices.append(i)

    if uncached_texts:
        vectors = ollama_embed(uncached_texts, model)
        if vectors is None:
            return None
        for text, vector in zip(uncached_texts, vectors):
            _embedding_cache[text] = vector

    return [_embedding_cache[text] for text in texts]


# --- Similarity ---


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# --- Compression ---


def template_compress(chunk_text: str) -> str:
    """
    Template-based compression for mid-tier chunks.
    Extracts structured fields without LLM call.
    """
    from executor import parse_context_entries

    entries = parse_context_entries(chunk_text)
    if not entries:
        return chunk_text[:100]

    e = entries[0]
    parts = [e["id"]]
    if e["did"]:
        parts.append(e["did"][:100])
    if e["files"]:
        parts.append("files:" + ",".join(e["files"]))
    return " | ".join(parts)


def llm_compress(
    chunks: list[str], task_prompt: str, model: str = "gemma4:26b"
) -> list[str]:
    """
    LLM-based compression for mid-tier chunks. Batches all chunks
    into one Gemma call for efficiency.
    Returns compressed versions, or falls back to template if Gemma unavailable.
    """
    if not chunks:
        return []

    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(chunks))
    prompt = (
        f"Compress each numbered summary to one sentence, "
        f"keeping only details relevant to this task: {task_prompt[:200]}\n\n"
        f"{numbered}\n\n"
        f"Respond with one compressed sentence per line, numbered to match."
    )

    result = ollama_generate(prompt, model)
    if result is None:
        return [template_compress(c) for c in chunks]

    # Parse numbered responses
    lines = [ln.strip() for ln in result.strip().split("\n") if ln.strip()]
    compressed = []
    for i, chunk in enumerate(chunks):
        # Try to find matching numbered line
        found = False
        for line in lines:
            if line.startswith(f"{i + 1}.") or line.startswith(f"{i + 1})"):
                # Strip the number prefix
                text = line.split(".", 1)[-1].strip() if "." in line else line
                if text.startswith(")"):
                    text = line.split(")", 1)[-1].strip()
                compressed.append(text)
                found = True
                break
        if not found:
            compressed.append(template_compress(chunk))

    return compressed


# --- Main Scoring Pipeline ---


def score_and_compress(
    context_chunks: list[str],
    task_prompt: str,
    threshold_keep: float = 0.6,
    threshold_drop: float = 0.35,
    embedding_model: str = "nomic-embed-text",
    compression_model: str = "gemma4:26b",
    use_llm_compression: bool = False,
) -> tuple[list[str], dict]:
    """
    Score context chunks against task prompt and compress based on relevance.

    Returns:
        (compressed_chunks, metrics_dict)

    Metrics include:
        chunks_kept, chunks_compressed, chunks_dropped,
        scoring_latency_ms, compression_latency_ms,
        scores (list of similarity scores per chunk)
    """
    metrics = {
        "chunks_kept": 0,
        "chunks_compressed": 0,
        "chunks_dropped": 0,
        "scoring_latency_ms": 0,
        "compression_latency_ms": 0,
        "scores": [],
        "ollama_available": True,
    }

    if not context_chunks:
        return [], metrics

    # Score all chunks
    t0 = time.time()
    all_texts = context_chunks + [task_prompt]
    embeddings = get_embeddings(all_texts, embedding_model)

    if embeddings is None:
        metrics["ollama_available"] = False
        return context_chunks, metrics  # fallback: keep everything

    task_embedding = embeddings[-1]
    chunk_embeddings = embeddings[:-1]

    scores = [cosine_similarity(ce, task_embedding) for ce in chunk_embeddings]
    metrics["scoring_latency_ms"] = int((time.time() - t0) * 1000)
    metrics["scores"] = [round(s, 4) for s in scores]

    # Categorize chunks
    kept = []  # verbatim
    mid_chunks = []  # to compress
    mid_indices = []  # track original positions

    for i, (chunk, score) in enumerate(zip(context_chunks, scores)):
        if score >= threshold_keep:
            kept.append(chunk)
            metrics["chunks_kept"] += 1
        elif score >= threshold_drop:
            mid_chunks.append(chunk)
            mid_indices.append(i)
            metrics["chunks_compressed"] += 1
        else:
            metrics["chunks_dropped"] += 1

    # Compress mid-tier chunks
    t1 = time.time()
    if mid_chunks:
        if use_llm_compression:
            compressed_mid = llm_compress(mid_chunks, task_prompt, compression_model)
        else:
            compressed_mid = [template_compress(c) for c in mid_chunks]
    else:
        compressed_mid = []
    metrics["compression_latency_ms"] = int((time.time() - t1) * 1000)

    # Reassemble in original order
    result = []
    mid_idx = 0
    for i, (chunk, score) in enumerate(zip(context_chunks, scores)):
        if score >= threshold_keep:
            result.append(chunk)
        elif score >= threshold_drop:
            result.append(compressed_mid[mid_idx])
            mid_idx += 1
        # dropped chunks: skip

    return result, metrics


def warmup(
    embedding_model: str = "nomic-embed-text",
    compression_model: str = "gemma4:26b",
    use_llm: bool = False,
) -> dict:
    """
    Warm up Ollama models with dummy calls. Returns status dict.
    """
    status = {"embedding": False, "compression": False}

    result = ollama_embed(["warmup"], embedding_model)
    if result is not None:
        status["embedding"] = True
        print(f"  Embedding model ({embedding_model}): ready")
    else:
        print(f"  Embedding model ({embedding_model}): NOT AVAILABLE")
        print(f"  Run: ollama pull {embedding_model}")

    if use_llm:
        result = ollama_generate("Say OK.", compression_model)
        if result is not None:
            status["compression"] = True
            print(f"  Compression model ({compression_model}): ready")
        else:
            print(f"  Compression model ({compression_model}): NOT AVAILABLE")
            print(f"  Run: ollama pull {compression_model}")

    return status
