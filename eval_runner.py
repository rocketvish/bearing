"""
Bearing - Evaluation runner

Runs the same task queue under multiple context format conditions,
collects metrics, optionally judges output quality, and produces
a comparison report.

Usage:
    bearing eval <project_dir>

Conditions:
    1. prose          — full prose context (v1 baseline)
    2. structured     — JSON hooks (upgrade 1)
    3. embedding      — relevance scoring + template compression
    4. embedding+llm  — relevance scoring + LLM compression

Requires:
    - A tasks.json in the project directory
    - Git repo (for codebase reset between conditions)
    - Ollama with nomic-embed-text (for conditions 3-4)
    - Ollama with gemma4:26b (for condition 4)
"""

import os
import re
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime
from tasks_schema import TaskQueue, TaskStatus


CONDITIONS = ["prose", "structured", "embedding", "embedding+llm"]

# Task statuses that count as "done" for eval purposes
DONE_STATUSES = {TaskStatus.COMPLETED, TaskStatus.AWAITING_REVIEW}


def snapshot_state(project_dir: str, eval_dir: str):
    """Save tasks.json and git state before eval begins."""
    tasks_path = os.path.join(project_dir, "tasks.json")
    snapshot_path = os.path.join(eval_dir, "tasks_original.json")
    shutil.copy2(tasks_path, snapshot_path)


def restore_state(project_dir: str, eval_dir: str):
    """Restore tasks.json and reset git working tree between conditions."""
    snapshot_path = os.path.join(eval_dir, "tasks_original.json")
    tasks_path = os.path.join(project_dir, "tasks.json")
    shutil.copy2(snapshot_path, tasks_path)

    # Reset codebase to clean state
    subprocess.run(
        ["git", "checkout", "."], cwd=project_dir, capture_output=True, check=True
    )
    subprocess.run(
        ["git", "clean", "-fd"], cwd=project_dir, capture_output=True, check=True
    )

    # Remove debug/ and status.md from previous condition
    debug_dir = os.path.join(project_dir, "debug")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    status_path = os.path.join(project_dir, "status.md")
    if os.path.exists(status_path):
        os.remove(status_path)


def capture_source_files(project_dir: str, condition_dir: str):
    """
    Copy all created source files to condition directory for judge evaluation.
    Captures src/, any new .md files, and any new .js files in root.
    """
    src_dir = os.path.join(project_dir, "src")
    if os.path.exists(src_dir):
        dest = os.path.join(condition_dir, "src")
        if os.path.exists(dest):
            shutil.rmtree(dest)
        shutil.copytree(src_dir, dest)

    # Copy any created .md files (API.md, etc.) — skip status.md and originals
    skip_md = {"README.md", "CLAUDE.md", "status.md"}
    for f in os.listdir(project_dir):
        if f.endswith(".md") and f not in skip_md:
            shutil.copy2(
                os.path.join(project_dir, f),
                os.path.join(condition_dir, f),
            )


def extract_task_paths(task) -> list[str]:
    """
    Extract file paths relevant to a task from its prompt and relevant_files.
    Used to filter source files for per-task judge evaluation.
    """
    paths = set()

    # Paths explicitly mentioned in the task prompt (src/... patterns)
    for match in re.findall(r"src/[\w/.-]+\.(?:js|ts|json|md)", task.prompt):
        paths.add(match)

    # Paths from relevant_files
    for f in task.relevant_files:
        paths.add(f)

    return sorted(paths)


def read_source_files(condition_dir: str, filter_paths: list[str] = None) -> str:
    """
    Read source files from a condition directory into a single string.
    Used to give the judge the actual code that was produced.

    If filter_paths is provided, prioritize those files first, then include
    remaining files. This ensures the judge sees task-relevant files before
    any truncation occurs.
    """
    priority_parts = []
    other_parts = []

    src_dir = os.path.join(condition_dir, "src")
    if os.path.exists(src_dir):
        for root, _dirs, files in os.walk(src_dir):
            for fname in sorted(files):
                filepath = os.path.join(root, fname)
                relpath = os.path.relpath(filepath, condition_dir)
                # Normalize to forward slashes for matching
                relpath_normalized = relpath.replace("\\", "/")
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                    entry = f"=== {relpath_normalized} ===\n{content}"
                except Exception:
                    entry = f"=== {relpath_normalized} ===\n[could not read]"

                if filter_paths and relpath_normalized in filter_paths:
                    priority_parts.append(entry)
                else:
                    other_parts.append(entry)

    # Read any .md files in condition root (API.md, etc.)
    for fname in sorted(os.listdir(condition_dir)):
        fpath = os.path.join(condition_dir, fname)
        if (
            fname.endswith(".md")
            and fname not in ("status.md", "results.md")
            and os.path.isfile(fpath)
        ):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                other_parts.append(f"=== {fname} ===\n{content}")
            except Exception:
                pass

    # Priority files first, then the rest
    all_parts = priority_parts + other_parts
    if not all_parts:
        return "No source files found."

    return "\n\n".join(all_parts)


def run_condition(project_dir: str, condition: str, eval_dir: str) -> dict:
    """
    Run bearing with a specific format condition.
    Returns metrics dict for this condition.
    """
    condition_dir = os.path.join(eval_dir, condition.replace("+", "_"))
    os.makedirs(condition_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"  Condition: {condition}")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "bearing",
        "run",
        project_dir,
        "--format",
        condition,
    ]
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    wall_time = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    tasks_path = os.path.join(project_dir, "tasks.json")
    queue = TaskQueue.load(tasks_path)

    # Copy artifacts to condition directory
    shutil.copy2(tasks_path, os.path.join(condition_dir, "tasks_final.json"))

    status_path = os.path.join(project_dir, "status.md")
    if os.path.exists(status_path):
        shutil.copy2(status_path, os.path.join(condition_dir, "status.md"))

    debug_dir = os.path.join(project_dir, "debug")
    if os.path.exists(debug_dir):
        condition_debug = os.path.join(condition_dir, "debug")
        if os.path.exists(condition_debug):
            shutil.rmtree(condition_debug)
        shutil.copytree(debug_dir, condition_debug)

    # Capture actual source files BEFORE git reset
    capture_source_files(project_dir, condition_dir)

    # Extract metrics per task
    task_metrics = []
    for task in queue.tasks:
        r = task.result
        task_metrics.append(
            {
                "id": task.id,
                "name": task.name,
                "status": r.status.value,
                "cost_usd": r.cost_usd,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cache_read_tokens": r.cache_read_tokens,
                "turns_used": r.turns_used,
                "context_chars_original": r.context_chars_original,
                "context_chars_compressed": r.context_chars_compressed,
                "chunks_kept": r.chunks_kept,
                "chunks_compressed": r.chunks_compressed,
                "chunks_dropped": r.chunks_dropped,
                "scoring_latency_ms": r.scoring_latency_ms,
                "compression_latency_ms": r.compression_latency_ms,
            }
        )

    completed = sum(1 for t in queue.tasks if t.result.status in DONE_STATUSES)
    failed = sum(1 for t in queue.tasks if t.result.status == TaskStatus.FAILED)

    return {
        "condition": condition,
        "wall_time_s": round(wall_time, 1),
        "total_cost": round(queue.total_cost, 4),
        "total_input_tokens": sum(t.result.input_tokens for t in queue.tasks),
        "total_output_tokens": sum(t.result.output_tokens for t in queue.tasks),
        "completed": completed,
        "failed": failed,
        "total_tasks": len(queue.tasks),
        "tasks": task_metrics,
    }


def judge_task(task_prompt: str, source_files: str, temp_dir: str) -> dict:
    """
    Use claude -p to judge task output quality by examining actual source files.
    Returns scores dict.
    """
    if len(source_files) > 15000:
        source_files = source_files[:15000] + "\n\n[... truncated for length ...]"

    judge_prompt = (
        "You are evaluating whether an AI coding agent completed a task correctly. "
        "The agent was given a task prompt and produced source files.\n\n"
        "Rate the ACTUAL SOURCE FILES on three dimensions (1-5 scale):\n"
        "  1 = not attempted\n"
        "  2 = partially attempted but major gaps\n"
        "  3 = mostly complete with some issues\n"
        "  4 = complete with minor issues\n"
        "  5 = fully complete and correct\n\n"
        "Dimensions:\n"
        "1. Completeness: Do the source files address everything in the task prompt?\n"
        "2. Correctness: Does the code look functional and error-free?\n"
        "3. Adherence: Did the code follow the specific instructions (file names, patterns, etc.)?\n\n"
        f"TASK PROMPT:\n{task_prompt[:1500]}\n\n"
        f"ACTUAL SOURCE FILES PRODUCED:\n{source_files}\n\n"
        "Respond ONLY with valid JSON on a single line, no other text:\n"
        '{"completeness": N, "correctness": N, "adherence": N, "notes": "brief explanation"}'
    )

    try:
        result = subprocess.run(
            [
                "claude",
                "-p",
                judge_prompt,
                "--model",
                "sonnet",
                "--output-format",
                "json",
            ],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout:
            parsed = json.loads(result.stdout)
            result_text = parsed.get("result", "")
            if isinstance(result_text, str):
                clean = result_text.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                json_start = clean.find("{")
                if json_start >= 0:
                    clean = clean[json_start:]
                    json_end = clean.rfind("}") + 1
                    if json_end > 0:
                        clean = clean[:json_end]
                scores = json.loads(clean)
                for key in ["completeness", "correctness", "adherence"]:
                    if key in scores and isinstance(scores[key], (int, float)):
                        scores[key] = max(1, min(5, int(scores[key])))
                return scores
    except Exception as e:
        return {
            "completeness": 0,
            "correctness": 0,
            "adherence": 0,
            "notes": f"judge parse failed: {str(e)[:100]}",
        }

    return {
        "completeness": 0,
        "correctness": 0,
        "adherence": 0,
        "notes": "judge failed",
    }


def run_judges(eval_dir: str, all_results: list[dict]) -> dict:
    """
    Run quality judgments on all conditions' outputs.
    Reads actual source files from each condition directory.
    """
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="bearing_judge_")

    print(f"\n{'=' * 60}")
    print("  Running quality judgments (Claude Sonnet)")
    print("  Judge evaluates actual source files, not session summaries")
    print(f"{'=' * 60}\n")

    judgments = {}
    for cond_result in all_results:
        condition = cond_result["condition"]
        condition_dir = os.path.join(eval_dir, condition.replace("+", "_"))
        tasks_path = os.path.join(condition_dir, "tasks_final.json")

        if not os.path.exists(tasks_path):
            continue

        queue = TaskQueue.load(tasks_path)
        cond_scores = []

        for task in queue.tasks:
            if task.result.status not in DONE_STATUSES:
                cond_scores.append(
                    {
                        "id": task.id,
                        "status": task.result.status.value,
                        "completeness": 0,
                        "correctness": 0,
                        "adherence": 0,
                        "notes": f"task status: {task.result.status.value}",
                    }
                )
                continue

            # Filter source files to prioritize task-relevant paths
            task_paths = extract_task_paths(task)
            source_files = read_source_files(condition_dir, filter_paths=task_paths)

            print(f"  Judging: {condition} / {task.id}...")
            scores = judge_task(task.prompt, source_files, temp_dir)
            scores["id"] = task.id
            scores["status"] = task.result.status.value
            cond_scores.append(scores)

        judgments[condition] = cond_scores

    shutil.rmtree(temp_dir, ignore_errors=True)
    return judgments


def write_report(eval_dir: str, all_results: list[dict], judgments: dict = None):
    """Write results.md comparison table and results.json raw data."""

    raw_data = {"timestamp": datetime.now().isoformat(), "results": all_results}
    if judgments:
        raw_data["judgments"] = judgments

    with open(os.path.join(eval_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    lines = [
        "# Bearing Evaluation Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
    ]

    header = "| Metric |"
    separator = "|--------|"
    for r in all_results:
        header += f" {r['condition']} |"
        separator += "--------|"
    lines.extend([header, separator])

    def row(label, key, fmt=str):
        line = f"| {label} |"
        for r in all_results:
            line += f" {fmt(r[key])} |"
        return line

    lines.append(row("Completed", "completed"))
    lines.append(row("Failed", "failed"))
    lines.append(row("Cost USD", "total_cost", lambda x: f"${x:.2f}"))
    lines.append(row("Input tokens", "total_input_tokens", lambda x: f"{x:,}"))
    lines.append(row("Output tokens", "total_output_tokens", lambda x: f"{x:,}"))
    lines.append(row("Wall time", "wall_time_s", lambda x: f"{x:.0f}s"))

    for r in all_results:
        total_original = sum(t["context_chars_original"] for t in r["tasks"])
        total_compressed = sum(t["context_chars_compressed"] for t in r["tasks"])
        r["total_context_original"] = total_original
        r["total_context_compressed"] = total_compressed
        r["total_chunks_dropped"] = sum(t["chunks_dropped"] for t in r["tasks"])
        r["total_chunks_kept"] = sum(t["chunks_kept"] for t in r["tasks"])

    lines.append(
        row("Context chars (original)", "total_context_original", lambda x: f"{x:,}")
    )
    lines.append(
        row("Context chars (final)", "total_context_compressed", lambda x: f"{x:,}")
    )
    lines.append(row("Chunks kept", "total_chunks_kept"))
    lines.append(row("Chunks dropped", "total_chunks_dropped"))

    lines.extend(["", ""])

    if judgments:
        lines.extend(["## Quality Scores (1-5, higher is better)", ""])

        header2 = "| Metric |"
        sep2 = "|--------|"
        for cond in [r["condition"] for r in all_results]:
            if cond in judgments:
                header2 += f" {cond} |"
                sep2 += "--------|"
        lines.extend([header2, sep2])

        for dim in ["completeness", "correctness", "adherence"]:
            line = f"| {dim.title()} |"
            for cond in [r["condition"] for r in all_results]:
                if cond in judgments:
                    scores = [t[dim] for t in judgments[cond] if t.get(dim, 0) > 0]
                    avg = sum(scores) / len(scores) if scores else 0
                    line += f" {avg:.1f} |"
            lines.append(line)

        lines.extend(["", ""])

    lines.extend(["## Per-Task Detail", ""])
    for task_idx in range(len(all_results[0]["tasks"])):
        task_id = all_results[0]["tasks"][task_idx]["id"]
        task_name = all_results[0]["tasks"][task_idx]["name"]
        lines.append(f"### {task_id}: {task_name}")
        lines.append("")

        header3 = "| Metric |"
        sep3 = "|--------|"
        for r in all_results:
            header3 += f" {r['condition']} |"
            sep3 += "--------|"
        lines.extend([header3, sep3])

        def task_row(label, key, fmt=str):
            line = f"| {label} |"
            for r in all_results:
                t = r["tasks"][task_idx]
                line += f" {fmt(t[key])} |"
            return line

        lines.append(task_row("Status", "status"))
        lines.append(task_row("Cost", "cost_usd", lambda x: f"${x:.2f}"))
        lines.append(task_row("Input tokens", "input_tokens", lambda x: f"{x:,}"))
        lines.append(task_row("Turns", "turns_used"))
        lines.append(task_row("Chunks kept", "chunks_kept"))
        lines.append(task_row("Chunks dropped", "chunks_dropped"))
        lines.append(task_row("Scoring ms", "scoring_latency_ms"))

        if judgments:
            for dim in ["completeness", "correctness", "adherence"]:
                line = f"| Quality: {dim} |"
                for r in all_results:
                    cond = r["condition"]
                    if cond in judgments and task_idx < len(judgments[cond]):
                        score = judgments[cond][task_idx].get(dim, 0)
                        line += f" {score} |"
                    else:
                        line += " - |"
                lines.append(line)

        lines.extend(["", ""])

    lines.extend(
        [
            "## Compaction Estimate",
            "",
            "Assuming compaction triggers at ~100K accumulated input tokens in a shared session:",
            "",
        ]
    )
    for r in all_results:
        cum_tokens = r["total_input_tokens"]
        est_compactions = cum_tokens // 100000
        lines.append(
            f"- **{r['condition']}**: {cum_tokens:,} total input tokens -> ~{est_compactions} compactions in a shared session (Bearing: 0)"
        )
    lines.append("")

    # Context efficiency analysis
    lines.extend(
        [
            "## Notes on Interpretation",
            "",
            "**Context size vs total tokens:** Injected context is ~50-300 tokens per task.",
            "Total per-task consumption is 100K-300K tokens (Claude Code reading files, writing",
            "code, running tests). Context compression affects <0.1% of total tokens per task.",
            "The value of compression is in compaction avoidance across long sessions, not",
            "per-task token savings.",
            "",
            "**Execution non-determinism:** Per-task cost and turns vary significantly across",
            "conditions due to inherent randomness in agentic execution (file reads, test",
            "retries, tool calls). Single-run results should be interpreted cautiously.",
            "Multi-trial runs (--trials N) are recommended for statistical confidence.",
            "",
        ]
    )

    with open(os.path.join(eval_dir, "results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_eval(project_dir: str, skip_judge: bool = False):
    """
    Run the full evaluation: 4 conditions, collect metrics, judge quality, write report.
    """
    project_dir = os.path.abspath(project_dir)
    eval_dir = os.path.join(project_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    tasks_path = os.path.join(project_dir, "tasks.json")
    if not os.path.exists(tasks_path):
        print(f"Error: tasks.json not found in {project_dir}")
        sys.exit(1)

    if not os.path.exists(os.path.join(project_dir, ".git")):
        print("Error: Project must be a git repo for eval (needed for codebase reset)")
        print("Run: git init && git add . && git commit -m 'benchmark baseline'")
        sys.exit(1)

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        print("The eval runner resets the codebase between conditions.")
        sys.exit(1)

    print("Bearing Evaluation")
    print(f"Project: {project_dir}")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print()

    snapshot_state(project_dir, eval_dir)

    from relevance import warmup

    print("Checking Ollama models...")
    queue = TaskQueue.load(tasks_path)
    warmup_status = warmup(
        embedding_model=queue.embedding_model,
        compression_model=queue.compression_model,
        use_llm=True,
    )
    print()

    conditions_to_run = list(CONDITIONS)
    if not warmup_status["embedding"]:
        print("Warning: Skipping embedding conditions (Ollama not available)")
        conditions_to_run = [c for c in conditions_to_run if "embedding" not in c]
    if not warmup_status["compression"]:
        print(
            "Warning: Skipping embedding+llm condition (compression model not available)"
        )
        conditions_to_run = [c for c in conditions_to_run if c != "embedding+llm"]

    all_results = []
    for condition in conditions_to_run:
        restore_state(project_dir, eval_dir)
        cond_result = run_condition(project_dir, condition, eval_dir)
        all_results.append(cond_result)

    restore_state(project_dir, eval_dir)

    judgments = None
    if not skip_judge:
        judgments = run_judges(eval_dir, all_results)

    write_report(eval_dir, all_results, judgments)

    print(f"\n{'=' * 60}")
    print("  Evaluation complete")
    print(f"{'=' * 60}")
    print(f"\nResults: {os.path.join(eval_dir, 'results.md')}")
    print(f"Raw data: {os.path.join(eval_dir, 'results.json')}")
