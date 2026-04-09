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
import sys
import json
import shutil
import subprocess
import time
from datetime import datetime
from tasks_schema import TaskQueue, TaskStatus


CONDITIONS = ["prose", "structured", "embedding", "embedding+llm"]


def snapshot_state(project_dir: str, eval_dir: str):
    """Save tasks.json and git state before eval begins."""
    tasks_path = os.path.join(project_dir, "tasks.json")
    snapshot_path = os.path.join(eval_dir, "tasks_original.json")
    shutil.copy2(tasks_path, snapshot_path)


def restore_state(project_dir: str, eval_dir: str):
    """Restore tasks.json and reset git working tree between conditions."""
    # Restore tasks.json
    snapshot_path = os.path.join(eval_dir, "tasks_original.json")
    tasks_path = os.path.join(project_dir, "tasks.json")
    shutil.copy2(snapshot_path, tasks_path)

    # Reset codebase to clean state
    subprocess.run(["git", "checkout", "."], cwd=project_dir,
                   capture_output=True, check=True)
    subprocess.run(["git", "clean", "-fd"], cwd=project_dir,
                   capture_output=True, check=True)

    # Remove debug/ and status.md from previous condition
    debug_dir = os.path.join(project_dir, "debug")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
    status_path = os.path.join(project_dir, "status.md")
    if os.path.exists(status_path):
        os.remove(status_path)


def run_condition(project_dir: str, condition: str, eval_dir: str) -> dict:
    """
    Run bearing with a specific format condition.
    Returns metrics dict for this condition.
    """
    condition_dir = os.path.join(eval_dir, condition.replace("+", "_"))
    os.makedirs(condition_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Condition: {condition}")
    print(f"{'='*60}\n")

    # Run bearing
    t0 = time.time()
    cmd = [
        sys.executable, "-m", "bearing", "run", project_dir,
        "--format", condition,
    ]
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    wall_time = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Collect results
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

    # Extract metrics per task
    task_metrics = []
    for task in queue.tasks:
        r = task.result
        task_metrics.append({
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
        })

    completed = sum(1 for t in queue.tasks if t.result.status == TaskStatus.COMPLETED)
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


def judge_task(task_prompt: str, task_output: str, temp_dir: str) -> dict:
    """
    Use claude -p to judge task output quality.
    Returns scores dict or None if judging fails.
    """
    judge_prompt = (
        "You are evaluating AI-generated code task output quality. "
        "Rate on three dimensions (1-5 scale):\n\n"
        "1. Completeness: Did the output address everything in the task prompt?\n"
        "2. Correctness: Is the code likely to work without errors?\n"
        "3. Adherence: Did the output follow specific instructions?\n\n"
        f"TASK PROMPT:\n{task_prompt[:1000]}\n\n"
        f"TASK OUTPUT:\n{task_output[:2000]}\n\n"
        "Respond ONLY with JSON: "
        '{"completeness": N, "correctness": N, "adherence": N, "notes": "brief explanation"}'
    )

    try:
        result = subprocess.run(
            ["claude", "-p", judge_prompt, "--model", "sonnet", "--output-format", "json"],
            cwd=temp_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.stdout:
            parsed = json.loads(result.stdout)
            # Extract the result text from claude -p output
            result_text = parsed.get("result", "")
            if isinstance(result_text, str):
                # Try to parse JSON from the result
                # Strip markdown fences if present
                clean = result_text.strip()
                if clean.startswith("```"):
                    clean = clean.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
                scores = json.loads(clean)
                return scores
    except Exception:
        pass

    return {"completeness": 0, "correctness": 0, "adherence": 0, "notes": "judge failed"}


def run_judges(eval_dir: str, all_results: list[dict]) -> dict:
    """
    Run quality judgments on all conditions' outputs.
    Returns dict mapping condition -> list of score dicts per task.
    """
    import tempfile
    temp_dir = tempfile.mkdtemp(prefix="bearing_judge_")

    print(f"\n{'='*60}")
    print(f"  Running quality judgments (Claude Sonnet)")
    print(f"{'='*60}\n")

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
            if task.result.status != TaskStatus.COMPLETED:
                cond_scores.append({
                    "id": task.id,
                    "status": task.result.status.value,
                    "completeness": 0,
                    "correctness": 0,
                    "adherence": 0,
                    "notes": "task did not complete",
                })
                continue

            print(f"  Judging: {condition} / {task.id}...")
            scores = judge_task(task.prompt, task.result.summary, temp_dir)
            scores["id"] = task.id
            scores["status"] = "completed"
            cond_scores.append(scores)

        judgments[condition] = cond_scores

    # Clean up temp dir
    shutil.rmtree(temp_dir, ignore_errors=True)
    return judgments


def write_report(eval_dir: str, all_results: list[dict],
                 judgments: dict = None):
    """Write results.md comparison table and results.json raw data."""

    # Write raw JSON
    raw_data = {"timestamp": datetime.now().isoformat(), "results": all_results}
    if judgments:
        raw_data["judgments"] = judgments

    with open(os.path.join(eval_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    # Write human-readable report
    lines = [
        "# Bearing Evaluation Report",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
    ]

    # Header row
    header = "| Metric |"
    separator = "|--------|"
    for r in all_results:
        header += f" {r['condition']} |"
        separator += "--------|"
    lines.extend([header, separator])

    # Metrics rows
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

    # Context metrics (from per-task data)
    for r in all_results:
        total_original = sum(t["context_chars_original"] for t in r["tasks"])
        total_compressed = sum(t["context_chars_compressed"] for t in r["tasks"])
        r["total_context_original"] = total_original
        r["total_context_compressed"] = total_compressed
        r["total_chunks_dropped"] = sum(t["chunks_dropped"] for t in r["tasks"])

    lines.append(row("Context chars (original)", "total_context_original", lambda x: f"{x:,}"))
    lines.append(row("Context chars (final)", "total_context_compressed", lambda x: f"{x:,}"))
    lines.append(row("Chunks dropped", "total_chunks_dropped"))

    lines.extend(["", ""])

    # Quality scores if available
    if judgments:
        lines.extend(["## Quality Scores (1-5)", ""])

        header2 = "| Metric |"
        sep2 = "|--------|"
        for cond in CONDITIONS:
            if cond in judgments:
                header2 += f" {cond} |"
                sep2 += "--------|"
        lines.extend([header2, sep2])

        for dim in ["completeness", "correctness", "adherence"]:
            line = f"| {dim.title()} |"
            for cond in CONDITIONS:
                if cond in judgments:
                    scores = [t[dim] for t in judgments[cond] if t.get(dim, 0) > 0]
                    avg = sum(scores) / len(scores) if scores else 0
                    line += f" {avg:.1f} |"
            lines.append(line)

        lines.extend(["", ""])

    # Per-task detail
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
        lines.append(task_row("Chunks kept", "chunks_kept"))
        lines.append(task_row("Chunks dropped", "chunks_dropped"))
        lines.append(task_row("Scoring ms", "scoring_latency_ms"))

        lines.extend(["", ""])

    # Compaction estimate
    lines.extend([
        "## Compaction Estimate",
        "",
        "Assuming compaction triggers at ~100K accumulated input tokens in a shared session:",
        "",
    ])
    for r in all_results:
        cum_tokens = r["total_input_tokens"]
        est_compactions = cum_tokens // 100000
        lines.append(f"- **{r['condition']}**: {cum_tokens:,} total input tokens → ~{est_compactions} compactions in a shared session (Bearing: 0)")
    lines.append("")

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

    # Verify git repo
    if not os.path.exists(os.path.join(project_dir, ".git")):
        print("Error: Project must be a git repo for eval (needed for codebase reset)")
        print("Run: git init && git add . && git commit -m 'benchmark baseline'")
        sys.exit(1)

    # Check for uncommitted changes
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir, capture_output=True, text=True,
    )
    if result.stdout.strip():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        print("The eval runner resets the codebase between conditions.")
        sys.exit(1)

    print("Bearing Evaluation")
    print(f"Project: {project_dir}")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print()

    # Snapshot
    snapshot_state(project_dir, eval_dir)

    # Check Ollama for embedding conditions
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
        print("Warning: Skipping embedding+llm condition (compression model not available)")
        conditions_to_run = [c for c in conditions_to_run if c != "embedding+llm"]

    # Run each condition
    all_results = []
    for condition in conditions_to_run:
        restore_state(project_dir, eval_dir)
        cond_result = run_condition(project_dir, condition, eval_dir)
        all_results.append(cond_result)

    # Restore to clean state after all conditions
    restore_state(project_dir, eval_dir)

    # Run quality judges
    judgments = None
    if not skip_judge:
        judgments = run_judges(eval_dir, all_results)

    # Write report
    write_report(eval_dir, all_results, judgments)

    print(f"\n{'='*60}")
    print(f"  Evaluation complete")
    print(f"{'='*60}")
    print(f"\nResults: {os.path.join(eval_dir, 'results.md')}")
    print(f"Raw data: {os.path.join(eval_dir, 'results.json')}")
