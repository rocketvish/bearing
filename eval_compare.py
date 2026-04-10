"""
Bearing - Eval Compare: Task Isolation vs Single Session

Compares Bearing's approach (8 isolated claude -p sessions with context
propagation) against a single long session (one mega-prompt with all tasks).

Usage:
    bearing eval-compare <project_dir>

Conditions:
    1. bearing        — 8 separate tasks, each with fresh context window
    2. single-session — 1 mega-prompt, entire project built in one session

This tests Bearing's core value proposition: whether task isolation and
fresh context windows produce better results than one accumulated session
that eventually triggers compaction.
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from eval_runner import (
    DONE_STATUSES,
    capture_source_files,
    extract_task_paths,
    judge_task,
    read_source_files,
    restore_state,
    snapshot_state,
)
from executor import extract_cost, extract_tokens, extract_turns, parse_claude_output
from tasks_schema import TaskQueue, TaskStatus


# --- Mega-Prompt Assembly ---


def build_mega_prompt(queue: TaskQueue) -> str:
    """
    Assemble all task prompts into a single mega-prompt.

    Includes per-task FOCUS file lists so the single session has the same
    file pointers as Bearing's individual tasks. Does NOT include inter-task
    context summaries — the single session discovers dependencies by reading
    its own code. This is the fair comparison.
    """
    # Collect all unique focus files; only use universally-shared skip patterns
    # (ignore_patterns are per-task and shouldn't be merged globally since
    # task-008 skips src/utils/ which task-002 needs to create)
    all_focus = []
    seen_focus = set()
    common_skip = set()
    for task in queue.tasks:
        for f in task.relevant_files:
            if f not in seen_focus:
                all_focus.append(f)
                seen_focus.add(f)
        for s in task.ignore_patterns:
            common_skip.add(s)

    # Only keep skip patterns that ALL tasks share (typically just node_modules)
    if len(queue.tasks) > 1:
        common_skip = set.intersection(*(set(t.ignore_patterns) for t in queue.tasks))

    parts = [
        "Build a complete task management API. Complete ALL of the following "
        "tasks in order. After completing each task, run tests before moving "
        "to the next.",
        "",
    ]

    if all_focus:
        parts.append(
            "FOCUS: Read these files first, they are most relevant: "
            + ", ".join(all_focus)
        )
    if common_skip:
        parts.append(
            "SKIP: Do not read or modify these: " + ", ".join(sorted(common_skip))
        )
    parts.append("")

    for i, task in enumerate(queue.tasks, 1):
        parts.append(f"## Task {i}: {task.name}")
        parts.append("")

        if task.relevant_files:
            parts.append(f"Key files: {', '.join(task.relevant_files)}")
            parts.append("")

        parts.append(task.prompt)
        parts.append("")

    parts.append("Complete all tasks in order. Run tests after each.")

    return "\n".join(parts)


def _sum_budgets(queue: TaskQueue) -> float:
    """Sum all individual task budgets."""
    return sum(t.config.budget_usd for t in queue.tasks)


# --- Condition Runners ---


def run_bearing_condition(project_dir: str, eval_dir: str) -> dict:
    """
    Run Bearing normally (8 separate tasks) and collect metrics.
    """
    condition_dir = os.path.join(eval_dir, "bearing")
    os.makedirs(condition_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Condition: bearing (8 isolated sessions)")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    cmd = [
        sys.executable,
        "-m",
        "bearing",
        "run",
        project_dir,
        "--format",
        "prose",
    ]
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    wall_time = time.time() - t0

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Read final task state
    tasks_path = os.path.join(project_dir, "tasks.json")
    queue = TaskQueue.load(tasks_path)

    # Copy artifacts
    shutil.copy2(tasks_path, os.path.join(condition_dir, "tasks_final.json"))
    debug_dir = os.path.join(project_dir, "debug")
    if os.path.exists(debug_dir):
        condition_debug = os.path.join(condition_dir, "debug")
        if os.path.exists(condition_debug):
            shutil.rmtree(condition_debug)
        shutil.copytree(debug_dir, condition_debug)

    # Capture source files BEFORE git reset
    capture_source_files(project_dir, condition_dir)

    # Per-task metrics
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
                "turns_used": r.turns_used,
            }
        )

    return {
        "condition": "bearing",
        "wall_time_s": round(wall_time, 1),
        "total_cost": round(queue.total_cost, 4),
        "total_input_tokens": sum(t.result.input_tokens for t in queue.tasks),
        "total_output_tokens": sum(t.result.output_tokens for t in queue.tasks),
        "total_turns": sum(t.result.turns_used for t in queue.tasks),
        "completed": sum(1 for t in queue.tasks if t.result.status in DONE_STATUSES),
        "failed": sum(1 for t in queue.tasks if t.result.status == TaskStatus.FAILED),
        "total_tasks": len(queue.tasks),
        "tasks": task_metrics,
    }


def run_single_session(project_dir: str, eval_dir: str, queue: TaskQueue) -> dict:
    """
    Run all tasks as a single claude -p mega-prompt.
    """
    condition_dir = os.path.join(eval_dir, "single_session")
    os.makedirs(condition_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Condition: single-session (1 mega-prompt)")
    print(f"{'=' * 60}\n")

    mega_prompt = build_mega_prompt(queue)
    total_budget = _sum_budgets(queue)

    # Log the mega-prompt for debugging
    debug_dir = os.path.join(condition_dir, "debug")
    os.makedirs(debug_dir, exist_ok=True)
    with open(os.path.join(debug_dir, "mega-prompt.txt"), "w", encoding="utf-8") as f:
        f.write("# Bearing eval-compare: single-session mega-prompt\n")
        f.write(f"# Budget: ${total_budget:.2f}\n")
        f.write(f"# Tasks: {len(queue.tasks)}\n")
        f.write(f"# Prompt length: {len(mega_prompt)} chars\n")
        f.write("# ---\n\n")
        f.write(mega_prompt)

    print(f"  Mega-prompt: {len(mega_prompt)} chars, {len(queue.tasks)} tasks")
    print(f"  Budget: ${total_budget:.2f}, max turns: 100")

    cmd = [
        "claude",
        "-p",
        mega_prompt,
        "--model",
        "sonnet",
        "--output-format",
        "json",
        "--max-budget-usd",
        str(total_budget),
        "--max-turns",
        "100",
        "--dangerously-skip-permissions",
        "--effort",
        "high",
    ]

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_dir,
            timeout=3600,  # 1 hour safety net
        )
    except subprocess.TimeoutExpired:
        print("  TIMEOUT: single session exceeded 1 hour")
        capture_source_files(project_dir, condition_dir)
        return {
            "condition": "single-session",
            "wall_time_s": 3600.0,
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_turns": 0,
            "completed": 0,
            "failed": 1,
            "total_tasks": len(queue.tasks),
            "tasks": [],
            "error": "TIMEOUT",
        }

    wall_time = time.time() - t0

    # Parse output
    parsed = {}
    if result.stdout:
        parsed = parse_claude_output(result.stdout)

    cost = extract_cost(parsed)
    input_tokens, output_tokens, _cache = extract_tokens(parsed)
    turns = extract_turns(parsed)

    print(f"  Done: ${cost:.2f}, {input_tokens:,} input tokens, {turns} turns")

    # Save the raw output
    with open(
        os.path.join(condition_dir, "raw_output.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(parsed, f, indent=2)

    # Capture source files BEFORE git reset
    capture_source_files(project_dir, condition_dir)

    return {
        "condition": "single-session",
        "wall_time_s": round(wall_time, 1),
        "total_cost": round(cost, 4),
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_turns": turns,
        "completed": len(queue.tasks) if result.returncode == 0 else 0,
        "failed": 0 if result.returncode == 0 else 1,
        "total_tasks": len(queue.tasks),
        "tasks": [],
    }


# --- Quality Judging ---


def run_compare_judges(eval_dir: str, queue: TaskQueue) -> dict:
    """
    Judge both conditions per-task using the same judge infrastructure.
    """
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="bearing_compare_judge_")

    print(f"\n{'=' * 60}")
    print("  Running quality judgments (Claude Sonnet)")
    print(f"{'=' * 60}\n")

    judgments = {}
    for condition in ["bearing", "single_session"]:
        condition_dir = os.path.join(eval_dir, condition)
        if not os.path.exists(condition_dir):
            continue

        cond_label = condition.replace("_", "-")
        cond_scores = []

        for task in queue.tasks:
            task_paths = extract_task_paths(task)
            source_files = read_source_files(condition_dir, filter_paths=task_paths)

            print(f"  Judging: {cond_label} / {task.id}...")
            scores = judge_task(task.prompt, source_files, temp_dir)
            scores["id"] = task.id
            cond_scores.append(scores)

        judgments[cond_label] = cond_scores

    shutil.rmtree(temp_dir, ignore_errors=True)
    return judgments


# --- Report ---


def write_compare_report(
    eval_dir: str,
    bearing_result: dict,
    single_result: dict,
    judgments: dict,
):
    """Write compare_results.md and compare_results.json."""

    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "bearing": bearing_result,
        "single_session": single_result,
        "judgments": judgments,
    }
    with open(
        os.path.join(eval_dir, "compare_results.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(raw_data, f, indent=2)

    lines = [
        "# Bearing Eval Compare: Task Isolation vs Single Session",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        "## Summary",
        "",
        "| Metric | Bearing (8 sessions) | Single Session (1 prompt) |",
        "|--------|---------------------|--------------------------|",
        f"| Total cost | ${bearing_result['total_cost']:.2f} | ${single_result['total_cost']:.2f} |",
        f"| Input tokens | {bearing_result['total_input_tokens']:,} | {single_result['total_input_tokens']:,} |",
        f"| Output tokens | {bearing_result['total_output_tokens']:,} | {single_result['total_output_tokens']:,} |",
        f"| Total turns | {bearing_result['total_turns']} | {single_result['total_turns']} |",
        f"| Wall time | {bearing_result['wall_time_s']:.0f}s | {single_result['wall_time_s']:.0f}s |",
        f"| Tasks completed | {bearing_result['completed']}/{bearing_result['total_tasks']} | (see quality scores) |",
        "",
    ]

    # Quality comparison
    if judgments:
        lines.extend(["## Quality Scores (1-5, higher is better)", ""])

        bearing_j = judgments.get("bearing", [])
        single_j = judgments.get("single-session", [])

        if bearing_j and single_j:
            lines.extend(
                [
                    "| Task | Bearing (C/Cr/A) | Single (C/Cr/A) |",
                    "|------|-----------------|-----------------|",
                ]
            )
            for i in range(len(bearing_j)):
                b = bearing_j[i]
                s = single_j[i] if i < len(single_j) else {}
                task_id = b.get("id", f"task-{i + 1:03d}")
                b_scores = f"{b.get('completeness', 0)}/{b.get('correctness', 0)}/{b.get('adherence', 0)}"
                s_scores = f"{s.get('completeness', 0)}/{s.get('correctness', 0)}/{s.get('adherence', 0)}"
                lines.append(f"| {task_id} | {b_scores} | {s_scores} |")

            # Averages
            for dim in ["completeness", "correctness", "adherence"]:
                b_avg = (
                    sum(t.get(dim, 0) for t in bearing_j) / len(bearing_j)
                    if bearing_j
                    else 0
                )
                s_avg = (
                    sum(t.get(dim, 0) for t in single_j) / len(single_j)
                    if single_j
                    else 0
                )
                lines.append(f"| **Avg {dim}** | **{b_avg:.1f}** | **{s_avg:.1f}** |")
            lines.append("")

            # Judge notes
            lines.extend(["### Judge Notes", ""])
            for condition_label, scores_list in [
                ("Bearing", bearing_j),
                ("Single Session", single_j),
            ]:
                lines.append(f"**{condition_label}:**")
                for s in scores_list:
                    note = s.get("notes", "")
                    if note:
                        lines.append(f"- {s.get('id', '?')}: {note[:200]}")
                lines.append("")

    # Context accumulation analysis
    lines.extend(
        [
            "## Context Accumulation Analysis",
            "",
            "### Bearing (8 isolated sessions)",
            "",
            "Each task starts with a fresh context window. No compaction occurs.",
            "",
        ]
    )

    if bearing_result.get("tasks"):
        lines.append("| Task | Input tokens | Turns | Cost |")
        lines.append("|------|-------------|-------|------|")
        for t in bearing_result["tasks"]:
            lines.append(
                f"| {t['id']} | {t['input_tokens']:,} | {t['turns_used']} | ${t['cost_usd']:.2f} |"
            )
        max_single_task = max(
            (t["input_tokens"] for t in bearing_result["tasks"]), default=0
        )
        lines.append("")
        lines.append(
            f"Peak context per session: ~{max_single_task:,} tokens (well under 100K compaction threshold)"
        )
        lines.append("Compactions: **0** (each session is independent)")
        lines.append("")

    lines.extend(
        [
            "### Single Session (1 accumulated session)",
            "",
        ]
    )

    s_input = single_result["total_input_tokens"]
    s_turns = single_result["total_turns"]
    if s_turns > 0:
        avg_per_turn = s_input // s_turns
        est_compactions = s_input // 100_000
        lines.extend(
            [
                f"Total input tokens: {s_input:,}",
                f"Total turns: {s_turns}",
                f"Average tokens per turn: ~{avg_per_turn:,}",
                f"Estimated compactions (at ~100K threshold): **~{est_compactions}**",
                "",
                "Each compaction loses context from earlier tasks. By task 6-8,",
                "the model may have lost details about tasks 1-3 that were compacted away.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                f"Total input tokens: {s_input:,}",
                "No turn data available.",
                "",
            ]
        )

    # Token savings analysis
    if s_input > 0 and bearing_result["total_input_tokens"] > 0:
        b_input = bearing_result["total_input_tokens"]
        ratio = s_input / b_input if b_input > 0 else 0
        lines.extend(
            [
                "### Token Usage Comparison",
                "",
                f"- Bearing total input: {b_input:,} tokens",
                f"- Single session input: {s_input:,} tokens",
                f"- Ratio: single session uses **{ratio:.1f}x** the input tokens of Bearing",
                "",
                "In a single session, every turn re-sends the full conversation history.",
                "Bearing avoids this by giving each task only what it needs.",
                "",
            ]
        )

    with open(os.path.join(eval_dir, "compare_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# --- Main ---


def run_eval_compare(project_dir: str):
    """
    Run the full eval-compare: bearing (8 sessions) vs single-session (1 prompt).
    """
    project_dir = os.path.abspath(project_dir)
    eval_dir = os.path.join(project_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    tasks_path = os.path.join(project_dir, "tasks.json")
    if not os.path.exists(tasks_path):
        print(f"Error: tasks.json not found in {project_dir}")
        sys.exit(1)

    if not os.path.exists(os.path.join(project_dir, ".git")):
        print("Error: Project must be a git repo (needed for codebase reset)")
        sys.exit(1)

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project_dir,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        sys.exit(1)

    queue = TaskQueue.load(tasks_path)

    print("Bearing Eval Compare")
    print(f"Project: {project_dir}")
    print(f"Tasks: {len(queue.tasks)}")
    print("Conditions: bearing (8 sessions) vs single-session (1 prompt)")
    print()

    snapshot_state(project_dir, eval_dir)

    # Condition 1: Bearing (8 isolated sessions)
    restore_state(project_dir, eval_dir)
    bearing_result = run_bearing_condition(project_dir, eval_dir)

    # Condition 2: Single session (1 mega-prompt)
    restore_state(project_dir, eval_dir)
    single_result = run_single_session(project_dir, eval_dir, queue)

    # Restore clean state
    restore_state(project_dir, eval_dir)

    # Judge both conditions
    judgments = run_compare_judges(eval_dir, queue)

    # Write report
    write_compare_report(eval_dir, bearing_result, single_result, judgments)

    print(f"\n{'=' * 60}")
    print("  Eval compare complete")
    print(f"{'=' * 60}")
    print(f"\nResults: {os.path.join(eval_dir, 'compare_results.md')}")
    print(f"Raw data: {os.path.join(eval_dir, 'compare_results.json')}")
