"""
Bearing - Eval: Agent Compression & Retrieval

Runs all 8 tasks as a single mega-prompt API-backed agent session under seven conditions:
    1. agent-raw              — No compression, no caching (baseline)
    2. agent-compressed       — API compression at threshold
    3. agent-cached           — Prompt caching, no compression
    4. agent-compressed-cached — Both compression + caching
    5. agent-retrieval        — Embedding-based selective history retrieval
    6. agent-retrieval-cached — Retrieval + prompt caching
    7. claude-p               — Claude Code CLI (or cached from eval-compare)

The mega-prompt naturally accumulates 30K+ tokens as the agent builds 8 features,
triggering compression/retrieval events. The key output is a per-turn input token
table showing accumulation curves, compression sawtooth, and retrieval drops.

Usage:
    bearing eval-agent <project_dir>
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

from agent import (
    COST_CACHE_CREATION_PER_MTOK,
    COST_CACHE_READ_PER_MTOK,
    COST_INPUT_PER_MTOK,
    COST_OUTPUT_PER_MTOK,
    pricing_for,
    run_agent,
)
from eval_compare import build_mega_prompt
from eval_runner import (
    capture_source_files,
    extract_task_paths,
    judge_task,
    read_source_files,
    restore_state,
    snapshot_state,
)
from executor import extract_cost, extract_tokens, extract_turns
from tasks_schema import TaskQueue


CONDITIONS = [
    "agent-raw",
    "agent-compressed",
    "agent-cached",
    "agent-compressed-cached",
    "agent-retrieval",
    "agent-retrieval-cached",
    "claude-p",
]

AGENT_MAX_TURNS = 80
COMPRESSION_THRESHOLD = 12000
COOLDOWN_SECONDS = 90
AGENT_PROVIDER = os.environ.get("BEARING_AGENT_PROVIDER", "openai")
AGENT_MODEL = os.environ.get("BEARING_AGENT_MODEL")


def _run_agent_condition(
    mega_prompt: str,
    project_dir: str,
    condition: str,
    condition_dir: str,
) -> dict:
    """Run one of the agent conditions with the mega-prompt."""
    if "retrieval" in condition:
        compression_mode = "retrieval"
    elif "compressed" in condition:
        compression_mode = "api"
    else:
        compression_mode = "none"
    use_caching = "cached" in condition

    print(
        f"\n  Running agent (compression={compression_mode}, "
        f"caching={use_caching}, max_turns={AGENT_MAX_TURNS})..."
    )
    result = run_agent(
        task_prompt=mega_prompt,
        project_dir=project_dir,
        provider=AGENT_PROVIDER,
        model=AGENT_MODEL,
        compression_mode=compression_mode,
        compression_threshold=COMPRESSION_THRESHOLD,
        max_turns=AGENT_MAX_TURNS,
        use_caching=use_caching,
    )

    # Capture source files
    capture_source_files(project_dir, condition_dir)

    return {
        "condition": condition,
        "provider": result.get("provider", AGENT_PROVIDER),
        "model": result.get("model", AGENT_MODEL),
        "status": result["status"],
        "total_input_tokens": result["total_input_tokens"],
        "total_output_tokens": result["total_output_tokens"],
        "total_cache_read_tokens": result.get("total_cache_read_tokens", 0),
        "total_cache_creation_tokens": result.get("total_cache_creation_tokens", 0),
        "total_thinking_tokens": result.get("total_thinking_tokens", 0),
        "turns_used": result["turns_used"],
        "per_turn_input_tokens": result["per_turn_input_tokens"],
        "per_turn_output_tokens": result["per_turn_output_tokens"],
        "per_turn_cache_read_tokens": result.get("per_turn_cache_read_tokens", []),
        "per_turn_cache_creation_tokens": result.get(
            "per_turn_cache_creation_tokens", []
        ),
        "per_turn_thinking_tokens": result.get("per_turn_thinking_tokens", []),
        "compressions": result["compressions"],
        "retrievals": result.get("retrievals", []),
        "cost_usd": result["cost_usd"],
        "wall_time_s": result["wall_time_s"],
        "summary": result["summary"],
    }


def _load_claude_p_from_cache(eval_dir: str) -> dict | None:
    """
    Try to load claude-p results from a previous eval-compare run.
    Returns a result dict if compare_results.json exists, None otherwise.
    """
    cache_path = os.path.join(eval_dir, "compare_results.json")
    if not os.path.exists(cache_path):
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    single = data.get("single_session")
    if not single:
        return None

    print("  Loaded claude-p results from eval/compare_results.json (cached)")
    return {
        "condition": "claude-p",
        "status": "completed" if single.get("completed", 0) > 0 else "error",
        "total_input_tokens": single.get("total_input_tokens", 0),
        "total_output_tokens": single.get("total_output_tokens", 0),
        "total_cache_read_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_thinking_tokens": 0,
        "turns_used": single.get("total_turns", 0),
        "per_turn_input_tokens": [],
        "per_turn_output_tokens": [],
        "per_turn_cache_read_tokens": [],
        "per_turn_cache_creation_tokens": [],
        "per_turn_thinking_tokens": [],
        "compressions": [],
        "retrievals": [],
        "cost_usd": round(single.get("total_cost", 0), 4),
        "wall_time_s": round(single.get("wall_time_s", 0), 1),
        "summary": "(cached from eval-compare)",
    }


def _run_claude_p_condition(
    mega_prompt: str,
    queue: TaskQueue,
    project_dir: str,
    condition_dir: str,
) -> dict:
    """Run claude -p with the mega-prompt, same as eval_compare's single session."""
    print("\n  Running claude -p with mega-prompt...")

    total_budget = sum(t.config.budget_usd for t in queue.tasks)

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

    print(f"  Mega-prompt: {len(mega_prompt)} chars, budget: ${total_budget:.2f}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=project_dir,
            timeout=3600,
        )
    except subprocess.TimeoutExpired:
        capture_source_files(project_dir, condition_dir)
        return {
            "condition": "claude-p",
            "status": "error",
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "turns_used": 0,
            "per_turn_input_tokens": [],
            "per_turn_output_tokens": [],
            "compressions": [],
            "retrievals": [],
            "cost_usd": 0.0,
            "wall_time_s": 3600.0,
            "summary": "Timed out",
        }

    wall_time = time.time() - t0

    # Capture source files BEFORE any reset
    capture_source_files(project_dir, condition_dir)

    # Parse output
    parsed = {}
    if result.stdout:
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError:
            parsed = {"result": result.stdout}

    cost = extract_cost(parsed)
    input_tokens, output_tokens, _ = extract_tokens(parsed)
    turns = extract_turns(parsed)

    summary = ""
    if isinstance(parsed.get("result"), str):
        summary = parsed["result"][:2000]

    return {
        "condition": "claude-p",
        "status": "completed" if result.returncode == 0 else "error",
        "total_input_tokens": input_tokens,
        "total_output_tokens": output_tokens,
        "total_cache_read_tokens": 0,
        "total_cache_creation_tokens": 0,
        "total_thinking_tokens": 0,
        "turns_used": turns,
        "per_turn_input_tokens": [],
        "per_turn_output_tokens": [],
        "per_turn_cache_read_tokens": [],
        "per_turn_cache_creation_tokens": [],
        "per_turn_thinking_tokens": [],
        "compressions": [],
        "retrievals": [],
        "cost_usd": round(cost, 4),
        "wall_time_s": round(wall_time, 1),
        "summary": summary,
    }


def _run_judges(
    eval_dir: str,
    queue: TaskQueue,
    all_results: list[dict],
) -> dict:
    """
    Run per-task quality judgments on all conditions.
    Each task is judged individually against source files produced.
    """
    import tempfile

    temp_dir = tempfile.mkdtemp(prefix="bearing_agent_judge_")

    print(f"\n{'=' * 60}")
    print("  Running quality judgments (Claude Sonnet)")
    print(f"  Judging {len(queue.tasks)} tasks per condition")
    print(f"{'=' * 60}\n")

    judgments = {}

    for cond_result in all_results:
        condition = cond_result["condition"]
        condition_dir = os.path.join(eval_dir, condition)

        if not os.path.exists(condition_dir):
            continue

        cond_scores = []
        for task in queue.tasks:
            task_paths = extract_task_paths(task)
            source_files = read_source_files(condition_dir, filter_paths=task_paths)

            print(f"  Judging: {condition} / {task.id}...")
            scores = judge_task(task.prompt, source_files, temp_dir)
            scores["id"] = task.id
            scores["name"] = task.name
            cond_scores.append(scores)

        judgments[condition] = cond_scores

    shutil.rmtree(temp_dir, ignore_errors=True)
    return judgments


def _write_report(
    eval_dir: str,
    all_results: list[dict],
    judgments: dict,
    queue: TaskQueue,
):
    """Write agent_results.md and agent_results.json."""
    # --- JSON ---
    raw_data = {
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "judgments": judgments,
    }
    with open(os.path.join(eval_dir, "agent_results.json"), "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2)

    # --- Markdown ---
    lines = [
        "# Agent Compression Eval Results",
        f"_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
        "",
        f"All {len(queue.tasks)} tasks run as a single mega-prompt agent session.",
        "",
        "## Summary",
        "",
    ]

    # Build summary table header dynamically from CONDITIONS
    # Abbreviate long condition names for table readability
    cond_labels = {c: c for c in CONDITIONS}
    cond_labels["agent-compressed-cached"] = "comp-cached"
    cond_labels["agent-retrieval"] = "retrieval"
    cond_labels["agent-retrieval-cached"] = "ret-cached"

    header = "| Metric |"
    sep = "|--------|"
    for cond in CONDITIONS:
        header += f" {cond_labels[cond]} |"
        sep += "----------|"
    lines.extend([header, sep])

    def _val(condition, key, fmt=str):
        for r in all_results:
            if r["condition"] == condition:
                return fmt(r.get(key, 0))
        return "-"

    metrics_rows = [
        ("Total input tokens", "total_input_tokens", lambda x: f"{x:,}"),
        ("Total output tokens", "total_output_tokens", lambda x: f"{x:,}"),
        ("Cache Read Tokens", "total_cache_read_tokens", lambda x: f"{x:,}"),
        ("Cache Creation Tokens", "total_cache_creation_tokens", lambda x: f"{x:,}"),
        ("Reasoning Tokens", "total_thinking_tokens", lambda x: f"{x:,}"),
        ("Cost USD", "cost_usd", lambda x: f"${x:.4f}"),
        ("Turns", "turns_used", str),
        ("Wall time", "wall_time_s", lambda x: f"{x:.0f}s"),
        ("Status", "status", str),
    ]

    for label, key, fmt in metrics_rows:
        row = f"| {label} |"
        for cond in CONDITIONS:
            row += f" {_val(cond, key, fmt)} |"
        lines.append(row)

    # Cache Hit Rate row
    row = "| Cache Hit Rate |"
    for cond in CONDITIONS:
        found = False
        for r in all_results:
            if r["condition"] == cond:
                total_in = r.get("total_input_tokens", 0)
                cache_read = r.get("total_cache_read_tokens", 0)
                rate = cache_read / total_in if total_in > 0 else 0
                row += f" {rate:.1%} |"
                found = True
                break
        if not found:
            row += " - |"
    lines.append(row)

    # Input/Output ratio row
    row = "| Input/Output ratio |"
    for cond in CONDITIONS:
        found = False
        for r in all_results:
            if r["condition"] == cond:
                out = r.get("total_output_tokens", 0)
                ratio = r.get("total_input_tokens", 0) / out if out > 0 else 0
                row += f" {ratio:.1f} |"
                found = True
                break
        if not found:
            row += " - |"
    lines.append(row)

    lines.extend(["", ""])

    # --- Cost Breakdown per condition ---
    lines.extend(["## Cost Breakdown", ""])

    for cond in CONDITIONS:
        r = next((r for r in all_results if r["condition"] == cond), None)
        if not r:
            continue

        total_in = r.get("total_input_tokens", 0)
        cache_read = r.get("total_cache_read_tokens", 0)
        cache_creation = r.get("total_cache_creation_tokens", 0)
        uncached = total_in - cache_read - cache_creation
        output = r.get("total_output_tokens", 0)
        thinking = r.get("total_thinking_tokens", 0)
        pricing = pricing_for(r.get("provider", "openai"), r.get("model"))
        input_price = pricing.get("input", COST_INPUT_PER_MTOK)
        cached_price = pricing.get("cached_input", COST_CACHE_READ_PER_MTOK)
        cache_creation_price = pricing.get(
            "cache_creation", COST_CACHE_CREATION_PER_MTOK
        )
        output_price = pricing.get("output", COST_OUTPUT_PER_MTOK)
        bill_reasoning_separately = pricing.get("reasoning_billed_separately", False)

        c_uncached = uncached * input_price / 1_000_000
        c_cache_read = cache_read * cached_price / 1_000_000
        c_cache_creation = cache_creation * cache_creation_price / 1_000_000
        c_output = output * output_price / 1_000_000
        c_thinking = (
            thinking * output_price / 1_000_000 if bill_reasoning_separately else 0
        )
        c_total = c_uncached + c_cache_read + c_cache_creation + c_output + c_thinking

        lines.extend(
            [
                f"### {cond}",
                "```",
                f"Input (uncached): {uncached:>8,} tokens"
                f" x ${input_price:.2f}/MTok = ${c_uncached:.4f}",
                f"Cache reads:      {cache_read:>8,} tokens"
                f" x ${cached_price:.2f}/MTok = ${c_cache_read:.4f}",
                f"Cache writes:     {cache_creation:>8,} tokens"
                f" x ${cache_creation_price:.2f}/MTok = "
                f"${c_cache_creation:.4f}",
                f"Output:           {output:>8,} tokens"
                f" x ${output_price:.2f}/MTok = ${c_output:.4f}",
                f"Reasoning:        {thinking:>8,} tokens"
                f" x ${output_price:.2f}/MTok = ${c_thinking:.4f}",
                f"Total:            ${c_total:.4f}",
                "```",
                "",
            ]
        )

    # --- Per-turn token accumulation (per condition) ---
    agent_conditions = [c for c in CONDITIONS if c != "claude-p"]
    any_per_turn = any(
        r.get("per_turn_input_tokens")
        for r in all_results
        if r["condition"] != "claude-p"
    )

    if any_per_turn:
        lines.extend(["## Per-Turn Token Accumulation", ""])

        for cond in agent_conditions:
            r = next((x for x in all_results if x["condition"] == cond), None)
            if not r or not r.get("per_turn_input_tokens"):
                continue

            cache_read_list = r.get("per_turn_cache_read_tokens", [])
            thinking_list = r.get("per_turn_thinking_tokens", [])
            has_cache = any(t > 0 for t in cache_read_list)
            has_thinking = any(t > 0 for t in thinking_list)

            # Build compression and cache-reset turn sets
            comp_turns = {}
            for c in r.get("compressions", []):
                comp_turns[c["turn"]] = c
            cache_reset_turns = {c["turn"] + 1 for c in r.get("compressions", [])}

            # Dynamic columns
            hdr = "| Turn | Input |"
            sep_row = "|------|-------|"
            if has_cache:
                hdr += " Cached |"
                sep_row += "--------|"
            if has_thinking:
                hdr += " Reasoning |"
                sep_row += "----------|"
            hdr += " Output | Notes |"
            sep_row += "--------|-------|"

            lines.extend([f"### {cond}", "", hdr, sep_row])

            n_turns = len(r["per_turn_input_tokens"])
            for i in range(n_turns):
                row = f"| {i + 1} | {r['per_turn_input_tokens'][i]:,} |"
                if has_cache:
                    cr = cache_read_list[i] if i < len(cache_read_list) else 0
                    row += f" {cr:,} |"
                if has_thinking:
                    th = thinking_list[i] if i < len(thinking_list) else 0
                    row += f" {th:,} |"
                out_list = r.get("per_turn_output_tokens", [])
                out_val = out_list[i] if i < len(out_list) else 0
                row += f" {out_val:,} |"

                notes = []
                if (i + 1) in comp_turns:
                    c = comp_turns[i + 1]
                    notes.append(
                        f"compressed {c['tokens_before']:,} -> ~{c['tokens_after']:,}"
                    )
                if (i + 1) in cache_reset_turns:
                    notes.append("cache reset")
                row += f" {'; '.join(notes)} |"
                lines.append(row)

            lines.extend(["", ""])

    # --- Compression events ---
    any_compressions = any(
        r.get("compressions") for r in all_results if r["condition"] != "claude-p"
    )
    if any_compressions:
        lines.extend(["## Compression Events", ""])

        for cond in agent_conditions:
            r = next((x for x in all_results if x["condition"] == cond), None)
            if not r or not r.get("compressions"):
                continue
            lines.extend(
                [
                    f"### {cond}",
                    "",
                    "| Turn | Tokens Before | Tokens After"
                    " | Ratio | Compression Tokens |",
                    "|------|--------------|-------------|-------|-------------------|",
                ]
            )
            for c in r["compressions"]:
                before = c["tokens_before"]
                after = c["tokens_after"]
                ratio = before / after if after > 0 else 0
                comp_tokens = c.get("compression_tokens", 0)
                lines.append(
                    f"| {c['turn']} | {before:,} | {after:,} | "
                    f"{ratio:.1f}x | {comp_tokens:,} |"
                )
            lines.extend(["", ""])

    # --- Retrieval Analysis ---
    any_retrievals = any(
        r.get("retrievals") for r in all_results if r["condition"] != "claude-p"
    )
    if any_retrievals:
        lines.extend(["## Retrieval Analysis", ""])

        for cond in agent_conditions:
            r = next((x for x in all_results if x["condition"] == cond), None)
            if not r or not r.get("retrievals"):
                continue

            lines.extend([f"### {cond}", ""])

            for ret in r["retrievals"]:
                lines.extend(
                    [
                        f"**Turn {ret['turn']}** — "
                        f"retrieved {ret['retrieved']}/{ret['total_stored']} turns "
                        f"(tokens before: {ret['tokens_before']:,}, "
                        f"ollama: {'yes' if ret['ollama_available'] else 'no'})",
                        "",
                    ]
                )

                scores = ret.get("scores", [])
                if scores:
                    score_vals = [s for _, s in scores]
                    min_s = min(score_vals)
                    max_s = max(score_vals)
                    mean_s = sum(score_vals) / len(score_vals)
                    sorted_vals = sorted(score_vals)
                    mid = len(sorted_vals) // 2
                    median_s = (
                        sorted_vals[mid]
                        if len(sorted_vals) % 2
                        else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
                    )

                    lines.extend(
                        [
                            "| Stat | Value |",
                            "|------|-------|",
                            f"| Min | {min_s:.4f} |",
                            f"| Max | {max_s:.4f} |",
                            f"| Mean | {mean_s:.4f} |",
                            f"| Median | {median_s:.4f} |",
                            "",
                            "Scores per turn:",
                            "",
                            "| Turn | Score | Status |",
                            "|------|-------|--------|",
                        ]
                    )
                    # Determine which turns were kept
                    kept_turns = set()
                    scored_sorted = sorted(scores, key=lambda x: x[1], reverse=True)
                    for i, (tn, _) in enumerate(scored_sorted):
                        if i < ret["retrieved"]:
                            kept_turns.add(tn)

                    for turn_num, score in scores:
                        status_str = "kept" if turn_num in kept_turns else "dropped"
                        lines.append(f"| {turn_num} | {score:.4f} | {status_str} |")

                    lines.extend(["", ""])

    # --- Per-task quality scores ---
    if judgments:
        lines.extend(
            [
                "## Quality Scores (1-5, higher is better)",
                "",
            ]
        )

        # Per-task table per condition
        for cond in CONDITIONS:
            cond_scores = judgments.get(cond, [])
            if not cond_scores:
                continue

            lines.extend(
                [
                    f"### {cond}",
                    "",
                    "| Task | Completeness | Correctness | Adherence | Notes |",
                    "|------|-------------|------------|-----------|-------|",
                ]
            )
            for s in cond_scores:
                task_id = s.get("id", "?")
                c = s.get("completeness", 0)
                cr = s.get("correctness", 0)
                a = s.get("adherence", 0)
                note = s.get("notes", "")[:100]
                lines.append(f"| {task_id} | {c} | {cr} | {a} | {note} |")

            if cond_scores:
                avg_c = sum(s.get("completeness", 0) for s in cond_scores) / len(
                    cond_scores
                )
                avg_cr = sum(s.get("correctness", 0) for s in cond_scores) / len(
                    cond_scores
                )
                avg_a = sum(s.get("adherence", 0) for s in cond_scores) / len(
                    cond_scores
                )
                lines.append(
                    f"| **Average** | **{avg_c:.1f}** | **{avg_cr:.1f}** | "
                    f"**{avg_a:.1f}** | |"
                )

            lines.extend(["", ""])

        # Cross-condition summary
        header = "| Dimension |"
        sep_row = "|-----------|"
        for cond in CONDITIONS:
            header += f" {cond_labels[cond]} |"
            sep_row += "----------|"
        lines.extend(
            [
                "### Summary Across Conditions",
                "",
                header,
                sep_row,
            ]
        )
        for dim in ["completeness", "correctness", "adherence"]:
            row = f"| {dim.title()} |"
            for cond in CONDITIONS:
                cond_scores = judgments.get(cond, [])
                if cond_scores:
                    vals = [s.get(dim, 0) for s in cond_scores if s.get(dim, 0) > 0]
                    avg = sum(vals) / len(vals) if vals else 0
                    row += f" {avg:.1f} |"
                else:
                    row += " - |"
            lines.append(row)

        lines.extend(["", ""])

    # --- Interpretation ---
    lines.extend(
        [
            "## Notes on Interpretation",
            "",
            "**Per-turn accumulation:** In the agent-raw condition, input tokens grow",
            "monotonically as the full conversation history is re-sent each turn.",
            "In agent-compressed, the sawtooth pattern shows tokens growing then",
            "dropping at compression events.",
            "",
            "**Token efficiency ratio:** Total input / total output. Lower means less",
            "waste — less re-reading of history per unit of code produced.",
            "",
            "**Compression cost:** The compression call itself uses tokens. Net savings",
            "= (tokens saved on subsequent turns) - (compression call tokens).",
            "",
            "**Prompt caching:** `use_caching=True` enables provider-specific",
            "prompt caching. Reports include cached token usage when the provider",
            "returns it.",
            "",
            "**claude-p comparison:** Claude Code's internal session management is a",
            "black box — we can't see per-turn tokens, only the total. The comparison",
            "shows whether our explicit compression beats Claude's built-in approach.",
            "",
            "**Mega-prompt:** All 8 tasks are run as a single agent session to ensure",
            "enough context accumulates (30K+ tokens) to trigger compression. This",
            "matches the eval-compare methodology for fair comparison.",
            "",
            "**Retrieval vs compression:** Retrieval keeps the original messages",
            "verbatim (no lossy summarization) but drops less-relevant turns entirely.",
            "The quality tradeoff: compression loses detail across all turns, retrieval",
            "loses entire turns but keeps kept ones lossless. Score distribution",
            "(min/max/mean/median) shows whether embedding similarity meaningfully",
            "differentiates turns or clusters them together.",
            "",
        ]
    )

    with open(os.path.join(eval_dir, "agent_results.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_eval_agent(project_dir: str):
    """
    Run the agent compression eval: 7 conditions, all tasks as mega-prompt.
    """
    project_dir = os.path.abspath(project_dir)
    eval_dir = os.path.join(project_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    # Validate prerequisites
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
        encoding="utf-8",
        errors="replace",
    )
    if result.stdout.strip():
        print("Error: Uncommitted changes detected. Commit or stash first.")
        print("The eval runner resets the codebase between conditions.")
        sys.exit(1)

    queue = TaskQueue.load(tasks_path)
    mega_prompt = build_mega_prompt(queue)

    print("Bearing Agent Compression Eval")
    print(f"Project: {project_dir}")
    print(f"Tasks: {len(queue.tasks)} (mega-prompt, {len(mega_prompt)} chars)")
    print(f"Conditions: {', '.join(CONDITIONS)}")
    print(f"Agent provider: {AGENT_PROVIDER}")
    if AGENT_MODEL:
        print(f"Agent model: {AGENT_MODEL}")
    print(f"Agent max turns: {AGENT_MAX_TURNS}")
    print(f"Compression threshold: {COMPRESSION_THRESHOLD:,} tokens")
    print()

    snapshot_state(project_dir, eval_dir)

    all_results = []

    for cond_idx, condition in enumerate(CONDITIONS):
        if cond_idx > 0:
            print(
                f"\n  Waiting {COOLDOWN_SECONDS}s between conditions "
                f"(rate limit cooldown)..."
            )
            time.sleep(COOLDOWN_SECONDS)

        condition_dir = os.path.join(eval_dir, condition)
        os.makedirs(condition_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"  Condition: {condition}")
        print(f"{'=' * 60}")

        # Reset codebase
        restore_state(project_dir, eval_dir)

        if condition == "claude-p":
            # Try cached results from eval-compare first
            cond_result = _load_claude_p_from_cache(eval_dir)
            if cond_result is None:
                cond_result = _run_claude_p_condition(
                    mega_prompt, queue, project_dir, condition_dir
                )
        else:
            cond_result = _run_agent_condition(
                mega_prompt, project_dir, condition, condition_dir
            )

        print(
            f"\n  Result: {cond_result['status']} | "
            f"${cond_result['cost_usd']:.4f} | "
            f"{cond_result['total_input_tokens']:,} input tokens | "
            f"{cond_result['turns_used']} turns | "
            f"{cond_result['wall_time_s']:.0f}s"
        )

        # Save condition result
        with open(
            os.path.join(condition_dir, "result.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(cond_result, f, indent=2)

        all_results.append(cond_result)

    # Restore clean state
    restore_state(project_dir, eval_dir)

    # Judge all conditions (per-task)
    judgments = _run_judges(eval_dir, queue, all_results)

    # Write report
    _write_report(eval_dir, all_results, judgments, queue)

    print(f"\n{'=' * 60}")
    print("  Agent compression eval complete")
    print(f"{'=' * 60}")
    print(f"\nResults: {os.path.join(eval_dir, 'agent_results.md')}")
    print(f"Raw data: {os.path.join(eval_dir, 'agent_results.json')}")
