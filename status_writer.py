"""
Bearing - Status writer

Writes human-readable status.md files from the current task queue state.
This is the orchestrator's primary output — designed to be read in an editor.
"""

from datetime import datetime
from tasks_schema import TaskQueue, TaskStatus, CheckpointLevel


STATUS_ICONS = {
    TaskStatus.QUEUED: "⬜",
    TaskStatus.RUNNING: "🔄",
    TaskStatus.COMPLETED: "✅",
    TaskStatus.FAILED: "❌",
    TaskStatus.AWAITING_REVIEW: "⏸️",
    TaskStatus.SKIPPED: "⏭️",
}


def format_tokens(result) -> str:
    """Format token count with cache breakdown when relevant."""
    total = result.total_tokens
    if total == 0:
        return ""
    if result.cache_read_tokens > 0:
        fresh = result.fresh_input_tokens
        cached = result.cache_read_tokens
        return f"{total:,} tokens ({cached:,} cached)"
    return f"{total:,} tokens"


def write_status(queue: TaskQueue, path: str, message: str = ""):
    """
    Write a status.md file summarizing the current state of all tasks.

    Designed to be glanceable — you open it in VS Code and immediately
    know what's done, what's running, what needs attention.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"# Bearing Status — {queue.project}",
        f"_Last updated: {now}_",
        "",
    ]

    if message:
        lines.extend([f"> **{message}**", ""])

    # Summary counts
    counts = {}
    for t in queue.tasks:
        counts[t.result.status] = counts.get(t.result.status, 0) + 1

    total = len(queue.tasks)
    completed = counts.get(TaskStatus.COMPLETED, 0)
    failed = counts.get(TaskStatus.FAILED, 0)
    paused = counts.get(TaskStatus.AWAITING_REVIEW, 0)

    lines.extend([
        f"**Progress: {completed}/{total} tasks complete**"
        + (f" | {failed} failed" if failed else "")
        + (f" | {paused} awaiting review" if paused else ""),
        "",
    ])

    # Usage summary — cost is the most accurate metric we have
    total_cost = queue.total_cost
    if total_cost > 0:
        lines.extend([f"**Session cost: ${total_cost:.2f}**", ""])

    lines.extend(["---", ""])

    # Task details
    for task in queue.tasks:
        icon = STATUS_ICONS.get(task.result.status, "?")
        lines.append(f"## {icon} {task.id}: {task.name}")
        lines.append("")

        # Config summary
        cfg = task.config
        config_parts = [
            f"model={cfg.model}",
            f"effort={cfg.effort}",
            f"budget=${cfg.budget_usd:.2f}",
            f"turns<={cfg.max_turns}",
        ]
        if cfg.worktree:
            config_parts.append(f"worktree={cfg.worktree}")
        lines.append(f"Config: `{' | '.join(config_parts)}`")
        lines.append("")

        # Dependencies
        if task.depends_on:
            lines.append(f"Depends on: {', '.join(task.depends_on)}")
            lines.append("")

        # Checkpoint policy
        if task.checkpoint != CheckpointLevel.AUTO:
            lines.append(f"Checkpoint: **{task.checkpoint.value}**")
            lines.append("")

        # Result details (if task has run)
        result = task.result
        if result.status != TaskStatus.QUEUED:
            usage_parts = []
            if result.cost_usd > 0:
                usage_parts.append(f"${result.cost_usd:.2f}")
            if result.turns_used > 0:
                usage_parts.append(f"{result.turns_used} turns")
            tok_str = format_tokens(result)
            if tok_str:
                usage_parts.append(tok_str)
            if usage_parts:
                lines.append(" | ".join(usage_parts))
                lines.append("")

            # Compression and relevance metrics
            if result.context_chars_original > 0 and result.context_chars_compressed > 0:
                lines.append(
                    f"Context: {result.context_chars_original} → "
                    f"{result.context_chars_compressed} chars"
                )
                if result.chunks_kept or result.chunks_dropped:
                    lines.append(
                        f"Relevance: kept={result.chunks_kept} "
                        f"compressed={result.chunks_compressed} "
                        f"dropped={result.chunks_dropped}"
                    )
                if result.scoring_latency_ms:
                    lines.append(
                        f"Latency: scoring={result.scoring_latency_ms}ms "
                        f"compression={result.compression_latency_ms}ms"
                    )
                lines.append("")

            if result.summary:
                lines.append("### Result")
                lines.append(result.summary[:1500])
                lines.append("")

            if result.error:
                lines.append("### Error")
                lines.append(f"```\n{result.error[:500]}\n```")
                lines.append("")

            if result.retry_count > 0:
                lines.append(f"_Retried {result.retry_count} time(s)_")
                lines.append("")

        lines.extend(["---", ""])

    # Footer
    next_task = queue.next_task()
    if queue.is_paused():
        lines.append("## Paused -- awaiting human review")
        lines.append("Review the results above, then edit `tasks.json` to continue.")
    elif queue.has_failures():
        lines.append("## Stopped -- task failure")
        lines.append("Check the error above. Fix `tasks.json` and re-run.")
    elif next_task:
        lines.append(f"## Next up: {next_task.id} -- {next_task.name}")
    else:
        lines.append("## All tasks complete")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
