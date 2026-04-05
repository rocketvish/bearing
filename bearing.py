"""
Bearing -- Planning-to-Execution Orchestrator for Claude Code

A thin orchestration layer that reads a task queue, executes tasks
via `claude -p`, writes results to status.md, and pauses at
human checkpoints.

Usage:
    bearing start <project_dir>     Start a planning session
    bearing init <project_dir>      Create starter files
    bearing run <project_dir>       Execute queued tasks
    bearing status <project_dir>    Show full status
    bearing summary <project_dir>   Quick check (for planner)
    bearing watch <project_dir>     Live updates as tasks run
    bearing validate <project_dir>  Check tasks.json syntax

The workflow:
    1. cd your-project && bearing start .
    2. Discuss what you want to build with the planner
    3. Planner writes tasks.json for you
    4. !bearing run . (from the planner session)
    5. !bearing summary . (to check progress)
"""

import sys
import os
import json
import time
import subprocess

from tasks_schema import (
    TaskQueue, Task, TaskResult, TaskStatus,
    CheckpointLevel, FailurePolicy, ExecutionConfig,
)
from executor import run_task, check_cli_installed
from status_writer import write_status


TASKS_FILE = "tasks.json"
STATUS_FILE = "status.md"
PLAN_FILE = "plan.md"

# Bearing's own directory (where this script lives)
BEARING_DIR = os.path.dirname(os.path.abspath(__file__))
PLANNER_PROMPT_PATH = os.path.join(BEARING_DIR, "planner_prompt.md")


def init_project(project_dir: str):
    """Create starter bearing files in a project directory."""
    project_dir = os.path.abspath(project_dir)
    project_name = os.path.basename(project_dir)

    tasks_path = os.path.join(project_dir, TASKS_FILE)
    status_path = os.path.join(project_dir, STATUS_FILE)
    plan_path = os.path.join(project_dir, PLAN_FILE)

    if os.path.exists(tasks_path):
        print(f"  {TASKS_FILE} already exists, skipping")
    else:
        example_queue = TaskQueue(
            project=project_name,
            description="Describe your goals here",
            tasks=[
                Task(
                    id="task-001",
                    name="Example task",
                    prompt=(
                        "Read the current codebase first.\n\n"
                        "Plan first, show me the plan before implementing.\n\n"
                        "TODO: Replace this with your actual task.\n\n"
                        "Run tests after.\n"
                        "Update CLAUDE.md with any architectural decisions."
                    ),
                    config=ExecutionConfig(
                        model="sonnet",
                        effort="high",
                        budget_usd=3.00,
                        max_turns=20,
                    ),
                    checkpoint=CheckpointLevel.PAUSE,
                ),
            ],
        )
        example_queue.save(tasks_path)
        print(f"  Created {TASKS_FILE}")

    if not os.path.exists(plan_path):
        with open(plan_path, "w", encoding="utf-8") as f:
            f.write(f"# Plan -- {project_name}\n\n")
            f.write("Use this file for your thinking. Goals, priorities,\n")
            f.write("architectural decisions, notes from planning sessions.\n\n")
            f.write("The orchestrator does not read this file.\n")
            f.write("This is your space.\n")
        print(f"  Created {PLAN_FILE}")

    if os.path.exists(tasks_path):
        queue = TaskQueue.load(tasks_path)
        write_status(queue, status_path, "Initialized -- edit tasks.json to begin")
        print(f"  Created {STATUS_FILE}")

    print(f"\nReady. Edit {TASKS_FILE} with your tasks, then run:")
    print(f"  bearing run {project_dir}")


def validate_tasks(project_dir: str) -> bool:
    """Validate tasks.json syntax and structure."""
    tasks_path = os.path.join(project_dir, TASKS_FILE)

    if not os.path.exists(tasks_path):
        print(f"Error: {TASKS_FILE} not found in {project_dir}")
        return False

    try:
        queue = TaskQueue.load(tasks_path)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {TASKS_FILE}: {e}")
        return False
    except Exception as e:
        print(f"Error: Could not parse {TASKS_FILE}: {e}")
        return False

    # Check for duplicate IDs
    ids = [t.id for t in queue.tasks]
    dupes = [i for i in ids if ids.count(i) > 1]
    if dupes:
        print(f"Error: Duplicate task IDs: {set(dupes)}")
        return False

    # Check dependencies reference valid task IDs
    for task in queue.tasks:
        for dep in task.depends_on:
            if dep not in ids:
                print(f"Error: Task {task.id} depends on unknown task '{dep}'")
                return False

    # Check for circular dependencies
    for task in queue.tasks:
        visited = set()
        stack = list(task.depends_on)
        while stack:
            dep_id = stack.pop()
            if dep_id == task.id:
                print(f"Error: Circular dependency involving {task.id}")
                return False
            if dep_id in visited:
                continue
            visited.add(dep_id)
            dep_task = next((t for t in queue.tasks if t.id == dep_id), None)
            if dep_task:
                stack.extend(dep_task.depends_on)

    print(f"Valid: {len(queue.tasks)} tasks, project '{queue.project}'")
    for task in queue.tasks:
        deps = f" (depends: {', '.join(task.depends_on)})" if task.depends_on else ""
        print(f"  {task.id}: {task.name}{deps}")
    return True


def propagate_context(queue: TaskQueue, completed_task: Task):
    """
    Auto-inject context and file relevance from a completed task
    into its dependents.

    Two things propagate:
    1. A text summary of what was done (appended to context field)
    2. The completed task's relevant_files (merged into dependent's
       relevant_files so the executor knows to focus on them)

    Won't duplicate on re-runs.
    """
    if not completed_task.result.summary:
        return

    summary_snippet = completed_task.result.summary[:300].replace("\n", " ").strip()
    context_entry = (
        f"[{completed_task.id}: {completed_task.name}] {summary_snippet}"
    )

    for task in queue.tasks:
        if completed_task.id in task.depends_on:
            # Propagate context text
            if completed_task.id not in task.context:
                if task.context:
                    task.context = f"{task.context}\n{context_entry}"
                else:
                    task.context = context_entry

            # Propagate file relevance
            if completed_task.relevant_files:
                existing = set(task.relevant_files)
                for f in completed_task.relevant_files:
                    if f not in existing:
                        task.relevant_files.append(f)


def start_planner(project_dir: str):
    """
    Launch an interactive AI session pre-loaded with Bearing knowledge.
    Asks which CLI and model to use, then starts the planner.
    """
    project_dir = os.path.abspath(project_dir)

    # Ask which CLI
    print("Which CLI for the planner?")
    print("  1. Claude Code (default)")
    print("  2. Codex")
    cli_choice = input("Choice [1]: ").strip()

    if cli_choice == "2":
        cli = "codex"
        models = [
            ("gpt-4.5", "GPT-4.5 (default)"),
            ("gpt-4o", "GPT-4o"),
            ("o3", "o3"),
        ]
    else:
        cli = "claude"
        models = [
            ("opus", "Opus (default - best for planning)"),
            ("sonnet", "Sonnet"),
            ("haiku", "Haiku"),
        ]

    if not check_cli_installed(cli):
        print(f"Error: '{cli}' not found in PATH.")
        sys.exit(1)

    # Ask which model
    print()
    print("Which model?")
    for i, (_, label) in enumerate(models, 1):
        print(f"  {i}. {label}")
    model_choice = input(f"Choice [1]: ").strip()

    try:
        model_idx = int(model_choice) - 1 if model_choice else 0
        model = models[model_idx][0]
    except (ValueError, IndexError):
        model = models[0][0]

    print()
    print(f"Starting {cli} ({model}) planner...")
    print()

    # Build the initial prompt — short, tells the agent to read
    # the full instructions from the planner_prompt.md file
    initial_parts = [
        f"Read {PLANNER_PROMPT_PATH} for your planning instructions.",
    ]

    status_path = os.path.join(project_dir, STATUS_FILE)
    tasks_path = os.path.join(project_dir, TASKS_FILE)

    if os.path.exists(status_path):
        initial_parts.append("Read status.md for previous results.")
    if os.path.exists(tasks_path):
        initial_parts.append("Read tasks.json for the current task queue.")

    initial_parts.append(
        "List the directory structure and read CLAUDE.md if it exists. "
        "Then ask what I want to work on."
    )

    initial_prompt = " ".join(initial_parts)

    # Build CLI command
    if cli == "claude":
        cmd = ["claude", "--model", model, initial_prompt]
    elif cli == "codex":
        cmd = ["codex", "--model", model, initial_prompt]
    else:
        cmd = [cli, initial_prompt]

    subprocess.run(cmd, cwd=project_dir)


def run_orchestrator(project_dir: str):
    """
    Main execution loop.

    Processes tasks sequentially, respecting dependencies and checkpoints.
    Writes status.md after each task completes.
    Pauses when a checkpoint requires human review.
    """
    project_dir = os.path.abspath(project_dir)
    tasks_path = os.path.join(project_dir, TASKS_FILE)
    status_path = os.path.join(project_dir, STATUS_FILE)

    if not check_cli_installed():
        print("Error: 'claude' CLI not found in PATH.")
        print("Install Claude Code: https://claude.ai/install.sh")
        sys.exit(1)

    if not os.path.exists(tasks_path):
        print(f"Error: {TASKS_FILE} not found in {project_dir}")
        print(f"Run: bearing init {project_dir}")
        sys.exit(1)

    queue = TaskQueue.load(tasks_path)
    print(f"Bearing -- {queue.project}")
    print(f"Tasks: {len(queue.tasks)}")
    print()

    write_status(queue, status_path, "Starting execution")

    while True:
        queue = TaskQueue.load(tasks_path)

        if queue.is_paused():
            print("\nPaused -- a task is awaiting review.")
            print(f"   Review {STATUS_FILE}, edit {TASKS_FILE}, then re-run.")
            write_status(queue, status_path, "Paused -- awaiting human review")
            break

        task = queue.next_task()
        if task is None:
            if queue.has_failures():
                print("\nStopped -- task failure. Check status.md.")
            else:
                print("\nAll tasks complete.")
            write_status(queue, status_path)
            break

        # Execute
        print(f">> Running: {task.id} -- {task.name}")
        print(f"   model={task.config.model} effort={task.config.effort} "
              f"budget=${task.config.budget_usd:.2f}")

        task.result.status = TaskStatus.RUNNING
        queue.save(tasks_path)
        write_status(queue, status_path, f"Running: {task.id}")

        result = run_task(task, project_dir)
        task.result = result

        if result.status == TaskStatus.COMPLETED:
            print(f"   OK -- ${result.cost_usd:.2f}, {result.turns_used} turns")

            # Auto-inject context into dependent tasks
            propagate_context(queue, task)

            if task.checkpoint == CheckpointLevel.PAUSE:
                task.result.status = TaskStatus.AWAITING_REVIEW
                print(f"   Checkpoint: pausing for review")
            elif task.checkpoint == CheckpointLevel.NOTIFY:
                print(f"   Checkpoint: review recommended (continuing)")

        elif result.status == TaskStatus.FAILED:
            print(f"   FAILED: {result.error[:100]}")

            if task.on_failure == FailurePolicy.RETRY_ONCE and task.result.retry_count == 0:
                print(f"   Retrying...")
                task.result.retry_count = 1
                task.result.status = TaskStatus.QUEUED

                original_prompt = task.prompt
                task.prompt = (
                    f"{original_prompt}\n\n"
                    f"PREVIOUS ATTEMPT FAILED:\n"
                    f"{result.error}\n\n"
                    f"Please fix the issue and try again."
                )

                queue.save(tasks_path)
                write_status(queue, status_path, f"Retrying: {task.id}")
                continue

            elif task.on_failure == FailurePolicy.SKIP:
                task.result.status = TaskStatus.SKIPPED
                print(f"   Skipping (failure policy)")

        queue.save(tasks_path)
        write_status(queue, status_path)

        time.sleep(2)


def show_status(project_dir: str):
    """Print current status to terminal."""
    tasks_path = os.path.join(project_dir, TASKS_FILE)
    if not os.path.exists(tasks_path):
        print(f"No {TASKS_FILE} found in {project_dir}")
        return

    queue = TaskQueue.load(tasks_path)

    print(f"Bearing -- {queue.project}")
    print()

    for task in queue.tasks:
        status_label = task.result.status.value.upper()
        cost = f" ${task.result.cost_usd:.2f}" if task.result.cost_usd > 0 else ""
        turns = f" {task.result.turns_used}t" if task.result.turns_used > 0 else ""
        print(f"  [{status_label}] {task.id}: {task.name}{cost}{turns}")

    total_cost = queue.total_cost
    if total_cost > 0:
        print(f"\nSession cost: ${total_cost:.2f}")


def show_summary(project_dir: str):
    """
    Compact summary designed for the planning session.

    When the planner asks "what's happening with execution?" -- this is
    the answer. Short enough to read in a second, detailed enough to
    know whether to keep planning or investigate.
    """
    tasks_path = os.path.join(project_dir, TASKS_FILE)
    if not os.path.exists(tasks_path):
        print(f"No {TASKS_FILE} found in {project_dir}")
        return

    queue = TaskQueue.load(tasks_path)

    total = len(queue.tasks)
    completed = [t for t in queue.tasks if t.result.status == TaskStatus.COMPLETED]
    failed = [t for t in queue.tasks if t.result.status == TaskStatus.FAILED]
    running = [t for t in queue.tasks if t.result.status == TaskStatus.RUNNING]
    paused = [t for t in queue.tasks if t.result.status == TaskStatus.AWAITING_REVIEW]

    total_cost = queue.total_cost
    cost_str = f", ${total_cost:.2f}" if total_cost > 0 else ""
    print(f"{queue.project}: {len(completed)}/{total} done{cost_str}")

    recently_done = [t for t in queue.tasks
                     if t.result.status in (TaskStatus.COMPLETED, TaskStatus.FAILED,
                                            TaskStatus.AWAITING_REVIEW)]
    if recently_done:
        last = recently_done[-1]
        label = {
            TaskStatus.COMPLETED: "OK",
            TaskStatus.AWAITING_REVIEW: "REVIEW",
            TaskStatus.FAILED: "FAIL",
        }.get(last.result.status, "?")
        print(f"  Last: [{label}] {last.id} -- {last.name}")
        if last.result.summary:
            snippet = last.result.summary[:200].replace("\n", " ")
            print(f"        {snippet}")
        if last.result.error:
            print(f"        ERROR: {last.result.error[:150]}")

    if running:
        for t in running:
            print(f"  Now:  {t.id} -- {t.name}")

    next_task = queue.next_task()
    if paused:
        print(f"  PAUSED -- awaiting review")
    elif failed:
        print(f"  STOPPED -- {failed[-1].id} failed")
    elif next_task:
        print(f"  Next: {next_task.id} -- {next_task.name}")
    elif not running:
        print(f"  All done")


def watch_status(project_dir: str, interval: float = 3.0):
    """
    Watch for task state changes and print updates as they happen.
    Ctrl+C to stop.
    """
    tasks_path = os.path.join(project_dir, TASKS_FILE)
    if not os.path.exists(tasks_path):
        print(f"No {TASKS_FILE} found in {project_dir}")
        return

    queue = TaskQueue.load(tasks_path)
    prev_states = {t.id: t.result.status for t in queue.tasks}

    print(f"Watching {queue.project}... (Ctrl+C to stop)")
    show_summary(project_dir)
    print()

    try:
        while True:
            time.sleep(interval)

            try:
                queue = TaskQueue.load(tasks_path)
            except Exception:
                continue

            current_states = {t.id: t.result.status for t in queue.tasks}

            for task_id, new_status in current_states.items():
                old_status = prev_states.get(task_id)
                if old_status != new_status:
                    task = next(t for t in queue.tasks if t.id == task_id)
                    now = time.strftime("%H:%M:%S")
                    label = new_status.value.upper()
                    line = f"[{now}] [{label}] {task_id}: {task.name}"

                    if new_status == TaskStatus.COMPLETED:
                        line += f" (${task.result.cost_usd:.2f})"
                    elif new_status == TaskStatus.FAILED:
                        line += f" -- {task.result.error[:80]}"

                    print(line)

            for task_id in current_states:
                if task_id not in prev_states:
                    task = next(t for t in queue.tasks if t.id == task_id)
                    now = time.strftime("%H:%M:%S")
                    print(f"[{now}] [ADDED] {task_id}: {task.name}")

            prev_states = current_states

            all_done = all(
                s in (TaskStatus.COMPLETED, TaskStatus.SKIPPED,
                      TaskStatus.FAILED, TaskStatus.AWAITING_REVIEW)
                for s in current_states.values()
            )
            if all_done and current_states:
                print()
                show_summary(project_dir)
                break

    except KeyboardInterrupt:
        print("\nStopped watching.")
        show_summary(project_dir)


def print_usage():
    print("Bearing -- Claude Code Task Orchestrator")
    print()
    print("Usage:")
    print("  bearing start <project_dir>     Start a planning session")
    print("  bearing init <project_dir>      Create starter files")
    print("  bearing validate <project_dir>   Check tasks.json")
    print("  bearing run <project_dir>        Execute queued tasks")
    print("  bearing status <project_dir>     Show full status")
    print("  bearing summary <project_dir>    Quick check (for planner)")
    print("  bearing watch <project_dir>      Live updates as tasks run")
    print()
    print("Workflow:")
    print("  1. cd your-project && bearing start .")
    print("  2. Discuss what you want to build with the planner")
    print("  3. Planner writes tasks.json for you")
    print("  4. !bearing run . (from the planner session)")
    print("  5. !bearing summary . (to check progress)")


def main():
    if len(sys.argv) < 2:
        print_usage()
        sys.exit(1)

    command = sys.argv[1]
    project_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    commands = {
        "start": start_planner,
        "init": init_project,
        "validate": validate_tasks,
        "run": run_orchestrator,
        "status": show_status,
        "summary": show_summary,
        "watch": watch_status,
    }

    if command in commands:
        commands[command](project_dir)
    else:
        print(f"Unknown command: {command}")
        print_usage()
        sys.exit(1)


if __name__ == "__main__":
    main()
