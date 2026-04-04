"""
Bearing - Task schema definitions

Defines the structure for tasks that flow between the planning layer
and execution layer. Tasks are stored as JSON, read/written as files.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import json


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_REVIEW = "awaiting_review"
    SKIPPED = "skipped"


class CheckpointLevel(str, Enum):
    """Controls when the orchestrator pauses for human review."""
    AUTO = "auto"              # Continue automatically on success
    NOTIFY = "notify"          # Log prominently but continue
    PAUSE = "pause"            # Stop and wait for human to resume


class FailurePolicy(str, Enum):
    RETRY_ONCE = "retry_once"
    PAUSE = "pause"
    SKIP = "skip"


@dataclass
class ExecutionConfig:
    """Per-task Claude Code configuration flags."""
    model: str = "sonnet"
    effort: str = "high"          # low | medium | high | max
    budget_usd: float = 3.00      # token ceiling (works on subscriptions too)
    max_turns: int = 20
    permission_mode: str = "auto"  # auto | default
    worktree: Optional[str] = None # git worktree name, None = use current branch
    fast_mode: bool = False


@dataclass
class TaskResult:
    """What comes back from a Claude Code execution."""
    status: TaskStatus = TaskStatus.QUEUED
    cost_usd: float = 0.0
    turns_used: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0     # cheap reads (0.1x cost) — subset of input_tokens
    summary: str = ""
    error: str = ""
    retry_count: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def fresh_input_tokens(self) -> int:
        """Input tokens excluding cheap cache reads."""
        return self.input_tokens - self.cache_read_tokens


@dataclass
class Task:
    """A single unit of work for Claude Code to execute."""
    id: str
    name: str
    prompt: str
    config: ExecutionConfig = field(default_factory=ExecutionConfig)
    depends_on: list[str] = field(default_factory=list)
    checkpoint: CheckpointLevel = CheckpointLevel.AUTO
    on_failure: FailurePolicy = FailurePolicy.PAUSE
    context: str = ""           # Relevant info from previous tasks
    result: TaskResult = field(default_factory=TaskResult)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = asdict(self.config)
        d["result"] = asdict(self.result)
        d["checkpoint"] = self.checkpoint.value
        d["on_failure"] = self.on_failure.value
        d["result"]["status"] = self.result.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        config_data = d.get("config", {})
        # Filter out unknown fields for forward compatibility
        valid_config_fields = {f.name for f in ExecutionConfig.__dataclass_fields__.values()}
        config = ExecutionConfig(**{k: v for k, v in config_data.items() if k in valid_config_fields})

        result_data = d.get("result", {})
        result_data["status"] = TaskStatus(result_data.get("status", "queued"))
        # Filter out unknown fields for forward compatibility
        valid_result_fields = {f.name for f in TaskResult.__dataclass_fields__.values()}
        result = TaskResult(**{k: v for k, v in result_data.items() if k in valid_result_fields})

        return cls(
            id=d["id"],
            name=d["name"],
            prompt=d["prompt"],
            config=config,
            depends_on=d.get("depends_on", []),
            checkpoint=CheckpointLevel(d.get("checkpoint", "auto")),
            on_failure=FailurePolicy(d.get("on_failure", "pause")),
            context=d.get("context", ""),
            result=result,
        )


@dataclass
class TaskQueue:
    """The full task file. Contains project info + ordered task list."""
    project: str
    description: str = ""
    tasks: list[Task] = field(default_factory=list)

    @property
    def total_tokens_used(self) -> int:
        return sum(t.result.total_tokens for t in self.tasks)

    @property
    def total_cost(self) -> float:
        return sum(t.result.cost_usd for t in self.tasks)

    def save(self, path: str):
        data = {
            "project": self.project,
            "description": self.description,
            "tasks": [t.to_dict() for t in self.tasks],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "TaskQueue":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tasks = [Task.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            project=data["project"],
            description=data.get("description", ""),
            tasks=tasks,
        )

    def next_task(self) -> Optional[Task]:
        """Returns the next queued task whose dependencies are met."""
        completed_ids = {
            t.id for t in self.tasks
            if t.result.status == TaskStatus.COMPLETED
        }
        for task in self.tasks:
            if task.result.status != TaskStatus.QUEUED:
                continue
            deps_met = all(dep in completed_ids for dep in task.depends_on)
            if deps_met:
                return task
        return None

    def has_failures(self) -> bool:
        return any(
            t.result.status == TaskStatus.FAILED for t in self.tasks
        )

    def is_paused(self) -> bool:
        return any(
            t.result.status == TaskStatus.AWAITING_REVIEW for t in self.tasks
        )
