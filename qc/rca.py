"""Bounded RCA investigation loop (ported from Verity's spine).

Runs an allowlisted sequence of evidence queries over a completed run and
records every call with its row count and truncation flag. A selector may
choose the next tool (for example a decision provider or an LLM), but it can
only pick from the allowlist; the loop is bounded and any finding is
``confirmed=False`` until an analyst confirms it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from .agent import CAUSE_TOOL_PLANS
from .evidence_query import ALLOWED_QUERIES, query_evidence

Selector = Callable[[Any, list["RCAStep"], tuple[str, ...]], tuple[str, dict] | None]


@dataclass
class RCAStep:
    step: int
    tool: str
    args: dict[str, Any]
    total_rows: int
    truncated: bool
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RCAResult:
    run_id: str
    steps: list[RCAStep]
    stop_reason: str
    confirmed: bool = False
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "steps": [step.to_dict() for step in self.steps],
            "stop_reason": self.stop_reason,
            "confirmed": self.confirmed,
            "summary": self.summary,
        }


def default_selector(
    result: Any, trace: list[RCAStep], remaining: tuple[str, ...]
) -> tuple[str, dict] | None:
    decisions = getattr(result, "decisions", None)
    cause = "UNKNOWN"
    if decisions is not None:
        decision = decisions.get("likely_cause")
        if decision is not None:
            cause = str(decision.value)
    for tool, args in CAUSE_TOOL_PLANS.get(cause, CAUSE_TOOL_PLANS["UNKNOWN"]):
        if tool in remaining:
            return tool, dict(args)
    return None


def investigate(
    result: Any,
    selector: Selector | None = None,
    max_steps: int = 4,
    limit: int = 50,
) -> RCAResult:
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")
    selector = selector or default_selector
    trace: list[RCAStep] = []
    remaining = tuple(ALLOWED_QUERIES)

    while len(trace) < max_steps and remaining:
        choice = selector(result, trace, remaining)
        if choice is None:
            return RCAResult(
                run_id=str(getattr(result, "run_id", "")),
                steps=trace,
                stop_reason="selector_done",
                summary=f"{len(trace)} evidence queries; awaiting analyst confirmation",
            )
        tool, args = choice
        if tool not in remaining:
            raise ValueError(f"selector chose unavailable or repeated tool {tool!r}")
        outcome = query_evidence(result, tool, limit=limit, **args)
        trace.append(
            RCAStep(
                step=len(trace) + 1,
                tool=tool,
                args=dict(args),
                total_rows=outcome.total_rows,
                truncated=outcome.truncated,
                rows=outcome.rows,
            )
        )
        remaining = tuple(name for name in remaining if name != tool)

    stop_reason = "completed" if not remaining else "max_steps"
    return RCAResult(
        run_id=str(getattr(result, "run_id", "")),
        steps=trace,
        stop_reason=stop_reason,
        summary=f"{len(trace)} evidence queries; awaiting analyst confirmation",
    )
