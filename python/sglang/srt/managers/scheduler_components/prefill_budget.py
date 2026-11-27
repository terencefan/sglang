"""Pure CPU budget decisions shared by the scheduler and its DP metadata vote."""

import math
from dataclasses import dataclass

from sglang.srt.managers.scheduler_components.prefill_cost_model import (
    PrefillCostModel,
    PrefillWork,
)


def consensus_budget(votes):
    """All configured ranks must have the same model and positive proposal.

    Zero is an unready rank, not an idle rank. Idle ranks still vote readiness.
    Missing metadata always disables control for the following step.
    """
    if not votes or any(len(v) != 2 or v[0] <= 0 or v[1] <= 0 for v in votes):
        return 0
    if len({v[0] for v in votes}) != 1:
        return 0
    return min(v[1] for v in votes)


@dataclass
class PrefillBudget:
    mode: str = "off"
    model: PrefillCostModel | None = None
    target_ms: float = 0
    group_budget_us: int = 0

    def __post_init__(self):
        if self.mode not in ("off", "shadow", "enforce"):
            raise ValueError("invalid Prefill budget mode")
        if not math.isfinite(self.target_ms) or self.target_ms < 0:
            raise ValueError("invalid Prefill time target")

    def vote(self):
        if self.mode != "enforce" or self.model is None or self.target_ms <= 0:
            return (0, 0)
        return self.model.fingerprint, max(1, int(self.target_ms * 1000))

    def receive(self, votes):
        self.group_budget_us = consensus_budget(votes)

    def limit(self, admitted, *, prefix, replay, remaining, page_size, alignment=None):
        if self.mode != "enforce" or self.model is None or self.group_budget_us <= 0:
            return None
        align = math.lcm(page_size, alignment or page_size)
        occupied = sum(w.rows + w.replay for w in admitted)
        cap = max(0, 8192 - occupied - replay)
        if cap == 0:
            return 0
        rows = self.model.chunk_rows(
            admitted,
            prefix=prefix,
            replay=replay,
            remaining=remaining,
            budget_ms=self.group_budget_us / 1000,
            page_size=align,
            hard_cap=cap,
        )
        if rows == 0 and not admitted:
            # Soft time budget may yield to progress; memory admission never does.
            return min(remaining, align) if cap >= min(remaining, align) else 0
        return rows


def admitted_work(adder):
    return [
        PrefillWork(
            int(r.extend_len),
            int(r.prefix_len),
            int(adder._replay_len(r, r.prefix_len)),
        )
        for r in adder.can_run_list
    ]


def candidate_limit(adder, *, prefix, replay, remaining, alignment=None):
    budget = getattr(adder, "prefill_time_budget", None)
    if budget is None:
        return None
    return budget.limit(
        admitted_work(adder),
        prefix=prefix,
        replay=replay,
        remaining=remaining,
        page_size=adder.page_size,
        alignment=alignment,
    )
