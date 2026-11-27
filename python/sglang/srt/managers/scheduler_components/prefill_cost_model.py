"""CPU-only Prefill cost contract; no device synchronization or global state.

Times are milliseconds. Coefficients must be calibrated for the execution
signature; these estimates never replace KV/workspace admission checks.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PrefillWork:
    rows: int
    prefix: int
    replay: int = 0

    def __post_init__(self):
        if min(self.rows, self.prefix, self.replay) < 0:
            raise ValueError("negative Prefill work")


@dataclass(frozen=True)
class PrefillCostModel:
    signature: str
    coefficients: tuple[float, float, float, float, float]

    def __post_init__(self):
        if not self.signature or len(self.coefficients) != 5:
            raise ValueError("execution signature and five coefficients required")
        if any(not math.isfinite(x) or x < 0 for x in self.coefficients):
            raise ValueError("cost coefficients must be finite and nonnegative")

    def predict_ms(self, work: Sequence[PrefillWork]) -> float:
        if not work:
            return 0.0
        rows = [w.rows + w.replay for w in work]
        features = (
            1,
            sum(rows),
            sum(q * q for q in rows),
            sum(q * w.prefix for q, w in zip(rows, work)),
            len(work),
        )
        return sum(c * f for c, f in zip(self.coefficients, features))

    def chunk_rows(
        self,
        admitted: Sequence[PrefillWork],
        *,
        prefix: int,
        replay: int,
        remaining: int,
        budget_ms: float,
        page_size: int,
        hard_cap: int = 8192,
    ) -> int:
        """Largest aligned candidate fitting the whole batch's time budget.

        Returns zero when nothing fits. The scheduler, not this estimator, owns
        the minimum-progress exception and all memory admission decisions.
        """
        if page_size <= 0 or hard_cap <= 0 or remaining < 0:
            raise ValueError("invalid row bounds")
        if not math.isfinite(budget_ms) or budget_ms < 0:
            raise ValueError("invalid time budget")
        PrefillWork(0, prefix, replay)
        cap = min(remaining, hard_cap, 8192)
        # A final partial page is legal; allocator admission rounds allocation.
        if (
            cap == remaining
            and cap > 0
            and self.predict_ms([*admitted, PrefillWork(cap, prefix, replay)])
            <= budget_ms
        ):
            return cap
        lo, hi = 0, cap // page_size
        while lo < hi:
            mid = (lo + hi + 1) // 2
            cost = self.predict_ms(
                [*admitted, PrefillWork(mid * page_size, prefix, replay)]
            )
            if cost <= budget_ms:
                lo = mid
            else:
                hi = mid - 1
        return lo * page_size


def finish_quote_ms(
    *,
    remaining_ms: float,
    queued_ms: float,
    dispatched_ms: float,
    compute_ms: float,
    reload_ms: float,
) -> float:
    costs = (remaining_ms, queued_ms, dispatched_ms, compute_ms, reload_ms)
    if any(not math.isfinite(x) or x < 0 for x in costs):
        raise ValueError("invalid quote component")
    return sum(costs)
