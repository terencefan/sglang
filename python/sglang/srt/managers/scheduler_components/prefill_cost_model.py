"""CPU-only Prefill cost contract; no device synchronization or global state.

Times are milliseconds. Coefficients must be calibrated for the execution
signature; these estimates never replace KV/workspace admission checks.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property


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
    coefficients: tuple[float, ...]
    dp_degree: int = 1

    def __post_init__(self):
        if not self.signature or len(self.coefficients) not in (5, 7):
            raise ValueError(
                "execution signature and five or seven coefficients required"
            )
        if not isinstance(self.dp_degree, int) or not 1 <= self.dp_degree <= 64:
            raise ValueError("invalid attention DP degree")
        if any(not math.isfinite(x) or x < 0 for x in self.coefficients):
            raise ValueError("cost coefficients must be finite and nonnegative")

    @property
    def shared(self):
        return len(self.coefficients) == 7

    @property
    def feature_contract(self):
        return "shared-dp-v2" if self.shared else "local-v1"

    @cached_property
    def fingerprint(self):
        key = json.dumps(
            [self.signature, self.coefficients, self.dp_degree, self.feature_contract],
            sort_keys=True,
        )
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:7], "big") or 1

    def predict_ms(
        self, work: Sequence[PrefillWork], *, global_num_tokens=None
    ) -> float:
        if not work and (not self.shared or not global_num_tokens):
            return 0.0
        rows = [w.rows + w.replay for w in work]
        features = (
            1,
            sum(rows),
            sum(q * q for q in rows),
            sum(q * w.prefix for q, w in zip(rows, work)),
            len(work),
        )
        if self.shared:
            if global_num_tokens is None:
                # Admission/queue estimates assume saturated peers of the same
                # size. This is a soft scheduling estimate, not a wall-time bound.
                shared = [sum(rows)] * self.dp_degree
            else:
                shared = list(global_num_tokens)
                if len(shared) != self.dp_degree or any(
                    not isinstance(q, int) or q < 0 for q in shared
                ):
                    raise ValueError("invalid shared DP token shape")
            features = (*features, sum(shared), max(shared))
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


def remaining_cost_ms(model, work, chunk=8192):
    """Sum fixed-chunk costs analytically, without walking every future chunk.

    A per-request upper-style quote: mixed batching can amortize the constant
    term. Reload costs are intentionally not invented here.
    """
    if chunk <= 0:
        raise ValueError("positive quote chunk required")
    full, tail = divmod(work.rows, chunk)
    c0, c1, c2, c3, c4 = model.coefficients[:5]
    if model.shared:
        c1 += model.dp_degree * model.coefficients[5] + model.coefficients[6]
    prefix_sum = full * work.prefix + chunk * full * (full - 1) // 2
    total = full * (c0 + c1 * chunk + c2 * chunk * chunk + c4) + c3 * chunk * prefix_sum
    if tail:
        total += model.predict_ms([PrefillWork(tail, work.prefix + full * chunk)])
    if work.replay:
        # Replay is charged once at the initial cached boundary.
        total += model.predict_ms([PrefillWork(work.replay, work.prefix)])
    return total
