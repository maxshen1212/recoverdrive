"""Exact (Clopper-Pearson) confidence bounds on violation rates, counted per fault episode.

Frames inside one episode are strongly dependent, so the unit is the episode: an episode counts as violated when any
of its frames violates the requirement. The bound holds for the tested fault distribution only.
"""

from __future__ import annotations

from collections import defaultdict

from scipy.stats import beta


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Two-sided exact binomial interval for k successes in n trials."""
    alpha = 1 - conf
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def episode_violation_bounds(rows: list[dict], conf: float = 0.95) -> list[dict]:
    """rows: one dict per fault frame with keys run, fault_id, episode_id, violated. One output row per fault type:
    fault_id, episodes, violated, rate, upper (the upper end of the two-sided interval)."""
    episodes: dict[str, dict[tuple, bool]] = defaultdict(dict)
    for r in rows:
        key = (r["run"], r["episode_id"])
        episodes[r["fault_id"]][key] = episodes[r["fault_id"]].get(key, False) or bool(r["violated"])
    out = []
    for fault_id in sorted(episodes):
        flags = list(episodes[fault_id].values())
        n, k = len(flags), sum(flags)
        out.append({"fault_id": fault_id, "episodes": n, "violated": k, "rate": k / n,
                    "upper": clopper_pearson(k, n, conf)[1]})
    return out
