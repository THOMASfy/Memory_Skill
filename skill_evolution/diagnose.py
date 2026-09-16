"""Gap scores from geometry and timestamp statistics — not an LLM rubric."""

from __future__ import annotations

import numpy as np

from .mathutil import effective_n, kmeans_two, softmax, weighted_std
from .types import EvolutionState, GapReport, Hit

# Kish N_eff gate. Below this, refuse to open a new operator.
MIN_NEFF = 5.0
SIM_FLOOR = 0.12
CV_REF = 0.35
ALIAS_SEP_REF = 0.25


def _timestamped(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if h.doc.time_days is not None]


def time_gap_score(hits: list[Hit], state: EvolutionState) -> tuple[float, float]:
    """G_time = (1-λ) · sigmoid(CV/CV_ref) if N_eff is enough, else 0.

    Time is unused by the baseline ranker (λ=0). High timestamp spread among
    near neighbours means dense sim cannot tell stale from current.
    """
    dated = _timestamped(hits)
    if len(dated) < 2:
        return 0.0, 0.0
    sims = np.array([h.sim for h in dated], dtype=np.float64)
    times = np.array([float(h.doc.time_days) for h in dated], dtype=np.float64)
    w = softmax(sims, temperature=1.0)
    n_eff = effective_n(w)
    if n_eff < MIN_NEFF:
        return 0.0, n_eff
    mu = float(np.sum(w * times))
    sd = weighted_std(times, w, mean=mu)
    # Ordinal days are ~7e5; use year-scale spread, not sd/|μ|.
    spread_years = sd / 365.25
    unused = 1.0 - float(state.lambda_recency)
    g = unused * (1.0 / (1.0 + np.exp(-(spread_years / max(CV_REF, 1e-6) - 1.0) * 4.0)))
    return float(np.clip(g, 0.0, 1.0)), float(n_eff)


def alias_gap_score(hits: list[Hit], rng: np.random.Generator) -> tuple[float, float]:
    """Two-means gap: both centroids close to the query cloud but far from each other."""
    if len(hits) < MIN_NEFF:
        return 0.0, float(len(hits))
    x = np.stack([h.doc.embedding for h in hits], axis=0)
    sims = np.array([h.sim for h in hits], dtype=np.float64)
    n_eff = effective_n(softmax(sims, temperature=1.0))
    if n_eff < MIN_NEFF:
        return 0.0, n_eff
    labels, centers = kmeans_two(x, rng)
    if labels.min() == labels.max():
        return 0.0, n_eff
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    mean_sim = [float(sims[labels == k].mean()) if np.any(labels == k) else 0.0 for k in (0, 1)]
    both_relevant = min(mean_sim) / max(max(mean_sim), 1e-8)
    g = both_relevant * (1.0 / (1.0 + np.exp(-(sep / ALIAS_SEP_REF - 1.0) * 4.0)))
    return float(np.clip(g, 0.0, 1.0)), float(n_eff)


def diagnose(hits: list[Hit], state: EvolutionState, rng: np.random.Generator) -> GapReport:
    if not hits:
        return GapReport(
            kind="unanswerable",
            sufficient=False,
            g_time=0.0,
            g_alias=0.0,
            g_unanswerable=1.0,
            n_eff_time=0.0,
            n_eff_alias=0.0,
            reason="empty retrieval",
        )
    g_unans = float(max(0.0, (SIM_FLOOR - hits[0].sim) / SIM_FLOOR))
    g_time, n_time = time_gap_score(hits, state)
    g_alias, n_alias = alias_gap_score(hits, rng)

    if g_unans >= 0.8:
        return GapReport("unanswerable", False, g_time, g_alias, g_unans, n_time, n_alias, "max sim below floor")

    # Recency fusion already pulled a current document to the top.
    if state.rank_slot_active and state.t_star is not None and hits[0].doc.time_days is not None:
        if hits[0].doc.time_days >= state.t_star - (state.sigma or 1.0):
            return GapReport(
                "sufficient",
                True,
                g_time,
                g_alias,
                g_unans,
                n_time,
                n_alias,
                "top hit is on the current side of t*",
            )

    if state.query_slot_active and state.pos_ids:
        top3 = [h.doc.id for h in hits[:3]]
        if top3 and all(i in set(state.pos_ids) for i in top3):
            return GapReport(
                "sufficient",
                True,
                g_time,
                g_alias,
                g_unans,
                n_time,
                n_alias,
                "top-3 lie in Rocchio M+",
            )

    scores = {"time": g_time, "alias": g_alias, "unanswerable": g_unans}
    kind = max(scores, key=scores.get)  # type: ignore[arg-type]
    if scores[kind] < 0.2:
        if hits[0].sim >= SIM_FLOOR and g_time < 0.2 and g_alias < 0.2:
            # High sim, no unused meridian → treat as enough for this loop.
            return GapReport("sufficient", True, g_time, g_alias, g_unans, n_time, n_alias, "no unused meridian")
        return GapReport("unknown", False, g_time, g_alias, g_unans, n_time, n_alias, "gap below open threshold")
    return GapReport(kind, False, g_time, g_alias, g_unans, n_time, n_alias, f"argmax gap={kind}")  # type: ignore[arg-type]
