"""Estimate Rocchio centroids, t*, σ, M+/M− from hits. Never store passage text."""

from __future__ import annotations

import numpy as np

from .mathutil import EPS, effective_n, kmeans_two, softmax, weighted_std
from .types import GapKind, Hit, Signals

SIGMA_FLOOR_DAYS = 30.0


def _dated(hits: list[Hit]) -> list[Hit]:
    return [h for h in hits if h.doc.time_days is not None]


def recency_kernel(t: float, t_star: float, sigma: float) -> float:
    """One-sided: documents older than t* decay; newer are not penalized."""
    age = max(0.0, t_star - t)
    return float(np.exp(-age / max(sigma, EPS)))


def estimate_signals(
    hits: list[Hit],
    kind: GapKind,
    rng: np.random.Generator,
    query_time_days: float | None = None,
) -> Signals:
    if kind == "time":
        return _time_signals(hits, query_time_days=query_time_days)
    if kind == "alias":
        return _alias_signals(hits, rng)
    return Signals(None, None, (), (), None, None, 0.0, {})


def _time_signals(hits: list[Hit], query_time_days: float | None = None) -> Signals:
    dated = _dated(hits)
    if len(dated) < 2:
        return Signals(None, None, (), (), None, None, 0.0, {})
    sims = np.array([h.sim for h in dated], dtype=np.float64)
    times = np.array([float(h.doc.time_days) for h in dated], dtype=np.float64)
    w = softmax(sims)
    n_eff = effective_n(softmax(sims, temperature=1.0))
    # Question timestamp (LongMemEval) is the as-of time; otherwise 90th percentile.
    t_star = float(query_time_days) if query_time_days is not None else float(np.quantile(times, 0.9))
    sigma = max(weighted_std(times, w, mean=float(np.sum(w * times))), SIGMA_FLOOR_DAYS)
    kappa = {h.doc.id: recency_kernel(float(h.doc.time_days), t_star, sigma) for h in dated}

    pos, neg = [], []
    for h in dated:
        k = kappa[h.doc.id]
        # High sim but old → negative anchor. Recent → positive source.
        if h.doc.time_days is not None and h.doc.time_days < t_star - sigma and h.sim > 0:
            neg.append(h)
        elif k >= np.exp(-1.0):
            pos.append(h)

    pos_mean = _mean_emb(pos)
    neg_mean = _mean_emb(neg)
    return Signals(
        t_star=t_star,
        sigma=sigma,
        pos_ids=tuple(h.doc.id for h in pos),
        neg_ids=tuple(h.doc.id for h in neg),
        pos_mean=pos_mean,
        neg_mean=neg_mean,
        n_eff=n_eff,
        kappa=kappa,
    )


def _alias_signals(hits: list[Hit], rng: np.random.Generator) -> Signals:
    x = np.stack([h.doc.embedding for h in hits], axis=0)
    sims = np.array([h.sim for h in hits], dtype=np.float64)
    labels, centers = kmeans_two(x, rng)
    # Keep the tighter / closer-to-query cluster as M+.
    q_proxy = (x * sims[:, None]).sum(axis=0) / max(float(sims.sum()), EPS)
    d0 = float(np.linalg.norm(centers[0] - q_proxy))
    d1 = float(np.linalg.norm(centers[1] - q_proxy))
    pos_k = 0 if d0 <= d1 else 1
    pos = [h for h, lab in zip(hits, labels) if lab == pos_k]
    neg = [h for h, lab in zip(hits, labels) if lab != pos_k]
    return Signals(
        t_star=None,
        sigma=None,
        pos_ids=tuple(h.doc.id for h in pos),
        neg_ids=tuple(h.doc.id for h in neg),
        pos_mean=_mean_emb(pos),
        neg_mean=_mean_emb(neg),
        n_eff=effective_n(softmax(sims)),
        kappa={},
    )


def _mean_emb(hits: list[Hit]) -> np.ndarray | None:
    if not hits:
        return None
    w = softmax(np.array([h.sim for h in hits], dtype=np.float64))
    x = np.stack([h.doc.embedding for h in hits], axis=0)
    return (w[:, None] * x).sum(axis=0)
