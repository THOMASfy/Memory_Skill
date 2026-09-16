"""Linear fusion of dense sim, recency kernel, and negative-id penalty."""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from .signals import recency_kernel
from .types import EvolutionState, Hit, Signals

LAMBDA_MAX = 0.60
LAMBDA_TIME_FLOOR = 0.40
ETA = 0.35
ETA_HARD = 0.50
ETA_TIME = 0.30
TRUST = 0.35
DELTA_DIV = 0.12


def estimate_lambda(hits: list[Hit], signals: Signals) -> float:
    """Fisher-style closed form: how much recency separates M+ vs M− relative to sim."""
    if not signals.pos_ids or signals.t_star is None or signals.sigma is None:
        return 0.0
    pos = [h for h in hits if h.doc.id in set(signals.pos_ids)]
    neg = [h for h in hits if h.doc.id in set(signals.neg_ids)]
    if not pos:
        return 0.0

    def mean_k(group: list[Hit]) -> float:
        if not group:
            return 0.0
        vals = [
            recency_kernel(float(h.doc.time_days), signals.t_star, signals.sigma)  # type: ignore[arg-type]
            for h in group
            if h.doc.time_days is not None
        ]
        return float(np.mean(vals)) if vals else 0.0

    dk = abs(mean_k(pos) - mean_k(neg))
    ds = abs(float(np.mean([h.sim for h in pos])) - (float(np.mean([h.sim for h in neg])) if neg else 0.0))
    lam = dk / (dk + ds + 1e-8)
    return float(np.clip(lam, 0.0, LAMBDA_MAX))


def apply_rank_update(
    state: EvolutionState,
    hits: list[Hit],
    signals: Signals,
    *,
    boost_recency: bool = False,
    hard_neg_ids: tuple[str, ...] = (),
    allow_recency: bool | None = None,
) -> EvolutionState:
    """allow_recency=False: keep λ=0 (non-temporal questions). None: legacy geometric path."""
    if allow_recency is False:
        lam = 0.0
    else:
        lam = estimate_lambda(hits, signals)
        if boost_recency or allow_recency is True:
            lam = max(lam, LAMBDA_TIME_FLOOR if boost_recency else 0.35)
        prev = state.lambda_recency
        lam = float(np.clip(lam, prev - TRUST, prev + TRUST))
        if boost_recency:
            lam = float(np.clip(max(lam, LAMBDA_TIME_FLOOR), 0.0, LAMBDA_MAX))
    eta = ETA if signals.neg_ids or state.neg_ids else 0.0
    hard = tuple(dict.fromkeys(list(hard_neg_ids) + list(state.hard_neg_ids)))
    return replace(
        state,
        lambda_recency=lam,
        eta_neg=eta,
        eta_hard=ETA_HARD if hard else state.eta_hard,
        t_star=signals.t_star if signals.t_star is not None else state.t_star,
        sigma=signals.sigma if signals.sigma is not None else state.sigma,
        pos_ids=signals.pos_ids or state.pos_ids,
        neg_ids=signals.neg_ids or state.neg_ids,
        hard_neg_ids=hard,
        rank_slot_active=True,
    )


def fused_score(sim: float, doc_id: str, time_days: float | None, state: EvolutionState) -> float:
    """score = (1-λ) sim + λ κ(t) − η_soft 1[M−] − η_hard 1[M_hard−]."""
    rec = 0.0
    if state.lambda_recency > 0 and time_days is not None and state.t_star is not None and state.sigma:
        rec = recency_kernel(time_days, state.t_star, state.sigma)
    if doc_id in set(state.hard_neg_ids):
        penalty = state.eta_hard or ETA_HARD
    elif doc_id in set(state.neg_ids):
        penalty = state.eta_neg
    else:
        penalty = 0.0
    bonus = 0.0
    if doc_id in set(getattr(state, "div_ids", ()) or ()):
        bonus = float(getattr(state, "delta_div", 0.0) or 0.0)
    return float((1.0 - state.lambda_recency) * sim + state.lambda_recency * rec - penalty + bonus)


def apply_multihop_rank(
    state: EvolutionState,
    hits: list[Hit],
    rng: np.random.Generator,
    overlap_fn,
) -> EvolutionState:
    """Keep complementary clusters; penalize only zero-overlap noise. Do not treat hop-2 as M−."""
    if len(hits) < 4:
        noise = tuple(h.doc.id for h in hits if overlap_fn(h) < 1e-6)
        return replace(
            state,
            lambda_recency=0.0,
            eta_neg=0.25 if noise else 0.0,
            neg_ids=noise,
            div_ids=(),
            delta_div=0.0,
            rank_slot_active=True,
            rank_task="multihop",
        )
    from .mathutil import kmeans_two

    x = np.stack([h.doc.embedding for h in hits], axis=0)
    labels, _centers = kmeans_two(x, rng)
    groups: dict[int, list[Hit]] = {0: [], 1: []}
    for h, lab in zip(hits, labels):
        groups.setdefault(int(lab), []).append(h)
    scored = []
    for k, grp in groups.items():
        if not grp:
            continue
        ov = float(np.mean([overlap_fn(h) for h in grp]))
        sim = float(np.mean([h.sim for h in grp]))
        scored.append((k, ov, sim, grp))
    scored.sort(key=lambda t: (t[1], t[2]), reverse=True)
    primary = scored[0][3] if scored else []
    div: list[Hit] = []
    for _k, ov, _sim, grp in scored[1:]:
        if ov > 1e-6:
            div.extend(grp)
    noise = tuple(h.doc.id for h in hits if overlap_fn(h) < 1e-6)
    pos = tuple(h.doc.id for h in primary)
    div_ids = tuple(h.doc.id for h in div if h.doc.id not in set(pos))
    return replace(
        state,
        lambda_recency=0.0,
        eta_neg=0.25 if noise else 0.0,
        eta_hard=0.0,
        pos_ids=pos or state.pos_ids,
        neg_ids=noise,
        hard_neg_ids=(),
        div_ids=div_ids,
        delta_div=DELTA_DIV if div_ids else 0.0,
        rank_slot_active=True,
        rank_task="multihop",
    )
