"""Rocchio in embedding space. This is the query-slot update — not string concat."""

from __future__ import annotations

import numpy as np

from dataclasses import replace

from .mathutil import l2_normalize
from .types import EvolutionState, Signals

# SMART-style defaults; β/γ only turn on when centroids exist.
ALPHA = 1.0
BETA = 0.75
GAMMA = 0.15


def rocchio_query(state: EvolutionState, signals: Signals) -> EvolutionState:
    q = state.q0.astype(np.float64)
    beta, gamma = 0.0, 0.0
    acc = ALPHA * q
    if signals.pos_mean is not None:
        beta = BETA
        acc = acc + beta * signals.pos_mean
    if signals.neg_mean is not None:
        gamma = GAMMA
        acc = acc - gamma * signals.neg_mean
    q_new = l2_normalize(acc, axis=0)
    return replace(
        state,
        q=q_new.astype(np.float32),
        alpha=ALPHA,
        beta=beta,
        gamma=gamma,
        pos_ids=signals.pos_ids,
        neg_ids=signals.neg_ids,
        query_slot_active=True,
        t_star=signals.t_star if signals.t_star is not None else state.t_star,
        sigma=signals.sigma if signals.sigma is not None else state.sigma,
    )
