from __future__ import annotations

import numpy as np

from .filter_update import filter_penalty
from .mathutil import l2_normalize
from .rank_update import fused_score
from .types import Document, EvolutionState, Hit


NOISE_SIM_HI = 0.40
Q0_SIM_LO = 0.50


def dense_retrieve(
    docs: list[Document],
    state: EvolutionState,
    top_k: int = 10,
) -> list[Hit]:
    q0 = l2_normalize(np.asarray(state.q0, dtype=np.float64), axis=0)
    q = l2_normalize(np.asarray(state.q, dtype=np.float64), axis=0)
    omega = float(getattr(state, "omega_hop", 0.0) or 0.0)
    v_slot = getattr(state, "v_slot", None)
    mu_n = getattr(state, "mu_noise", None)
    vs = None
    if v_slot is not None and omega > 0:
        vs = l2_normalize(np.asarray(v_slot, dtype=np.float64), axis=0)
    mu = None
    if mu_n is not None and float(state.gamma or 0) > 0:
        mu = l2_normalize(np.asarray(mu_n, dtype=np.float64), axis=0)
    gamma = float(state.gamma or 0)
    hits: list[Hit] = []
    for doc in docs:
        v = l2_normalize(np.asarray(doc.embedding, dtype=np.float64), axis=0)
        s0 = float(np.dot(q0, v))
        sq = float(np.dot(q, v))
        if vs is not None:
            sim = max(s0, omega * float(np.dot(vs, v)))
        else:
            sim = sq
        if mu is not None:
            sn = float(np.dot(mu, v))
            if sn > NOISE_SIM_HI and s0 < Q0_SIM_LO:
                sim = sim - gamma * sn
        score = fused_score(sim, doc.id, doc.time_days, state)
        score = score - filter_penalty(doc.id, doc.text or "", state)
        hits.append(Hit(doc=doc, sim=sim, score=score))
    hits.sort(key=lambda h: h.score, reverse=True)
    return hits[: min(top_k, len(hits))]
