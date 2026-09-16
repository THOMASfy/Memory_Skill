"""Compile a MemoryReview into EvolutionState (one slot per round)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np

from .mathutil import l2_normalize, orthogonal_noise
from .filter_update import apply_filter_update
from .memory_review import MemoryReview, hop_phrase, is_abstract_missing
from .query_update import rocchio_query
from .rank_update import apply_multihop_rank, apply_rank_update
from .signals import estimate_signals
from .types import EvolutionState, Hit, Signals, SlotName

ALPHA_Q0 = 1.0
BETA_MISS = 0.0
GAMMA_NOISE = 0.15
DEAD_DELTA = 0.15
OMEGA_HOP = 0.85


def apply_memory_skill(
    state: EvolutionState,
    hits: list[Hit],
    review: MemoryReview,
    *,
    slot: SlotName,
    encode_query: Callable[[str], np.ndarray] | None,
    question: str,
    rng: np.random.Generator,
    query_time_days: float | None,
    dead_end: bool = False,
    rank_task: str = "",
) -> EvolutionState:
    if slot == "query":
        return _query_from_memory(
            state, hits, review, encode_query, question, rng, query_time_days, dead_end
        )
    if slot == "filter":
        return apply_filter_update(state, hits, review, question)
    return _rank_from_memory(state, hits, review, rng, query_time_days, question, rank_task)


def _signals_from_memory(
    hits: list[Hit],
    review: MemoryReview,
    rng: np.random.Generator,
    query_time_days: float | None,
) -> Signals:
    pos = [h for h in hits if h.doc.id in set(review.relevant_ids)]
    neg_ids = tuple(dict.fromkeys(list(review.soft_neg_ids) + list(review.hard_neg_ids)))
    neg = [h for h in hits if h.doc.id in set(neg_ids)]
    if pos or neg:
        pos_mean = (
            np.mean(np.stack([h.doc.embedding for h in pos], axis=0), axis=0) if pos else None
        )
        neg_mean = (
            np.mean(np.stack([h.doc.embedding for h in neg], axis=0), axis=0) if neg else None
        )
        return Signals(
            t_star=query_time_days,
            sigma=30.0 if query_time_days is not None else None,
            pos_ids=tuple(h.doc.id for h in pos),
            neg_ids=neg_ids,
            pos_mean=pos_mean,
            neg_mean=neg_mean,
            n_eff=float(len(hits)),
            kappa={},
        )
    return estimate_signals(hits, "time", rng, query_time_days=query_time_days)


def _mu_dead(hits: list[Hit], review: MemoryReview) -> np.ndarray | None:
    dead = [h for h in hits if h.doc.id in set(review.soft_neg_ids) or h.doc.id in set(review.hard_neg_ids)]
    if len(dead) < 2:
        dead = hits[3:] if len(hits) > 3 else []
    if not dead:
        return None
    return np.mean(np.stack([h.doc.embedding for h in dead], axis=0), axis=0)


def _mu_topk(hits: list[Hit], k: int | None = None) -> np.ndarray | None:
    take = hits[: k or len(hits)]
    if not take:
        return None
    stacked = np.stack([h.doc.embedding for h in take], axis=0).astype(np.float64)
    return np.mean(stacked, axis=0)


def _compose_query(
    q0: np.ndarray,
    v_slot: np.ndarray | None,
    mu: np.ndarray | None,
    *,
    beta: float,
    gamma: float,
) -> np.ndarray:
    q0n = l2_normalize(np.asarray(q0, dtype=np.float64).reshape(-1), axis=0)
    acc = ALPHA_Q0 * q0n
    basis: list[np.ndarray] = [q0n]
    if v_slot is not None:
        vs = l2_normalize(np.asarray(v_slot, dtype=np.float64).reshape(-1), axis=0)
        acc = acc + beta * vs
        basis.append(vs)
    if mu is not None and gamma:
        noise = orthogonal_noise(mu, *basis)
        acc = acc - gamma * noise
    return l2_normalize(acc, axis=0)


def _query_from_memory(
    state: EvolutionState,
    hits: list[Hit],
    review: MemoryReview,
    encode_query: Callable[[str], np.ndarray] | None,
    question: str,
    rng: np.random.Generator,
    query_time_days: float | None,
    dead_end: bool,
) -> EvolutionState:
    focus = review.query_focus.strip()
    if focus and focus in set(state.focus_blacklist):
        focus = " ".join(t for t in focus.split() if t not in set(state.focus_blacklist)).strip()
    hop = (review.hop_phrase or "").strip() or hop_phrase(question)
    slot_text = hop
    if not slot_text:
        cand = " ".join(k for k in review.missing_keys[:3] if not is_abstract_missing(k)).strip()
        slot_text = cand if cand and not is_abstract_missing(cand) else ""
    mu = _mu_topk(hits)
    mu_n = None
    if mu is not None:
        q0n = np.asarray(state.q0, dtype=np.float64).reshape(-1)
        vs_tmp = None
        if slot_text and encode_query is not None:
            vs_tmp = np.asarray(encode_query(slot_text), dtype=np.float64)
        mu_n = orthogonal_noise(mu, q0n, vs_tmp) if vs_tmp is not None else orthogonal_noise(mu, q0n)

    if dead_end:
        bl = tuple(dict.fromkeys(list(state.focus_blacklist) + [state.query_focus] + [focus]))
        bl = tuple(x for x in bl if x)
        return replace(
            state,
            q=np.asarray(state.q0, dtype=np.float32),
            v_slot=None,
            omega_hop=0.0,
            mu_noise=None if mu_n is None else mu_n.astype(np.float32),
            query_slot_active=True,
            query_focus="",
            focus_blacklist=bl,
            alpha=ALPHA_Q0,
            beta=0.0,
            gamma=DEAD_DELTA if mu_n is not None else 0.0,
        )

    extra = state
    v_arr = None
    omega = 0.0
    gamma = 0.0
    if slot_text and encode_query is not None and not is_abstract_missing(slot_text):
        v_arr = np.asarray(encode_query(slot_text), dtype=np.float32)
        omega = OMEGA_HOP
        gamma = GAMMA_NOISE if mu_n is not None else 0.0
    elif review.relevant_ids and (review.soft_neg_ids or review.hard_neg_ids):
        sig = _signals_from_memory(hits, review, rng, query_time_days)
        extra = rocchio_query(state, sig)
        return replace(
            extra,
            q=extra.q.astype(np.float32),
            v_slot=None,
            omega_hop=0.0,
            mu_noise=None,
            pos_ids=review.relevant_ids or extra.pos_ids,
            neg_ids=review.soft_neg_ids or extra.neg_ids,
            hard_neg_ids=review.hard_neg_ids or extra.hard_neg_ids,
            query_slot_active=True,
            query_focus=focus or extra.query_focus,
            query_locked=review.query_locked or extra.query_locked,
        )

    return replace(
        extra,
        q=np.asarray(state.q0, dtype=np.float32),
        v_slot=v_arr,
        omega_hop=float(omega),
        mu_noise=None if mu_n is None else mu_n.astype(np.float32),
        alpha=ALPHA_Q0,
        beta=float(omega),
        gamma=float(gamma),
        pos_ids=review.relevant_ids or extra.pos_ids,
        neg_ids=review.soft_neg_ids or extra.neg_ids,
        hard_neg_ids=review.hard_neg_ids or extra.hard_neg_ids,
        query_slot_active=True,
        query_focus=slot_text or focus or extra.query_focus,
        query_locked=review.query_locked or extra.query_locked,
    )


def _rank_from_memory(
    state: EvolutionState,
    hits: list[Hit],
    review: MemoryReview,
    rng: np.random.Generator,
    query_time_days: float | None,
    question: str,
    rank_task: str,
) -> EvolutionState:
    task = (rank_task or "").strip()
    if task == "multihop":
        from .memory_review import _attribute_terms, _overlap, _terms

        keys = _attribute_terms(question, _terms(question))

        def _ov(h):
            return _overlap(keys, h.doc.text) if keys else 0.0

        return apply_multihop_rank(state, hits, rng, _ov)
    if task == "time":
        sig = estimate_signals(hits, "time", rng, query_time_days=query_time_days)
        updated = apply_rank_update(
            state,
            hits,
            sig,
            boost_recency=True,
            hard_neg_ids=tuple(sig.neg_ids) + tuple(review.hard_neg_ids),
            allow_recency=True,
        )
        return replace(
            updated,
            eta_neg=max(updated.eta_neg, 0.30) if updated.neg_ids else updated.eta_neg,
            rank_task="time",
            query_locked=True,
        )
    sig = _signals_from_memory(hits, review, rng, query_time_days)
    updated = apply_rank_update(
        state,
        hits,
        sig,
        boost_recency=review.time_mismatch,
        hard_neg_ids=review.hard_neg_ids,
        allow_recency=review.time_mismatch,
    )
    soft = tuple(dict.fromkeys(list(updated.neg_ids) + list(review.soft_neg_ids)))
    return replace(
        updated,
        neg_ids=soft,
        eta_neg=max(updated.eta_neg, 0.35) if soft else updated.eta_neg,
        query_locked=not review.missing_keys,
    )
