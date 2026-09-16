"""Compile a memory-informed FILTER slot: drop ids + bind-gate. No passage text."""

from __future__ import annotations

from dataclasses import replace

from .memory_review import (
    MemoryReview,
    _GENERIC,
    _STOP,
    _has_term,
    _term_or_syn,
    parse_target_slot,
)
from .types import EvolutionState, Hit

FILTER_ETA = 0.75
POLLUTE_K = 5
POLLUTE_MIN = 2


_WEAK_BIND = {
    "size", "name", "brand", "count", "number", "title", "item", "place",
    "color", "colour", "date", "year", "gift", "party",
}


def distinctive_bind(question: str) -> tuple[str, ...]:
    tgt = parse_target_slot(question)
    out: list[str] = []
    for t in tgt.bind:
        if t in _GENERIC or t in _STOP or t in _WEAK_BIND or len(t) < 4:
            continue
        if t not in out:
            out.append(t)
        if len(out) >= 6:
            break
    return tuple(out)


def doc_bound(text: str, bind: tuple[str, ...]) -> bool:
    if not bind:
        return True
    return any(_has_term(b, text) or _term_or_syn(b, text) for b in bind)


def filter_ready(question: str, hits: list[Hit], review: MemoryReview) -> bool:
    bind = distinctive_bind(question)
    unbound = sum(
        1 for h in hits[:POLLUTE_K] if bind and not doc_bound(h.doc.text or "", bind)
    )
    return bool(bind) and unbound >= POLLUTE_MIN


def apply_filter_update(
    state: EvolutionState,
    hits: list[Hit],
    review: MemoryReview,
    question: str,
) -> EvolutionState:
    bind = distinctive_bind(question)
    rel = set(review.relevant_ids)
    noisy = set(review.soft_neg_ids) | set(review.hard_neg_ids)
    unbound_top = [
        h for h in hits[:POLLUTE_K] if bind and not doc_bound(h.doc.text or "", bind)
    ]
    polluted = bool(bind) and len(unbound_top) >= POLLUTE_MIN
    drop: list[str] = []
    for h in hits:
        if h.doc.id in rel:
            continue
        unbound = bool(bind) and not doc_bound(h.doc.text or "", bind)
        if unbound and (polluted or h.doc.id in noisy):
            drop.append(h.doc.id)
    drop_ids = tuple(dict.fromkeys(drop))[:24]
    require = polluted
    return replace(
        state,
        filter_bind=bind,
        filter_drop_ids=drop_ids,
        filter_eta=FILTER_ETA if (drop_ids or require) else 0.0,
        filter_require_bind=require,
        filter_slot_active=True,
    )


def filter_penalty(doc_id: str, text: str, state: EvolutionState) -> float:
    if not getattr(state, "filter_slot_active", False):
        return 0.0
    eta = float(getattr(state, "filter_eta", 0.0) or 0.0)
    if eta <= 0:
        return 0.0
    if doc_id in set(getattr(state, "filter_drop_ids", ()) or ()):
        return eta
    bind = tuple(getattr(state, "filter_bind", ()) or ())
    if getattr(state, "filter_require_bind", False) and bind and not doc_bound(text or "", bind):
        return eta
    return 0.0
