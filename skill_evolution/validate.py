"""Reject fill-in, invariant breaks, and multi-slot edits in one round."""

from __future__ import annotations

from .slots import SlotSnapshot, parse_skill
from .types import Document, SlotName

MAX_SNIPPET = 24


def assert_invariants(baseline: SlotSnapshot, current_text: str) -> None:
    now = parse_skill(current_text)
    if now.invariant_hash != baseline.invariant_hash:
        raise ValueError("SKILL.md backbone changed; evolve slots are the only writable region")


def assert_one_slot(before: SlotSnapshot, after_text: str, slot: SlotName) -> None:
    after = parse_skill(after_text)
    q_changed = after.query_body != before.query_body
    r_changed = after.rank_body != before.rank_body
    f_changed = after.filter_body != before.filter_body
    n_changed = int(q_changed) + int(r_changed) + int(f_changed)
    if n_changed > 1:
        raise ValueError("more than one slot changed in one round")
    if slot == "query":
        if not q_changed:
            raise ValueError("query slot was selected but body did not change")
        if r_changed or f_changed:
            raise ValueError("non-query slot mutated during a query update")
    elif slot == "rank":
        if not r_changed:
            raise ValueError("rank slot was selected but body did not change")
        if q_changed or f_changed:
            raise ValueError("non-rank slot mutated during a rank update")
    elif slot == "filter":
        if not f_changed:
            raise ValueError("filter slot was selected but body did not change")
        if q_changed or r_changed:
            raise ValueError("non-filter slot mutated during a filter update")
    else:
        raise ValueError(f"unknown slot {slot}")


def assert_no_passages(skill_text: str, docs: list[Document]) -> None:
    body = skill_text.lower()
    for doc in docs:
        snippet = "".join(doc.text.split())
        if len(snippet) < MAX_SNIPPET:
            continue
        # Any long whitespace-stripped window from a passage is fill-in.
        for i in range(0, len(snippet) - MAX_SNIPPET + 1, MAX_SNIPPET):
            piece = snippet[i : i + MAX_SNIPPET].lower()
            if piece and piece in "".join(body.split()):
                raise ValueError(f"passage snippet leaked into SKILL.md from doc {doc.id}")
