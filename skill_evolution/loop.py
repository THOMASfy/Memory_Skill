"""Agentic RAG: retrieve → audit → update QUERY / FILTER / RANK → retrieve.

The loop LLM is only a memory auditor (enough to answer?). A dedicated reader
answers once after evolution, in the eval harness — never inside this loop.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .compile_skill import write_state_slots
from .diagnose import MIN_NEFF, diagnose
from .filter_update import filter_ready
from .memory_review import review_memory
from .memory_update import apply_memory_skill
from .query_update import rocchio_query
from .rank_update import apply_rank_update
from .retrieve import dense_retrieve
from .signals import estimate_signals
from .slots import parse_skill
from .types import Document, EvolutionState, GapReport, RoundLog, SlotName
from .validate import assert_invariants, assert_no_passages, assert_one_slot
from .workspace import SkillWorkspace


def infer_rank_task(question_type: str | None, question: str) -> str:
    """QUERY/FILTER stay the default evolve path. RANK is not forced by LME type."""
    return ""

DEFAULT_SKILL = Path(__file__).resolve().parent.parent / "dense-vector-retrieval" / "SKILL.md"


@dataclass
class EvolutionResult:
    evidence: list
    sufficient: bool
    rounds: list[RoundLog] = field(default_factory=list)
    working_skill_text: str | None = None


def decide_action(
    *,
    missing: tuple[str, ...],
    covers: bool,
    used: set[SlotName],
    dead_end: bool,
    round_index: int,
    max_rounds: int,
    gap_kind: str,
    hop_ok: bool = True,
    filter_ok: bool = False,
    time_mismatch: bool = False,
) -> tuple[str, SlotName | None, str]:
    """One slot per round. Inject only when the operator has a concrete job.

    Stop when auditor covers (concrete value in Top-3). Empty missing without
    covers is NOT enough — do not invent QUERY/RANK just to fill a round.

    QUERY: still missing + unused + bindable hop (never abstract labels).
    FILTER: still missing + Top-5 polluted by unbound slices.
    RANK: still missing + question has a time cue and timestamps disagree.
    Otherwise stop (hand evidence to Reader / abstain).
    """
    if gap_kind == "unanswerable":
        return "stop", None, "unanswerable"
    if covers:
        return "stop", None, "enough"
    if round_index == max_rounds:
        return "stop", None, "max_rounds"
    if dead_end:
        if filter_ok and "filter" not in used:
            return "inject", "filter", "dead_end_filter"
        return "stop", None, "dead_end_stop"
    need = bool(missing) or not covers
    if need and "query" not in used and hop_ok and bool(missing):
        return "inject", "query", "missing_query_once"
    if need and filter_ok and "filter" not in used:
        return "inject", "filter", "memory_filter"
    if need and time_mismatch and "rank" not in used:
        return "inject", "rank", "time_mismatch_rank"
    return "stop", None, "no_operator"


def _choose_slot(round_index: int, used: set[SlotName], gap: GapReport) -> SlotName | None:
    if gap.sufficient or gap.kind == "unanswerable":
        return None
    plan: list[SlotName] = ["query", "rank"]
    if gap.kind == "unknown":
        return None
    for slot in plan:
        if slot not in used:
            if slot == "query" and gap.n_eff_alias < MIN_NEFF and gap.kind == "alias":
                continue
            if slot == "rank" and gap.n_eff_time < MIN_NEFF and gap.kind == "time":
                continue
            if slot == "query" and gap.kind == "time" and gap.n_eff_time < MIN_NEFF:
                continue
            return slot
    return None


def run_evolution(
    query_vec: np.ndarray,
    documents: list[Document],
    *,
    baseline_skill_md: Path | None = None,
    max_rounds: int = 3,
    top_k: int = 10,
    seed: int = 0,
    persist_workspace: bool = False,
    query_time_days: float | None = None,
    question: str | None = None,
    encode_query: Callable[[str], np.ndarray] | None = None,
    llm_complete: Callable[[str], str] | None = None,
    question_type: str | None = None,
) -> EvolutionResult:
    rng = np.random.default_rng(seed)
    skill_md = baseline_skill_md or DEFAULT_SKILL
    ws = SkillWorkspace.open(skill_md)
    baseline = parse_skill(ws.skill_md.read_text(encoding="utf-8"))
    state = EvolutionState(q0=query_vec.copy(), q=query_vec.copy())
    used: set[SlotName] = set()
    logs: list[RoundLog] = []
    hits = []
    last_gap: GapReport | None = None
    prev_proxy = -1.0
    prev_slot: SlotName | None = None
    rank_task = infer_rank_task(question_type, question or "")
    use_memory = bool((question or "").strip()) and (
        encode_query is not None or llm_complete is not None
    )
    try:
        for t in range(1, max_rounds + 1):
            hits = dense_retrieve(documents, state, top_k=top_k)
            gap = diagnose(hits, state, rng)
            mem = (
                review_memory(
                    question or "",
                    hits,
                    llm_complete=llm_complete,
                    query_time_days=query_time_days,
                )
                if use_memory
                else None
            )
            if mem is not None and not mem.covers_question and gap.sufficient:
                gap = GapReport(
                    kind=gap.kind if gap.kind != "sufficient" else "unknown",
                    sufficient=False,
                    g_time=gap.g_time,
                    g_alias=gap.g_alias,
                    g_unanswerable=gap.g_unanswerable,
                    n_eff_time=gap.n_eff_time,
                    n_eff_alias=gap.n_eff_alias,
                    reason=f"geometry said sufficient; {mem.note}",
                )
            last_gap = gap
            keys = tuple(mem.missing_keys) if mem is not None else ()
            dead_end = bool(
                mem is not None
                and prev_slot == "query"
                and mem.proxy <= prev_proxy + 1e-6
            )
            if mem is not None:
                kind, slot, why = decide_action(
                    missing=mem.missing_keys,
                    covers=mem.covers_question,
                    used=used,
                    dead_end=dead_end,
                    round_index=t,
                    max_rounds=max_rounds,
                    gap_kind=gap.kind,
                    hop_ok=bool((mem.hop_phrase or "").strip()),
                    filter_ok=filter_ready(question or "", hits, mem),
                    time_mismatch=bool(mem.time_mismatch),
                )
            else:
                if t == max_rounds or gap.kind == "unanswerable" or gap.sufficient:
                    kind, slot, why = "stop", None, gap.reason
                else:
                    slot = _choose_slot(t, used, gap)
                    kind, why = ("stop", "no eligible slot") if slot is None else ("inject", f"compiled {slot}")
            if kind == "stop":
                note = f"{why}; {mem.note}" if mem is not None else why
                logs.append(RoundLog(t, gap, None, True, [h.doc.id for h in hits], note[:240], keys))
                break
            if slot is None:
                logs.append(RoundLog(t, gap, None, False, [h.doc.id for h in hits], why, keys))
                break
            before = parse_skill(ws.skill_md.read_text(encoding="utf-8"))
            try:
                if mem is not None:
                    new_state = apply_memory_skill(
                        state,
                        hits,
                        mem,
                        slot=slot,
                        encode_query=encode_query,
                        question=question or "",
                        rng=rng,
                        query_time_days=query_time_days,
                        dead_end=dead_end and slot == "query",
                        rank_task=rank_task,
                    )
                else:
                    sig = estimate_signals(
                        hits,
                        gap.kind if gap.kind in {"time", "alias"} else "time",
                        rng,
                        query_time_days=query_time_days,
                    )
                    if sig.n_eff < MIN_NEFF and gap.kind == "time":
                        logs.append(RoundLog(t, gap, None, False, [h.doc.id for h in hits], "N_eff below gate", keys))
                        break
                    if slot == "query":
                        new_state = rocchio_query(state, sig)
                        if new_state.beta == 0.0 and new_state.gamma == 0.0:
                            slot = "rank"
                            new_state = apply_rank_update(state, hits, sig)
                    else:
                        new_state = apply_rank_update(state, hits, sig)
                write_state_slots(ws.skill_md, new_state)
                text = ws.skill_md.read_text(encoding="utf-8")
                assert_invariants(baseline, text)
                assert_one_slot(before, text, slot)
                assert_no_passages(text, documents)
            except ValueError as exc:
                logs.append(RoundLog(t, gap, slot, False, [h.doc.id for h in hits], f"rollback: {exc}", keys))
                write_state_slots(ws.skill_md, state)
                continue
            state = new_state
            used.add(slot)
            extra = f"{why}; {mem.note}" if mem is not None else f"{why}"
            logs.append(RoundLog(t, gap, slot, True, [h.doc.id for h in hits], extra, keys))
            if mem is not None:
                prev_proxy = mem.proxy
                prev_slot = slot
        snapshot = ws.skill_md.read_text(encoding="utf-8")
        sufficient = bool(last_gap and last_gap.sufficient)
        if use_memory:
            sufficient = bool(mem is not None and mem.covers_question)
        result = EvolutionResult(
            evidence=hits,
            sufficient=sufficient,
            rounds=logs,
            working_skill_text=snapshot,
        )
        return result
    finally:
        if not persist_workspace:
            ws.destroy()
