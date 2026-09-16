"""Step-by-step trace of QUERY/RANK injection, stop/discard, reader, judge.

Gold ids are used only to print ranks — they never enter the auditor or updates.

  python -u -m skill_evolution.case_trace
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

from .compile_skill import write_state_slots
from .diagnose import diagnose
from .filter_update import filter_ready
from .longmemeval_eval import (
    AUDITOR_SYSTEM,
    BGEEncoder,
    READER_SYSTEM,
    READER_TEMPLATE,
    _env_first,
    _load_dotenv,
    chat,
    download_split,
    format_history,
    instance_documents,
    llm_client,
    load_instances,
    parse_judge_label,
    parse_lme_date,
    run_judge,
    select_reader_hits,
    session_text,
)
from .loop import DEFAULT_SKILL, decide_action
from .memory_review import review_memory
from .memory_update import apply_memory_skill
from .retrieve import dense_retrieve
from .slots import parse_skill
from .types import EvolutionState, SlotName
from .validate import assert_invariants, assert_no_passages, assert_one_slot
from .workspace import SkillWorkspace

# Cases that looked broken in the 21-row evo log.
DEFAULT_IDS = (
    "e47becba",  # degree gold@6, enough=1 too early
    "3b6f954b",  # study abroad gold@4, enough=1
    "6ade9755",  # yoga gold@1, reader ok, judge fail
    "51a45a95",  # coupon gold@1, reader missed store
    "58ef2f1c",  # volunteer gold@1 but still QUERY
    "f8c5f88b",  # tennis gold@2, reader ok, judge fail
    "66f24dbb",  # birthday gold@1, QUERY + dead-end
    "af8d2e46",  # shirts gold@1, reader I-don't-know
)

OUT = Path("runs/longmemeval/case_trace.txt")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64).reshape(-1)
    y = np.asarray(b, dtype=np.float64).reshape(-1)
    den = float(np.linalg.norm(x) * np.linalg.norm(y)) + 1e-12
    return float(np.dot(x, y) / den)


def _gold_rank(ids: list[str], gold: set[str]) -> int | None:
    for i, x in enumerate(ids, start=1):
        if x in gold:
            return i
    return None


def _flags(gold_rank: int | None, missing_llm, missing_after, action: str) -> list[str]:
    tags = []
    if missing_llm and not missing_after:
        tags.append("启发式清空了LLM-missing")
    if gold_rank is not None and gold_rank <= 3 and action == "query":
        tags.append("gold已在Top-3仍QUERY")
    if gold_rank is not None and gold_rank > 5 and action == "stop":
        tags.append("gold不在Top-5却停机")
    if 3 < (gold_rank or 0) <= 10 and action == "stop":
        tags.append("gold在4-10却停机(更该RANK或给Reader)")
    if action == "query-dead-end":
        tags.append("dead-end舍弃原QUERY(β=0,只减噪声)")
    return tags


def trace_one(
    entry: dict,
    encoder: BGEEncoder,
    client,
    model: str,
    official: bool,
    *,
    top_k: int,
    max_rounds: int,
    seed: int,
    do_qa: bool,
) -> str:
    lines: list[str] = []
    qid = entry["question_id"]
    question = entry["question"]
    qdate = entry.get("question_date") or ""
    gold = {str(x) for x in entry.get("answer_session_ids") or []}
    gold_answer = entry.get("answer") or ""
    qtype = entry.get("question_type")
    q_text = f"Current date: {qdate}\n{question}"
    texts = [session_text(s, str(d)) for s, d in zip(entry["haystack_sessions"], entry["haystack_dates"])]
    doc_vecs = encoder.encode(texts, is_query=False)
    q_vec = encoder.encode([q_text], is_query=True)[0]
    docs = instance_documents(entry, doc_vecs)
    q_days = parse_lme_date(qdate)

    def encode_q(text: str) -> np.ndarray:
        return encoder.encode([text], is_query=True)[0]

    def llm_json(prompt: str) -> str:
        return chat(
            client,
            model,
            prompt,
            max_tokens=512,
            thinking=False,
            official_deepseek=official,
            include_reasoning=False,
            system=AUDITOR_SYSTEM,
            mode="json",
        )

    lines.append("=" * 78)
    lines.append(f"CASE {qid}  type={qtype}")
    lines.append(f"Q: {question}")
    lines.append(f"gold_answer: {gold_answer}")
    lines.append(f"gold_session_ids: {sorted(gold)}  (只用于打印名次, 不进审计)")
    lines.append("")

    rng = np.random.default_rng(seed)
    ws = SkillWorkspace.open(DEFAULT_SKILL)
    baseline = parse_skill(ws.skill_md.read_text(encoding="utf-8"))
    state = EvolutionState(q0=q_vec.copy(), q=q_vec.copy())
    used: set[SlotName] = set()
    prev_proxy = -1.0
    prev_slot: SlotName | None = None
    hits = []
    stop_reason = ""

    try:
        for t in range(1, max_rounds + 1):
            hits = dense_retrieve(docs, state, top_k=top_k)
            top_ids = [h.doc.id for h in hits]
            gr = _gold_rank(top_ids, gold)
            gap = diagnose(hits, state, rng)
            heur = review_memory(q_text, hits, llm_complete=None, query_time_days=q_days)
            mem = review_memory(q_text, hits, llm_complete=llm_json, query_time_days=q_days)
            dead_end = bool(prev_slot == "query" and mem.proxy <= prev_proxy + 1e-6)
            lines.append(f"--- round {t} 检索 ---")
            lines.append(f"  gold_rank={gr}  top5={top_ids[:5]}")
            lines.append(f"  cos(q,q0)={_cos(state.q, state.q0):.4f}  α={state.alpha:.2f} β={state.beta:.2f} γ={state.gamma:.2f}")
            lines.append(f"  启发式 missing={list(heur.missing_keys)} slot={heur.slot} cause={heur.cause}")
            lines.append(f"  LLM missing={list(mem.missing_llm)}  生效 missing={list(mem.missing_keys)} covers={mem.covers_question}")
            lines.append(f"  cause={mem.cause} proxy={mem.proxy:.3f} dead_end={dead_end}")
            kind, slot, why = decide_action(
                missing=mem.missing_keys,
                covers=mem.covers_question,
                used=used,
                dead_end=dead_end,
                round_index=t,
                max_rounds=max_rounds,
                gap_kind=gap.kind,
                hop_ok=bool((mem.hop_phrase or "").strip()),
                filter_ok=filter_ready(q_text, hits, mem),
                time_mismatch=bool(mem.time_mismatch),
            )
            lines.append(f"  decide: {kind} slot={slot} why={why}")

            if kind == "stop":
                action = "stop"
                tags = _flags(gr, mem.missing_llm, mem.missing_keys, action)
                lines.append(f"  判断: 停机  reason={why}")
                if tags:
                    lines.append(f"  问题标记: {tags}")
                lines.append("")
                break

            action = slot or "stop"
            tags = _flags(gr, mem.missing_llm, mem.missing_keys, action)
            lines.append(f"  判断: 注入槽={slot}  {why}")
            if tags:
                lines.append(f"  问题标记: {tags}")

            before = parse_skill(ws.skill_md.read_text(encoding="utf-8"))
            try:
                new_state = apply_memory_skill(
                    state,
                    hits,
                    mem,
                    slot=slot,
                    encode_query=encode_q,
                    question=q_text,
                    rng=rng,
                    query_time_days=q_days,
                    dead_end=dead_end and slot == "query",
                )
                write_state_slots(ws.skill_md, new_state)
                text = ws.skill_md.read_text(encoding="utf-8")
                assert_invariants(baseline, text)
                assert_one_slot(before, text, slot)
                assert_no_passages(text, docs)
            except ValueError as exc:
                lines.append(f"  舍弃: 校验失败 rollback  slot={slot}  err={exc}")
                write_state_slots(ws.skill_md, state)
                continue

            dq = 1.0 - _cos(new_state.q, state.q)
            lines.append(
                f"  注入: α={new_state.alpha:.2f} β={new_state.beta:.2f} γ={new_state.gamma:.2f} "
                f"Δcos(q_new,q_old)={dq:.4f} cos(q_new,q0)={_cos(new_state.q, state.q0):.4f}"
            )
            lines.append(f"  query_focus={new_state.query_focus!r}  blacklist={new_state.focus_blacklist}")
            if dead_end and slot == "query":
                lines.append("  舍弃: dead-end 丢掉上一轮focus, β=0 只对Top-k均值做正交减噪")
            state = new_state
            used.add(slot)
            prev_proxy = mem.proxy
            prev_slot = slot
            lines.append("")
    finally:
        ws.destroy()

    retrieved = [h.doc.id for h in hits]
    gr_final = _gold_rank(retrieved, gold)
    lines.append(f"终态 gold_rank={gr_final} top10={retrieved[:10]}")
    gold_in_reader = gr_final is not None and gr_final <= top_k
    lines.append(f"Reader可见gold session: {gold_in_reader} (Top-{top_k})")

    if do_qa and client is not None:
        reader_hits = select_reader_hits(hits, top_k, str(qtype or ""))
        history = format_history(
            reader_hits, entry, qtype=str(qtype or ""), question_date=qdate
        )
        prompt = READER_TEMPLATE.format(history=history, date=qdate, question=question)
        hyp = chat(
            client,
            model,
            prompt,
            max_tokens=1200,
            thinking=False,
            official_deepseek=official,
            include_reasoning=False,
            system=READER_SYSTEM,
            mode="reader",
        )
        verdict = run_judge(client, model, official, "", question, gold_answer, hyp)
        parsed = parse_judge_label(verdict)
        lines.append("")
        lines.append("--- Reader ---")
        lines.append(hyp[:800])
        lines.append("")
        lines.append("--- Judge raw ---")
        lines.append((verdict or "")[:600])
        lines.append(f"parse_judge_label → {parsed}")
        ans_l = gold_answer.lower().strip()
        hyp_has = bool(ans_l) and ans_l[:40] in (hyp or "").lower()
        lines.append(f"hypothesis是否含gold答案子串(粗检): {hyp_has}")
        if hyp_has and parsed is False:
            lines.append("  问题标记: Reader已含答案但Judge=false (解析/截断/过严)")
        if (not hyp_has) and gold_in_reader:
            lines.append("  问题标记: gold session在Top-k但Reader没用上(截断/没读到/说不知道)")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Trace injection/stop/reader/judge on known bad cases.")
    p.add_argument("--ids", default=",".join(DEFAULT_IDS))
    p.add_argument("--split", default="s")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--no-qa", action="store_true", help="Skip reader/judge (faster, IR/auditor only).")
    args = p.parse_args(argv)

    api_key = _env_first("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", default="")
    base_url = _env_first("LLM_API_BASE", "OPENAI_BASE_URL", default="http://10.0.62.177:4000/v1")
    model = _env_first("LLM_MODEL", default="deepseek-v4-free")
    if not api_key:
        api_key = "-"
    official = "api.deepseek.com" in base_url
    client = llm_client(api_key, base_url)
    print(f"LLM model={model} base={base_url}", flush=True)

    want = [x.strip() for x in args.ids.split(",") if x.strip()]
    instances = load_instances(download_split(args.split))
    by_id = {e["question_id"]: e for e in instances}
    missing = [i for i in want if i not in by_id]
    if missing:
        print("missing ids:", missing, file=sys.stderr)

    encoder = BGEEncoder(_env_first("ENCODER", default="BAAI/bge-small-en-v1.5"))
    chunks: list[str] = []
    for i, qid in enumerate(want):
        entry = by_id.get(qid)
        if entry is None:
            continue
        print(f"tracing {qid} ({i+1}/{len(want)})", flush=True)
        text = trace_one(
            entry,
            encoder,
            client,
            model,
            official,
            top_k=args.top_k,
            max_rounds=args.max_rounds,
            seed=i + 1,
            do_qa=not args.no_qa,
        )
        chunks.append(text)
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text("\n".join(chunks), encoding="utf-8")
        print(f"  wrote {qid} -> {OUT}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(chunks), encoding="utf-8")
    print(f"Saved {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
