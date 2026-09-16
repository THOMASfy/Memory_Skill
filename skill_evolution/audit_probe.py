"""Probe LLM missing_keys auditor on a few LongMemEval questions.

  python -u -m skill_evolution.audit_probe --ids 76d63226,118b2229,e47becba,726462e0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .longmemeval_eval import (
    BGEEncoder,
    _env_first,
    _load_dotenv,
    download_split,
    instance_documents,
    llm_client,
    load_instances,
    parse_lme_date,
    session_text,
    chat,
)
from .memory_review import review_memory
from .retrieve import dense_retrieve
from .types import EvolutionState


DEFAULT_IDS = (
    "76d63226",  # Samsung TV size, often miss top-10
    "726462e0",  # discount, R@10=0 in evo log
    "118b2229",  # commute
    "e47becba",  # degree, false-cover
    "1e043500",  # spotify playlist
    "58bf7951",  # play / Glass Menagerie
)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser(description="Test LLM missing_keys auditor.")
    p.add_argument("--split", default="s")
    p.add_argument("--ids", default=",".join(DEFAULT_IDS))
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--model", default=None)
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    args = p.parse_args(argv)

    api_key = args.api_key or _env_first("LLM_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_API_KEY", default="")
    base_url = args.base_url or _env_first("LLM_API_BASE", "OPENAI_BASE_URL", default="http://10.0.62.177:4000/v1")
    model = args.model or _env_first("LLM_MODEL", default="deepseek-v4-free")
    if not api_key:
        print("Set LLM_API_KEY (LiteLLM sk-...).", file=sys.stderr)
        return 2
    official = "api.deepseek.com" in base_url
    client = llm_client(api_key, base_url)
    print(f"LLM model={model} base={base_url}")

    want = {x.strip() for x in args.ids.split(",") if x.strip()}
    instances = load_instances(download_split(args.split))
    picked = [e for e in instances if e["question_id"] in want]
    if len(picked) < len(want):
        found = {e["question_id"] for e in picked}
        print("missing ids:", sorted(want - found), file=sys.stderr)

    encoder = BGEEncoder(_env_first("ENCODER", default="BAAI/bge-small-en-v1.5"))
    rows = []
    for entry in picked:
        qid = entry["question_id"]
        question = entry["question"]
        qdate = entry.get("question_date") or ""
        gold = [str(x) for x in entry.get("answer_session_ids") or []]
        texts = [session_text(s, str(d)) for s, d in zip(entry["haystack_sessions"], entry["haystack_dates"])]
        doc_vecs = encoder.encode(texts, is_query=False)
        q_text = f"Current date: {qdate}\n{question}"
        q_vec = encoder.encode([q_text], is_query=True)[0]
        docs = instance_documents(entry, doc_vecs)
        hits = dense_retrieve(docs, EvolutionState(q0=q_vec, q=q_vec), top_k=args.top_k)
        top_ids = [h.doc.id for h in hits]
        gold_set = set(gold)
        rank = next((i + 1 for i, x in enumerate(top_ids) if x in gold_set), None)

        heuristic = review_memory(q_text, hits, llm_complete=None, query_time_days=parse_lme_date(qdate))
        audit = review_memory(
            q_text,
            hits,
            llm_complete=lambda prompt, _c=client, _m=model, _od=official: chat(
                _c, _m, prompt, max_tokens=512, thinking=False, official_deepseek=_od, include_reasoning=False
            ),
            query_time_days=parse_lme_date(qdate),
        )
        llm_used = str(audit.note).startswith("attr-llm:")
        expect_missing = rank is None or rank > 5
        row = {
            "id": qid,
            "question": question,
            "gold_rank": rank,
            "expect_missing_keys_nonempty": expect_missing,
            "heuristic_missing": list(heuristic.missing_keys),
            "heuristic_slot": heuristic.slot,
            "llm_ok": llm_used,
            "llm_missing": list(audit.missing_keys),
            "llm_slot": audit.slot,
            "llm_note": audit.note,
            "top3": top_ids[:3],
        }
        rows.append(row)
        print("=" * 72)
        print(f"{qid} gold_rank={rank} (None means not in top-{args.top_k})")
        print(f"Q: {question}")
        print(f"heuristic missing={list(heuristic.missing_keys)} slot={heuristic.slot}")
        print(f"llm_ok={llm_used} missing={list(audit.missing_keys)} slot={audit.slot}")
        print(f"note: {audit.note}")
        if expect_missing and not audit.missing_keys:
            print("WARN: gold not in top-5 but auditor missing_keys empty")
        if (rank is not None and rank <= 3) and audit.missing_keys:
            print("NOTE: gold already in top-3; missing_keys still set (may be extra attributes)")

    out = os.path.join("runs", "longmemeval", "audit_probe.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    n_ok = sum(1 for r in rows if r["llm_ok"])
    n_hit = sum(
        1
        for r in rows
        if r["expect_missing_keys_nonempty"] == bool(r["llm_missing"])
    )
    print("=" * 72)
    print(f"JSON parsed {n_ok}/{len(rows)}; missing nonempty aligned with gold not-in-top5: {n_hit}/{len(rows)}")
    print(f"Saved {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
