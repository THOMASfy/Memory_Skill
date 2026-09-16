"""Retrieval-only probe: dense q0 vs one QUERY slate (no LLM, no QA).

  python -u -m skill_evolution.slate_probe
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .longmemeval_eval import (
    BGEEncoder,
    _env_first,
    _load_dotenv,
    download_split,
    instance_documents,
    load_instances,
    parse_lme_date,
    session_text,
)
from .memory_review import hop_phrase, is_abstract_missing, parse_target_slot, review_memory
from .memory_update import apply_memory_skill
from .retrieve import dense_retrieve
from .types import EvolutionState


DEFAULT_IDS = ("e47becba", "118b2229", "76d63226")


def _gold_rank(ids: list[str], gold: set[str]) -> int | None:
    for i, x in enumerate(ids, start=1):
        if x in gold:
            return i
    return None


def probe_one(entry: dict, encoder: BGEEncoder, top_k: int) -> str:
    qid = entry["question_id"]
    question = entry["question"]
    qdate = entry.get("question_date") or ""
    gold = {str(x) for x in entry.get("answer_session_ids") or []}
    q_text = f"Current date: {qdate}\n{question}"
    texts = [session_text(s, str(d)) for s, d in zip(entry["haystack_sessions"], entry["haystack_dates"])]
    doc_vecs = encoder.encode(texts, is_query=False)
    q_vec = encoder.encode([q_text], is_query=True)[0]
    docs = instance_documents(entry, doc_vecs)
    tgt = parse_target_slot(q_text)
    hop = hop_phrase(q_text, tgt)

    dense_state = EvolutionState(q0=q_vec.copy(), q=q_vec.copy())
    dense_hits = dense_retrieve(docs, dense_state, top_k=top_k)
    dense_ids = [h.doc.id for h in dense_hits]
    dense_gr = _gold_rank(dense_ids, gold)

    mem = review_memory(q_text, dense_hits, llm_complete=None, query_time_days=parse_lme_date(qdate))

    def encode_q(text: str) -> np.ndarray:
        return encoder.encode([text], is_query=True)[0]

    rng = np.random.default_rng(0)
    slate_state = apply_memory_skill(
        dense_state,
        dense_hits,
        mem,
        slot="query",
        encode_query=encode_q,
        question=q_text,
        rng=rng,
        query_time_days=parse_lme_date(qdate),
        dead_end=False,
    )
    slate_hits = dense_retrieve(docs, slate_state, top_k=top_k)
    slate_ids = [h.doc.id for h in slate_hits]
    slate_gr = _gold_rank(slate_ids, gold)

    lines = [
        f"qid={qid}",
        f"Q: {question}",
        f"tvc={tgt.cls}/{tgt.grain} abstract_label={tgt.label!r}",
        f"hop={hop!r} encode_abstract={is_abstract_missing(hop or tgt.label)}",
        f"auditor missing={list(mem.missing_keys)} covers={mem.covers_question} hop_phrase={mem.hop_phrase!r}",
        f"slate omega={slate_state.omega_hop:.2f} gamma={slate_state.gamma:.2f} v_slot={'yes' if slate_state.v_slot is not None else 'no'}",
        f"dense gold_rank={dense_gr} top5={dense_ids[:5]}",
        f"slate gold_rank={slate_gr} top5={slate_ids[:5]}",
        f"delta_rank={(None if dense_gr is None or slate_gr is None else dense_gr - slate_gr)}  (+ means gold moved up)",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    _load_dotenv()
    p = argparse.ArgumentParser()
    p.add_argument("--ids", default=",".join(DEFAULT_IDS))
    p.add_argument("--split", default="s")
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args(argv)
    want = [x.strip() for x in args.ids.split(",") if x.strip()]
    instances = load_instances(download_split(args.split))
    by_id = {e["question_id"]: e for e in instances}
    missing = [i for i in want if i not in by_id]
    if missing:
        print("missing ids:", missing, file=sys.stderr)
    encoder = BGEEncoder(_env_first("ENCODER", default="BAAI/bge-small-en-v1.5"))
    chunks = []
    for qid in want:
        entry = by_id.get(qid)
        if entry is None:
            continue
        print(f"probe {qid}", flush=True)
        text = probe_one(entry, encoder, args.top_k)
        chunks.append(text)
        print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
