import re
import os

os.environ.setdefault("EMBEDDING_DEVICE", "cpu")

from skill_evolution.longmemeval_eval import (
    BGEEncoder,
    _env_first,
    build_chunk_documents,
    download_split,
    evidence_supports_answer,
    load_instances,
    parse_lme_date,
    select_reader_hits,
    session_text,
)
from skill_evolution.memory_review import parse_target_slot, review_memory, value_asserted_in_text
from skill_evolution.retrieve import dense_retrieve
from skill_evolution.types import EvolutionState

entry = next(e for e in load_instances(download_split("s")) if e["question_id"] == "e47becba")
q = entry["question"]
print("Q", q)
gold = str(entry["answer_session_ids"][0])
for sid, d, s in zip(entry["haystack_session_ids"], entry["haystack_dates"], entry["haystack_sessions"]):
    if str(sid) != gold:
        continue
    t = session_text(s, str(d))
    print("gold len", len(t), "answer", entry.get("answer"))
    for m in re.finditer(r".{0,50}(degree|bachelor|master|MBA|PhD|graduate).{0,80}", t, re.I):
        print("CTX", m.group(0).replace("\n", " | ")[:220])
    tgt = parse_target_slot(q)
    print("tgt", tgt)
    print("value_asserted full gold", value_asserted_in_text(tgt, t, q))

enc = BGEEncoder(_env_first("ENCODER", default="BAAI/bge-small-en-v1.5"))
docs = build_chunk_documents(entry, enc)
q_text = f"Current date: {entry.get('question_date')}\n{q}"
q_vec = enc.encode([q_text], is_query=True)[0]
hits = dense_retrieve(docs, EvolutionState(q0=q_vec, q=q_vec), top_k=30)
rh = select_reader_hits(hits, 10, "single-session-user")
print("reader top", [getattr(h.doc, "session_id", None) or h.doc.id for h in rh[:5]])
print("supports", evidence_supports_answer(rh, entry, q))
mem = review_memory(q_text, hits[:10], llm_complete=None, query_time_days=parse_lme_date(entry.get("question_date")))
print("heuristic covers", mem.covers_question, "missing", mem.missing_keys)
print("note", mem.note[:200])
for i, h in enumerate(hits[:3], 1):
    txt = h.doc.text or ""
    print(f"top{i}", h.doc.id, "assert", value_asserted_in_text(tgt, txt, q), "snip", " ".join(txt.split())[:120])
