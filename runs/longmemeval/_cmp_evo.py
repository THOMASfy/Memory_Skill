import json
from collections import Counter, defaultdict
from pathlib import Path

def load(p):
    rows = []
    for line in Path(p).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows

evo = load(r"D:\Memory_AS_Skill\Demo4\runs\longmemeval\s_k10_evo.jsonl")
dense = load(r"D:\Memory_AS_Skill\Demo4\runs\longmemeval\s_k10_dense.jsonl")
dmap = {r["question_id"]: r for r in dense}

n = len(evo)
slots = Counter()
r1_slot = Counter()
nround = Counter()
enough = 0
query_n = 0
r1_gold5 = r1_miss5 = 0
salvage = drop10 = keep = worse5 = 0
r1_in10_final_out = 0
qa_both = qa_d_only = qa_e_only = 0
hit10_d_only = hit10_e_only = hit10_both = 0
tvc = Counter()
miss_lab = Counter()
why = Counter()

def gold_rank(ids, gold):
    g = set(gold or [])
    for i, x in enumerate(ids or [], 1):
        if x in g:
            return i
    return None

for r in evo:
    d = dmap.get(r["question_id"], {})
    gold = r.get("gold_session_ids") or []
    abst = r.get("abstention")
    evo_log = r.get("evolution") or []
    nround[len(evo_log)] += 1
    if evo_log:
        r1_slot[evo_log[0].get("slot")] += 1
        note = str(evo_log[0].get("note") or "")
        if "enough" in note:
            enough += 1
        if "missing_query" in note or evo_log[0].get("slot") == "query":
            query_n += 1
        for tok in note.replace(";", " ").split():
            if tok.startswith("tvc="):
                tvc[tok] += 1
            if tok in {"enough", "missing_query_once", "dead_end_rank", "missing_after_query_rank", "max_rounds", "enough=1"}:
                why[tok] += 1
        mk = tuple(evo_log[0].get("missing_keys") or [])
        if mk:
            miss_lab[mk[0]] += 1
        r1t = evo_log[0].get("top") or []
        gr1 = gold_rank(r1t, gold)  # only top5 logged
        # top is top5 only
        if not abst and gold:
            if gr1 is not None:
                r1_gold5 += 1
            else:
                r1_miss5 += 1
    for e in evo_log:
        slots[e.get("slot")] += 1

    if not abst and gold:
        fr = gold_rank(r.get("retrieved_ids"), gold)
        dr = gold_rank(d.get("retrieved_ids"), gold)
        de10 = dr is not None and dr <= 10
        ee10 = fr is not None and fr <= 10
        if de10 and ee10:
            hit10_both += 1
        elif de10:
            hit10_d_only += 1
        elif ee10:
            hit10_e_only += 1
        # r1 top5 miss then final in 5
        g1 = gold_rank(evo_log[0].get("top") if evo_log else [], gold)
        if g1 is None and fr is not None and fr <= 5:
            salvage += 1
        if g1 is not None and (fr is None or fr > 10):
            r1_in10_final_out += 1  # r1 log only top5 so this is r1 in top5 then final out of 10
        if de10 and not ee10:
            drop10 += 1
        if dr is not None and fr is not None and fr > dr:
            worse5 += 1

    dq, eq = d.get("qa_correct"), r.get("qa_correct")
    if dq and eq:
        qa_both += 1
    elif dq and not eq:
        qa_d_only += 1
    elif eq and not dq:
        qa_e_only += 1

out = Path(r"D:\Memory_AS_Skill\Demo4\runs\longmemeval\_cmp_evo.txt")
lines = [
    f"n={n} enough_r1={enough} query_r1={query_n}",
    f"rounds={dict(nround)} r1_slot={dict(r1_slot)} slots={dict(slots)}",
    f"r1 gold in logged top5={r1_gold5} miss_top5={r1_miss5}",
    f"salvage r1miss5->final@5={salvage} r1in5->final_out10={r1_in10_final_out}",
    f"dense@10 only={hit10_d_only} evo@10 only={hit10_e_only} both={hit10_both} drop vs dense@10={drop10}",
    f"qa both={qa_both} dense_only={qa_d_only} evo_only={qa_e_only}",
    f"why={dict(why)}",
    f"tvc={tvc.most_common(15)}",
    f"miss_lab={miss_lab.most_common(12)}",
]
out.write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
