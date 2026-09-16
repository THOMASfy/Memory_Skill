import json
from collections import Counter
from pathlib import Path

p = Path(r"D:\Memory_AS_Skill\Demo4\runs\longmemeval\s_k10_evo.jsonl")
rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
out = Path(r"D:\Memory_AS_Skill\Demo4\runs\longmemeval\_inspect_qa.txt")
lines = []
qa = Counter(str(r.get("qa_correct")) for r in rows)
lines.append(f"n={len(rows)} qa={dict(qa)}")
# judge prefixes
jv = Counter()
hyp_need = 0
gold_hit10_false = 0
gold_hit10_true = 0
has_json_true = 0
has_json_false = 0
no_json = 0
slots = Counter()
stop_r = Counter()
miss_empty_stop = 0
query_then_worse = 0
for r in rows:
    v = (r.get("judge_verdict") or "")[:80].replace("\n", " | ")
    jv[v[:40]] += 1
    h = r.get("hypothesis") or ""
    if h.lower().startswith("we need") or "we need" in h[:80].lower():
        hyp_need += 1
    vfull = r.get("judge_verdict") or ""
    if '"correct"' in vfull.lower() or '"correct"' in vfull:
        if "true" in vfull.lower() and "false" not in vfull.lower()[:80]:
            has_json_true += 1
        elif "false" in vfull.lower():
            has_json_false += 1
    else:
        no_json += 1
    gold = set(r.get("gold_session_ids") or [])
    ids = r.get("retrieved_ids") or []
    hit = bool(gold) and any(x in gold for x in ids[:10])
    if hit and r.get("qa_correct") is False:
        gold_hit10_false += 1
    if hit and r.get("qa_correct") is True:
        gold_hit10_true += 1
    evo = r.get("evolution") or []
    stop_r[len(evo)] += 1
    for e in evo:
        slots[e.get("slot")] += 1
    if evo and not (evo[-1].get("missing_keys") or []):
        miss_empty_stop += 1

lines.append(f"hyp_starts_we_need={hyp_need}")
lines.append(f"gold@10 & qa false={gold_hit10_false} true={gold_hit10_true}")
lines.append(f"judge json-ish true={has_json_true} false={has_json_false} no_json={no_json}")
lines.append(f"rounds dist={dict(stop_r)}")
lines.append(f"slots={dict(slots)}")
lines.append(f"last_round missing empty={miss_empty_stop}")
lines.append("--- judge prefix ---")
for k, c in jv.most_common(12):
    lines.append(f"  {c:4d} {k!r}")
# 8 examples: gold hit, qa false
lines.append("--- examples gold@10 qa=false ---")
n = 0
for r in rows:
    gold = set(r.get("gold_session_ids") or [])
    ids = r.get("retrieved_ids") or []
    if not (gold and any(x in gold for x in ids[:10]) and r.get("qa_correct") is False):
        continue
    n += 1
    if n > 8:
        break
    rank = next((i + 1 for i, x in enumerate(ids) if x in gold), None)
    lines.append(f"id={r.get('question_id')} type={r.get('question_type')} gold_rank={rank} R10={r.get('recall@10')}")
    lines.append(f"  Q: {(r.get('question') or '')[:120]}")
    lines.append(f"  hyp: {(r.get('hypothesis') or '')[:220].replace(chr(10),' / ')}")
    lines.append(f"  judge: {(r.get('judge_verdict') or '')[:220].replace(chr(10),' / ')}")
    for e in r.get("evolution") or []:
        lines.append(
            f"  r{e.get('round')} slot={e.get('slot')} miss={e.get('missing_keys')} top={e.get('top')}"
        )
        lines.append(f"    note={(e.get('note') or '')[:180]}")

# injection: r1 miss vs r2 rank of gold
lines.append("--- injection salvage (r1 gold not in top5, later in top5) ---")
salv_q = salv_r = 0
worse = 0
same = 0
for r in rows:
    gold = set(r.get("gold_session_ids") or [])
    evo = r.get("evolution") or []
    if not gold or len(evo) < 2:
        continue
    def gr(top):
        return next((i + 1 for i, x in enumerate(top or []) if x in gold), 99)
    r1, r2 = gr(evo[0].get("top")), gr(evo[1].get("top"))
    slot = evo[0].get("slot")
    if r1 > 5 and r2 <= 5:
        if slot == "query":
            salv_q += 1
        else:
            salv_r += 1
    elif r2 > r1:
        worse += 1
    else:
        same += 1
lines.append(f"salvage QUERY={salv_q} RANK={salv_r} later_worse={worse} later_same_or_better={same}")
out.write_text("\n".join(lines), encoding="utf-8")
print("wrote", out)
