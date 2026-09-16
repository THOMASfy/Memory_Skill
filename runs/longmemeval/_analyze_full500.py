"""Compare evo full500 vs dense: evolution adoption, QA gap, hit_all."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(p: Path) -> list[dict]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def hit_all(retrieved, gold, k):
    if not gold:
        return None
    top = set(retrieved[:k])
    return 1.0 if all(g in top for g in gold) else 0.0


def recall(retrieved, gold, k):
    if not gold:
        return None
    top = set(retrieved[:k])
    return sum(1 for g in gold if g in top) / len(gold)


def gold_rank(retrieved, gold):
    for i, r in enumerate(retrieved):
        if r in gold:
            return i + 1
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else sum(xs) / len(xs)


def main() -> None:
    evo = load(ROOT / "s_k10_evo_full500.jsonl")
    dense = load(ROOT / "s_k10_dense.jsonl")
    print("n evo", len(evo), "n dense", len(dense))

    slot_ctr: Counter = Counter()
    slot_any: Counter = Counter()
    r1_slots: Counter = Counter()
    rounds_n = []
    query_focus = 0
    filter_bind = 0
    rank_non_id = 0
    gold_up = gold_down = gold_same = gold_late = 0
    r1_miss_final_hit = 0
    salvage = []
    suf_true = suf_false = 0
    abstain = idk = qa_t = qa_f = 0
    by_type_slots: dict = defaultdict(Counter)
    empty_evo = 0

    for r in evo:
        evo_log = r.get("evolution") or []
        if not evo_log:
            empty_evo += 1
        rounds_n.append(len(evo_log))
        slots_used = set()
        for lg in evo_log:
            s = lg.get("slot")
            if s:
                slot_ctr[s] += 1
                slots_used.add(s)
        for s in slots_used:
            slot_any[s] += 1
        if evo_log:
            r1_slots[evo_log[0].get("slot")] += 1
            by_type_slots[r.get("question_type")][evo_log[0].get("slot")] += 1
            gold = set(r.get("gold_session_ids") or [])
            r1_ids = [x.split("::")[0] for x in (evo_log[0].get("top") or [])]
            final = r.get("retrieved_ids") or []
            r1_hit = any(g in r1_ids for g in gold)
            fin_hit = any(g in final[:10] for g in gold)
            r1r = gold_rank(r1_ids, gold)
            fr = gold_rank(final, gold)
            if not r1_hit and fin_hit:
                r1_miss_final_hit += 1
                salvage.append((r["question_id"], r.get("question_type"), r1r, fr))
            if r1r and fr:
                if fr < r1r:
                    gold_up += 1
                elif fr > r1r:
                    gold_down += 1
                else:
                    gold_same += 1
            elif not r1r and fr:
                gold_late += 1

        sq = r.get("skill_query_slot") or ""
        if "focus:" in sq:
            query_focus += 1
        if "require_bind" in (r.get("skill_filter_slot") or ""):
            filter_bind += 1
        sr = r.get("skill_rank_slot") or ""
        if sr and "恒等" not in sr and "score(d) = sim(q, d)" not in sr:
            rank_non_id += 1

        if r.get("evo_sufficient") is True:
            suf_true += 1
        elif r.get("evo_sufficient") is False:
            suf_false += 1
        if r.get("reader_abstained"):
            abstain += 1
        hyp = (r.get("hypothesis") or "").strip().lower()
        if hyp.startswith("i do not know"):
            idk += 1
        if r.get("qa_correct") is True:
            qa_t += 1
        elif r.get("qa_correct") is False:
            qa_f += 1

    print("\n=== EVOLUTION ADOPTION ===")
    print("empty evolution logs:", empty_evo)
    print("slot updates total:", dict(slot_ctr))
    print("questions with any slot:", dict(slot_any), "of", len(evo))
    print("r1 slot:", dict(r1_slots))
    print("rounds mean", round(mean(rounds_n) or 0, 3), "dist", dict(Counter(rounds_n)))
    print("skill_query focus:", query_focus)
    print("skill_filter require_bind:", filter_bind)
    print("skill_rank non-identity:", rank_non_id)
    print("evo_sufficient T/F", suf_true, suf_false)
    print("gold r1->final up/down/same/late_in", gold_up, gold_down, gold_same, gold_late)
    print("r1 top miss -> final top10 hit:", r1_miss_final_hit)
    print("sample salvage", salvage[:10])

    ed = {r["question_id"]: r for r in dense}
    ee = {r["question_id"]: r for r in evo}
    common = sorted(set(ed) & set(ee))
    print("\npaired", len(common))

    better_r10 = worse_r10 = same_r10 = 0
    better_r5 = worse_r5 = 0
    evo_only_hit = []
    dense_only_hit = []
    abs_with_r10 = idk_with_r10 = 0
    ok_with_r10 = wrong_with_r10 = 0
    gate_vs_llm: Counter = Counter()
    evo_qa_down = []

    for qid in common:
        a, b = ee[qid], ed[qid]
        if a.get("abstention") or b.get("abstention"):
            continue
        er10, dr10 = a.get("recall@10"), b.get("recall@10")
        er5, dr5 = a.get("recall@5"), b.get("recall@5")
        if er10 is None or dr10 is None:
            continue
        if er10 > dr10 + 1e-9:
            better_r10 += 1
        elif er10 < dr10 - 1e-9:
            worse_r10 += 1
        else:
            same_r10 += 1
        if er5 is not None and dr5 is not None:
            if er5 > dr5 + 1e-9:
                better_r5 += 1
            elif er5 < dr5 - 1e-9:
                worse_r5 += 1
        eg = set(a.get("gold_session_ids") or [])
        if eg:
            eh = any(g in (a.get("retrieved_ids") or [])[:10] for g in eg)
            dh = any(g in (b.get("retrieved_ids") or [])[:10] for g in eg)
            if eh and not dh:
                evo_only_hit.append(qid)
            if dh and not eh:
                dense_only_hit.append(qid)
        if er10 is not None and er10 >= 0.999:
            hyp = (a.get("hypothesis") or "").strip().lower()
            if a.get("reader_abstained"):
                abs_with_r10 += 1
                gate_vs_llm["abstain_gate"] += 1
            elif hyp.startswith("i do not know"):
                idk_with_r10 += 1
                gate_vs_llm["reader_idk"] += 1
            elif a.get("qa_correct"):
                gate_vs_llm["correct"] += 1
            else:
                gate_vs_llm["wrong_answer"] += 1
            if a.get("qa_correct"):
                ok_with_r10 += 1
            else:
                wrong_with_r10 += 1
            if (a.get("qa_correct") is False) and (b.get("qa_correct") is True) and er10 >= (dr10 or 0):
                evo_qa_down.append(qid)

    print("\n=== PAIRED RETRIEVAL ===")
    print("R@10 better/worse/same", better_r10, worse_r10, same_r10)
    print("R@5 better/worse", better_r5, worse_r5)
    print("evo-only gold@10", len(evo_only_hit), "dense-only", len(dense_only_hit))
    print("sample evo-only", evo_only_hit[:10])

    print("\n=== QA vs RET when evo R@10=1 ===")
    print("ok/wrong", ok_with_r10, wrong_with_r10)
    print("abstain_gate / reader_idk", abs_with_r10, idk_with_r10)
    print("breakdown", dict(gate_vs_llm))
    print("evo ret>=dense but qa worse vs dense", len(evo_qa_down))

    # dense hit_all
    print("\n=== DENSE HIT_ALL (recomputed) ===")
    d_hit5, d_hit10, d_rec5, d_rec10 = [], [], [], []
    d_any5, d_any10 = [], []
    d_hit10_t: dict = defaultdict(list)
    d_hit5_t: dict = defaultdict(list)
    for r in dense:
        if r.get("abstention"):
            continue
        ret = r.get("retrieved_ids") or []
        gold = r.get("gold_session_ids") or []
        if not gold:
            continue
        h5, h10 = hit_all(ret, gold, 5), hit_all(ret, gold, 10)
        r5, r10 = recall(ret, gold, 5), recall(ret, gold, 10)
        d_hit5.append(h5)
        d_hit10.append(h10)
        d_rec5.append(r5)
        d_rec10.append(r10)
        d_any5.append(1.0 if any(g in ret[:5] for g in gold) else 0.0)
        d_any10.append(1.0 if any(g in ret[:10] for g in gold) else 0.0)
        qt = r.get("question_type")
        d_hit10_t[qt].append(h10)
        d_hit5_t[qt].append(h5)

    print(f"dense n={len(d_rec10)}")
    print(f"Recall@5={mean(d_rec5):.4f} Recall@10={mean(d_rec10):.4f}")
    print(f"Hit-all@5={mean(d_hit5):.4f} Hit-all@10={mean(d_hit10):.4f}")
    print(f"Any-gold Hit@5={mean(d_any5):.4f} Hit@10={mean(d_any10):.4f}")
    print("Hit-all@10 by type:")
    for t, v in sorted(d_hit10_t.items()):
        print(f"  {t}: {mean(v):.4f} n={len(v)}")
    print("Hit-all@5 by type:")
    for t, v in sorted(d_hit5_t.items()):
        print(f"  {t}: {mean(v):.4f} n={len(v)}")

    e_hit5, e_hit10, e_any5, e_any10 = [], [], [], []
    e_hit10_t: dict = defaultdict(list)
    for r in evo:
        if r.get("abstention"):
            continue
        ret = r.get("retrieved_ids") or []
        gold = r.get("gold_session_ids") or []
        if not gold:
            continue
        e_hit5.append(hit_all(ret, gold, 5))
        e_hit10.append(hit_all(ret, gold, 10))
        e_any5.append(1.0 if any(g in ret[:5] for g in gold) else 0.0)
        e_any10.append(1.0 if any(g in ret[:10] for g in gold) else 0.0)
        e_hit10_t[r.get("question_type")].append(hit_all(ret, gold, 10))
    print("\n=== EVO HIT_ALL ===")
    print(f"Hit-all@5={mean(e_hit5):.4f} Hit-all@10={mean(e_hit10):.4f}")
    print(f"Any-gold Hit@5={mean(e_any5):.4f} Hit@10={mean(e_any10):.4f}")
    print("Hit-all@10 by type:")
    for t, v in sorted(e_hit10_t.items()):
        print(f"  {t}: {mean(v):.4f} n={len(v)}")

    print("\n=== EVO QA FAILURE MODES ===")
    print("qa T/F", qa_t, qa_f, "idk", idk, "abstained", abstain, "suf T/F", suf_true, suf_false)
    fail_modes: Counter = Counter()
    fail_by_type: Counter = Counter()
    pref_fail = 0
    for r in evo:
        if r.get("abstention"):
            continue
        if not (r.get("recall@10") or 0) >= 0.999:
            continue
        if r.get("qa_correct"):
            continue
        hyp = (r.get("hypothesis") or "").strip().lower()
        if r.get("reader_abstained"):
            fail_modes["gate_abstain"] += 1
        elif hyp.startswith("i do not know"):
            fail_modes["reader_idk"] += 1
        else:
            fail_modes["answered_wrong"] += 1
        fail_by_type[r.get("question_type")] += 1
        if r.get("question_type") == "single-session-preference":
            pref_fail += 1
    print("among R@10~1 qa=false:", dict(fail_modes))
    print("fail by type:", dict(fail_by_type), "pref", pref_fail)

    # also count R@10=1 but hit_all=0 (multi-gold partial)
    partial = 0
    for r in evo:
        if r.get("abstention"):
            continue
        if (r.get("recall@10") or 0) > 0 and (r.get("hit_all@10") or 0) < 1:
            if not r.get("qa_correct"):
                partial += 1
    print("qa=false with partial recall (hit_all@10=0, recall>0):", partial)

    print("\nr1 slots by type:")
    for t, c in sorted(by_type_slots.items()):
        print(f"  {t}: {dict(c)}")

    print("\nQA by type evo vs dense:")
    for t in sorted({r["question_type"] for r in evo}):
        ea = [int(bool(r["qa_correct"])) for r in evo if r["question_type"] == t and r.get("qa_correct") is not None]
        da = [int(bool(r["qa_correct"])) for r in dense if r["question_type"] == t and r.get("qa_correct") is not None]
        print(f"  {t}: evo={sum(ea)/len(ea):.3f}({len(ea)}) dense={sum(da)/len(da):.3f}({len(da)})")

    imp_qa_up = imp_qa_down = imp_qa_same = 0
    for qid in evo_only_hit:
        a, b = ee[qid], ed[qid]
        if a.get("qa_correct") is None:
            continue
        if bool(a["qa_correct"]) and not bool(b.get("qa_correct")):
            imp_qa_up += 1
        elif (not bool(a["qa_correct"])) and bool(b.get("qa_correct")):
            imp_qa_down += 1
        else:
            imp_qa_same += 1
    print("evo-only salvage QA up/down/same", imp_qa_up, imp_qa_down, imp_qa_same)

    # suf=True but IDK / suf=False but answered
    suf_idk = suf_ok = nosuf_idk = nosuf_ok = 0
    for r in evo:
        hyp = (r.get("hypothesis") or "").strip().lower()
        is_idk = hyp.startswith("i do not know") or bool(r.get("reader_abstained"))
        if r.get("evo_sufficient") is True:
            if is_idk:
                suf_idk += 1
            elif r.get("qa_correct"):
                suf_ok += 1
        elif r.get("evo_sufficient") is False:
            if is_idk:
                nosuf_idk += 1
            elif r.get("qa_correct"):
                nosuf_ok += 1
    print("suf=True -> idk/correct", suf_idk, suf_ok)
    print("suf=False -> idk/correct", nosuf_idk, nosuf_ok)

    # check whether gold answers appear in skill slots (cheat sniff)
    # no gold answer field in jsonl; sniff pasted session text
    long_slots = sum(
        1
        for r in evo
        if len((r.get("skill_query_slot") or "") + (r.get("skill_filter_slot") or "")) > 2500
    )
    print("long skill slots (>2500 chars):", long_slots)


if __name__ == "__main__":
    main()
