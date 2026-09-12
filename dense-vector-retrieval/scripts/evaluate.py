#!/usr/bin/env python3
"""Official IR metrics via pytrec_eval. Compare to paper.claimed_metrics when set."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pytrec_eval

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, load_ir_data, parse_metric, require_meta_match, work_paths  # noqa: E402


def load_run(path: Path) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        qid, _, did, _rank, score, *_ = line.split()
        results[qid][did] = float(score)
    return dict(results)


def hit_at_k(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k: int) -> float:
    scores = []
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = sorted(results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)[:k]
        scores.append(1.0 if any(did in relevant for did, _ in ranked) else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def mrr_at_k(qrels: dict[str, dict[str, int]], results: dict[str, dict[str, float]], k: int) -> float:
    scores = []
    for qid, rels in qrels.items():
        relevant = {d for d, r in rels.items() if r > 0}
        if not relevant:
            continue
        ranked = sorted(results.get(qid, {}).items(), key=lambda kv: kv[1], reverse=True)[:k]
        rr = 0.0
        for rank, (did, _) in enumerate(ranked, start=1):
            if did in relevant:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(qrels, results, k_values: list[int]) -> dict[str, float]:
    measures = {
        f"ndcg_cut.{','.join(map(str, k_values))}",
        f"map_cut.{','.join(map(str, k_values))}",
        f"recall.{','.join(map(str, k_values))}",
        f"P.{','.join(map(str, k_values))}",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(results)
    agg: dict[str, float] = {}
    if not per_query:
        raise SystemExit("No overlap between qrels and run. Check query ids / split.")
    keys = next(iter(per_query.values())).keys()
    for key in keys:
        agg[key] = sum(row[key] for row in per_query.values()) / len(per_query)
    for k in k_values:
        agg[f"mrr_{k}"] = mrr_at_k(qrels, results, k)
        agg[f"hit_{k}"] = hit_at_k(qrels, results, k)
    return agg


def pretty_metrics(agg: dict[str, float], k_values: list[int]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in k_values:
        mapping = {
            f"nDCG@{k}": f"ndcg_cut_{k}",
            f"MAP@{k}": f"map_cut_{k}",
            f"Recall@{k}": f"recall_{k}",
            f"P@{k}": f"P_{k}",
            f"MRR@{k}": f"mrr_{k}",
            f"Hit@{k}": f"hit_{k}",
        }
        for nice, raw in mapping.items():
            if raw in agg:
                out[nice] = round(agg[raw], 4)
    return out


def primary_value(pretty: dict[str, float], primary: str) -> float:
    family, k = parse_metric(primary)
    key = f"{family}@{k}"
    if key not in pretty:
        raise SystemExit(f"Primary {key} missing from metrics: {list(pretty)}")
    return pretty[key]


def compare_paper(cfg: dict, pretty: dict[str, float]) -> tuple[bool, list[str]]:
    claimed = (cfg.get("paper") or {}).get("claimed_metrics") or {}
    claimed = {k: v for k, v in claimed.items() if v is not None}
    if not claimed:
        return True, ["No paper.claimed_metrics provided; skip table comparison."]
    tol_cfg = (cfg.get("paper") or {}).get("tolerance") or {}
    rank_tol = float(tol_cfg.get("ranking", 0.01))
    set_tol = float(tol_cfg.get("set", 0.02))
    ok = True
    lines = []
    for name, target in claimed.items():
        target = float(target)
        if name not in pretty:
            ok = False
            lines.append(f"MISSING {name}: claimed={target} (not computed)")
            continue
        got = pretty[name]
        family, _ = parse_metric(name)
        tol = set_tol if family in {"Recall", "Hit"} else rank_tol
        delta = got - target
        status = "OK" if abs(delta) <= tol else "FAIL"
        if status == "FAIL":
            ok = False
        lines.append(f"{status} {name}: got={got:.4f} paper={target:.4f} delta={delta:+.4f} tol={tol}")
    return ok, lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = work_paths(args.work_dir)
    require_meta_match(cfg, paths["meta"])
    _, _, qrels = load_ir_data(cfg)

    if cfg["eval"].get("ignore_identical_ids"):
        for qid, rels in list(qrels.items()):
            rels.pop(qid, None)

    dataset = cfg["data"]["dataset"]
    run_path = paths["runs"] / f"{dataset}.run"
    if not run_path.exists():
        raise SystemExit(f"Missing run file {run_path}. Run retrieve first.")
    results = load_run(run_path)

    k_values = list(cfg["eval"]["k_values"])
    agg = evaluate(qrels, results, k_values)
    pretty = pretty_metrics(agg, k_values)
    primary = cfg["eval"]["primary"]
    main_score = primary_value(pretty, primary)

    ok, cmp_lines = compare_paper(cfg, pretty)
    payload = {
        "dataset": dataset,
        "split": cfg["data"].get("split"),
        "primary": primary,
        "primary_value": main_score,
        "metrics": pretty,
        "n_eval_queries": len(qrels),
        "paper_comparison": cmp_lines,
        "paper_match": ok,
    }
    paths["metrics"].write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not ok:
        print("\nPAPER MISMATCH. Do not retune. Follow SKILL.md diagnostics in order.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
