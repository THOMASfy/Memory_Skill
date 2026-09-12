#!/usr/bin/env python3
"""Exact dense retrieval. Flat inner product only in reproduce mode."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, require_meta_match, work_paths  # noqa: E402


def write_trec(path: Path, results: dict[str, dict[str, float]], tag: str = "dense") -> None:
    lines = []
    for qid, scores in results.items():
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        for rank, (did, score) in enumerate(ranked, start=1):
            lines.append(f"{qid} Q0 {did} {rank} {score:.6f} {tag}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def exact_search(
    queries: np.ndarray,
    corpus: np.ndarray,
    query_ids: list[str],
    corpus_ids: list[str],
    top_k: int,
    chunk: int = 4096,
) -> dict[str, dict[str, float]]:
    """Inner product search. For L2-normalized vectors this is cosine."""
    nq, nd = queries.shape[0], corpus.shape[0]
    top_k = min(top_k, nd)
    results: dict[str, dict[str, float]] = {qid: {} for qid in query_ids}

    # Keep a running top-k per query via chunked matmul to bound memory.
    best_scores = np.full((nq, top_k), -np.inf, dtype=np.float32)
    best_idx = np.full((nq, top_k), -1, dtype=np.int64)
    for start in tqdm(range(0, nd, chunk), desc="retrieve"):
        block = corpus[start : start + chunk]
        scores = queries @ block.T  # [nq, b]
        # merge with current best
        cand_scores = np.concatenate([best_scores, scores], axis=1)
        cand_idx = np.concatenate(
            [best_idx, np.broadcast_to(np.arange(start, start + block.shape[0]), scores.shape)],
            axis=1,
        )
        part = np.argpartition(cand_scores, -top_k, axis=1)[:, -top_k:]
        row = np.arange(nq)[:, None]
        best_scores = cand_scores[row, part]
        best_idx = cand_idx[row, part]
        # sort each row
        order = np.argsort(-best_scores, axis=1)
        best_scores = np.take_along_axis(best_scores, order, axis=1)
        best_idx = np.take_along_axis(best_idx, order, axis=1)

    for i, qid in enumerate(query_ids):
        for j in range(top_k):
            di = int(best_idx[i, j])
            if di < 0:
                continue
            results[qid][corpus_ids[di]] = float(best_scores[i, j])
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = work_paths(args.work_dir)
    require_meta_match(cfg, paths["meta"])

    idx_type = cfg.get("index", {}).get("type", "flat")
    if cfg["task"].get("mode", "reproduce") == "reproduce" and idx_type != "flat":
        raise SystemExit("reproduce mode forbids ANN. Set index.type=flat.")
    if idx_type != "flat":
        raise SystemExit(
            f"index.type={idx_type} is not implemented in this skill on purpose. "
            "Use flat for paper-comparable numbers."
        )

    queries = np.load(paths["queries"])
    corpus = np.load(paths["corpus"])
    query_ids = json.loads(paths["query_ids"].read_text(encoding="utf-8"))
    corpus_ids = json.loads(paths["corpus_ids"].read_text(encoding="utf-8"))
    k_values = list(cfg["eval"].get("k_values") or [10])
    top_k = max(k_values)

    results = exact_search(queries, corpus, query_ids, corpus_ids, top_k=top_k)

    if cfg["eval"].get("ignore_identical_ids"):
        for qid, scores in results.items():
            scores.pop(qid, None)

    dataset = cfg["data"]["dataset"]
    run_path = paths["runs"] / f"{dataset}.run"
    write_trec(run_path, results)
    (paths["runs"] / f"{dataset}.results.json").write_text(
        json.dumps(results), encoding="utf-8"
    )
    print(f"RETRIEVED queries={len(results)} top_k={top_k} run={run_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
