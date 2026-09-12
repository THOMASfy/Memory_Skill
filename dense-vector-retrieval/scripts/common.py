"""Shared helpers for the dense-vector-retrieval skill scripts."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

SKILL_ROOT = Path(__file__).resolve().parent.parent

CONTRACT_KEYS = (
    "architecture",
    "model_name_or_path",
    "query_model",
    "doc_model",
    "pooling",
    "normalize",
    "score",
    "query_max_length",
    "doc_max_length",
    "query_prefix",
    "passage_prefix",
    "append_eos",
    "padding_side",
    "dtype",
)


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config is not a mapping: {path}")
    return cfg


def work_paths(work_dir: str | Path) -> dict[str, Path]:
    root = Path(work_dir)
    paths = {
        "root": root,
        "embeddings": root / "embeddings",
        "runs": root / "runs",
        "metrics": root / "metrics.json",
        "meta": root / "embeddings" / "encode_meta.json",
        "corpus": root / "embeddings" / "corpus.npy",
        "queries": root / "embeddings" / "queries.npy",
        "corpus_ids": root / "embeddings" / "corpus_ids.json",
        "query_ids": root / "embeddings" / "query_ids.json",
    }
    paths["embeddings"].mkdir(parents=True, exist_ok=True)
    paths["runs"].mkdir(parents=True, exist_ok=True)
    return paths


def contract_from_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    enc = cfg["encode"]
    data = cfg["data"]
    return {
        "data.dataset": data.get("dataset"),
        "data.split": data.get("split"),
        "data.title_mode": data.get("title_mode"),
        "data.symmetric": bool(data.get("symmetric", False)),
        **{k: enc.get(k) for k in CONTRACT_KEYS},
        "index.type": cfg.get("index", {}).get("type"),
        "index.metric": cfg.get("index", {}).get("metric"),
        "eval.primary": cfg.get("eval", {}).get("primary"),
    }


def fingerprint(contract: dict[str, Any]) -> str:
    blob = json.dumps(contract, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def require_meta_match(cfg: dict[str, Any], meta_path: Path) -> dict[str, Any]:
    if not meta_path.exists():
        raise SystemExit(f"Missing encode meta: {meta_path}. Run encode first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    expected = fingerprint(contract_from_cfg(cfg))
    got = meta.get("fingerprint")
    if got != expected:
        raise SystemExit(
            "encode_meta.json fingerprint does not match current YAML. "
            f"meta={got} config={expected}. Re-encode; do not reuse stale vectors."
        )
    return meta


def compose_document(title: str | None, text: str, title_mode: str) -> str:
    title = (title or "").strip()
    text = (text or "").strip()
    if title_mode == "text_only" or not title:
        return text
    if title_mode == "dpr_sep":
        return f"{title} [SEP] {text}"
    if title_mode == "concat_space":
        return f"{title} {text}".strip()
    raise ValueError(f"Unknown title_mode: {title_mode}")


def _load_jsonl_map(path: Path, id_key: str = "_id") -> dict[str, Any]:
    items: dict[str, Any] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            items[str(obj[id_key])] = obj
    return items


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.strip().split()
            if not parts:
                continue
            if i == 0 and parts[0].lower() in {"query-id", "qid"}:
                continue
            if len(parts) == 3:
                qid, did, rel = parts
            elif len(parts) >= 4:
                qid, _, did, rel = parts[0], parts[1], parts[2], parts[3]
            else:
                raise ValueError(f"Unreadable qrels line: {line!r}")
            qrels.setdefault(str(qid), {})[str(did)] = int(float(rel))
    return qrels


def load_ir_data(cfg: dict[str, Any]) -> tuple[dict[str, dict], dict[str, str], dict[str, dict[str, int]]]:
    """Return corpus {id: {title,text}}, queries {id: text}, qrels {qid: {did: rel}}."""
    data = cfg["data"]
    corpus_path = Path(data["corpus_path"])
    queries_path = Path(data["queries_path"])
    qrels_path = Path(data["qrels_path"])
    for p in (corpus_path, queries_path, qrels_path):
        if not str(p) or not p.exists():
            raise SystemExit(
                f"Missing IR file: {p}. Point data.corpus_path / queries_path / qrels_path "
                "at official BEIR jsonl/tsv (or equivalent)."
            )

    raw_corpus = _load_jsonl_map(corpus_path)
    raw_queries = _load_jsonl_map(queries_path)
    qrels = _load_qrels(qrels_path)

    title_mode = data.get("title_mode") or "concat_space"
    corpus: dict[str, dict] = {}
    for cid, obj in raw_corpus.items():
        corpus[cid] = {
            "title": obj.get("title") or "",
            "text": obj.get("text") or obj.get("contents") or "",
        }
        corpus[cid]["composed"] = compose_document(
            corpus[cid]["title"], corpus[cid]["text"], title_mode
        )

    queries: dict[str, str] = {}
    for qid, obj in raw_queries.items():
        if isinstance(obj, str):
            queries[qid] = obj
        else:
            queries[qid] = obj.get("text") or obj.get("query") or ""

    return corpus, queries, qrels


def parse_metric(name: str) -> tuple[str, int]:
    m = re.match(r"^(nDCG|NDCG|MAP|MRR|Recall|Hit|Accuracy|P)@(\d+)$", str(name).strip())
    if not m:
        raise ValueError(
            f"eval.primary must look like nDCG@10, MRR@10, Recall@100, Hit@20. Got: {name}"
        )
    family = m.group(1)
    if family.upper() == "NDCG":
        family = "nDCG"
    if family == "Accuracy":
        family = "Hit"
    return family, int(m.group(2))
