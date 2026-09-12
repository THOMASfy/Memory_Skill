#!/usr/bin/env python3
"""Fail fast if the encoding/eval contract is incomplete or inconsistent."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, parse_metric  # noqa: E402

REQUIRED_TOP = ("task", "paper", "data", "encode", "index", "eval")
REQUIRED_DATA = ("benchmark", "dataset", "split", "title_mode", "corpus_path", "queries_path", "qrels_path")
REQUIRED_ENCODE = (
    "architecture",
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

VALID_POOLING = {"cls", "cls_pooler", "mean", "eos"}
VALID_SCORE = {"ip", "cos_sim"}
VALID_ARCH = {"auto", "dpr", "bi_encoder"}
VALID_TITLE = {"concat_space", "dpr_sep", "text_only"}
VALID_INDEX = {"flat", "hnsw", "ivf"}

# Each rule: (predicate(model_ids) -> bool, expected encode/data fields)
FAMILY_RULES: list[tuple[str, callable, dict]] = []


def _ids(enc: dict) -> str:
    parts = [
        str(enc.get("model_name_or_path") or ""),
        str(enc.get("query_model") or ""),
        str(enc.get("doc_model") or ""),
    ]
    return " ".join(parts).lower()


def _add(name: str, pred, expected: dict) -> None:
    FAMILY_RULES.append((name, pred, expected))


_add(
    "dpr",
    lambda s: "dpr-question_encoder" in s or "dpr-ctx_encoder" in s or "facebook-dpr-" in s,
    {
        "architecture": "dpr",
        "pooling": "cls_pooler",
        "normalize": False,
        "score": "ip",
        "query_prefix": "",
        "passage_prefix": "",
        "title_mode": "dpr_sep",
    },
)
_add(
    "contriever",
    lambda s: "facebook/contriever" in s or s.endswith("contriever") or "/contriever-" in s,
    {
        "pooling": "mean",
        "normalize": False,
        "score": "ip",
        "query_prefix": "",
        "passage_prefix": "",
    },
)
_add(
    "e5-instruct",
    lambda s: "e5-mistral" in s or ("e5-" in s and "instruct" in s),
    {"pooling": "eos", "normalize": True, "score": "cos_sim", "append_eos": True, "passage_prefix": ""},
)
_add(
    "e5",
    lambda s: (
        ("intfloat/e5-" in s or "multilingual-e5" in s or "/e5-small" in s or "/e5-base" in s or "/e5-large" in s)
        and "instruct" not in s
        and "mistral" not in s
    ),
    {
        "pooling": "mean",
        "normalize": True,
        "score": "cos_sim",
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    },
)
_add(
    "bge-m3",
    lambda s: "bge-m3" in s,
    {"pooling": "cls", "normalize": True, "score": "cos_sim"},
)
_add(
    "bge",
    lambda s: "bge-" in s and "bge-m3" not in s,
    {"pooling": "cls", "normalize": True, "score": "cos_sim", "passage_prefix": ""},
)
_add(
    "tas-b",
    lambda s: "tas-b" in s,
    {"pooling": "cls", "normalize": False, "score": "ip", "query_prefix": "", "passage_prefix": ""},
)


def _missing(obj: dict, keys: tuple[str, ...], prefix: str) -> list[str]:
    out = []
    for k in keys:
        if k not in obj or obj[k] is None:
            out.append(f"{prefix}.{k}")
    return out


def _norm_prefix(value) -> str:
    if value is None:
        return ""
    return str(value)


def validate(cfg: dict) -> list[str]:
    errors: list[str] = []
    for k in REQUIRED_TOP:
        if k not in cfg:
            errors.append(f"missing section: {k}")
    if errors:
        return errors

    errors.extend(_missing(cfg["data"], REQUIRED_DATA, "data"))
    errors.extend(_missing(cfg["encode"], REQUIRED_ENCODE, "encode"))
    for path_key in ("corpus_path", "queries_path", "qrels_path"):
        if not str(cfg["data"].get(path_key) or "").strip():
            errors.append(f"data.{path_key} is empty; point it at official corpus/queries/qrels")

    data, enc, index, ev, task = cfg["data"], cfg["encode"], cfg["index"], cfg["eval"], cfg["task"]

    if data.get("title_mode") not in VALID_TITLE:
        errors.append(f"data.title_mode must be one of {sorted(VALID_TITLE)}")
    if enc.get("pooling") not in VALID_POOLING:
        errors.append(f"encode.pooling must be one of {sorted(VALID_POOLING)}")
    if enc.get("score") not in VALID_SCORE:
        errors.append(f"encode.score must be one of {sorted(VALID_SCORE)}")
    if enc.get("architecture") not in VALID_ARCH:
        errors.append(f"encode.architecture must be one of {sorted(VALID_ARCH)}")
    if index.get("type") not in VALID_INDEX:
        errors.append(f"index.type must be one of {sorted(VALID_INDEX)}")

    arch = enc.get("architecture")
    if arch in {"dpr", "bi_encoder"}:
        if not enc.get("query_model") or not enc.get("doc_model"):
            errors.append("dpr/bi_encoder requires encode.query_model and encode.doc_model")
    else:
        if not enc.get("model_name_or_path"):
            errors.append("encode.model_name_or_path is required for architecture=auto")

    if enc.get("normalize") and enc.get("score") == "ip":
        # ranking-equivalent; allowed but metric field should stay consistent
        pass
    if (not enc.get("normalize")) and enc.get("score") == "cos_sim":
        errors.append(
            "encode.normalize=false with score=cos_sim will not match DPR/Contriever papers. "
            "Use score=ip, or normalize both query and corpus."
        )

    if enc.get("pooling") == "eos" and enc.get("padding_side") != "left":
        # right padding still works if we pick last non-pad; warn not error
        pass

    mode = task.get("mode", "reproduce")
    if mode == "reproduce" and index.get("type") != "flat":
        errors.append(
            "task.mode=reproduce requires index.type=flat (exact search). "
            "ANN results are not comparable to paper tables."
        )

    if data.get("symmetric") and enc.get("passage_prefix") and enc.get("passage_prefix") != enc.get("query_prefix"):
        errors.append(
            "data.symmetric=true but passage_prefix != query_prefix. "
            "Quora-style tasks must encode passages with the query prefix."
        )

    try:
        family, k = parse_metric(ev.get("primary", "nDCG@10"))
        if k not in set(ev.get("k_values") or []):
            errors.append(f"eval.primary k={k} is not in eval.k_values")
        _ = family
    except ValueError as e:
        errors.append(str(e))

    override = bool(enc.get("paper_override", False))
    model_blob = _ids(enc)
    matched = None
    for name, pred, expected in FAMILY_RULES:
        if pred(model_blob):
            matched = name
            if not override:
                for field, want in expected.items():
                    if field == "title_mode":
                        got = data.get(field)
                    else:
                        got = enc.get(field)
                    if field in {"query_prefix", "passage_prefix"}:
                        if _norm_prefix(got) != _norm_prefix(want):
                            errors.append(
                                f"family {name}: {field} must be {want!r} (got {got!r}). "
                                "Copy the official string, including trailing spaces. "
                                "Set encode.paper_override=true only if the paper's official script differs."
                            )
                    elif got != want:
                        errors.append(
                            f"family {name}: {field} expected {want!r}, got {got!r}. "
                            "Set encode.paper_override=true only if the paper's official script differs."
                        )
            break

    if matched is None and not override and mode == "reproduce":
        errors.append(
            "Unknown model family. Read the model card Usage code, fill pooling/normalize/prefixes, "
            "then set encode.paper_override=true. Do not guess cosine+mean."
        )

    # BGE retrieval papers usually include the English s2p instruction.
    if matched == "bge" and not override:
        qp = _norm_prefix(enc.get("query_prefix"))
        if qp not in {
            "Represent this sentence for searching relevant passages: ",
            "为这个句子生成表示以用于检索相关文章：",
            "",
        }:
            errors.append(
                "BGE query_prefix must be the official English/Chinese instruction or empty "
                "(v1.5 convenience). Trailing space on the English string is required."
            )
        if mode == "reproduce" and qp == "":
            errors.append(
                "Reproducing BGE tables: query_prefix is empty. Most paper numbers used the "
                "official instruction. Confirm the paper, then paper_override=true if they omitted it."
            )

    if matched == "e5" and data.get("symmetric"):
        if _norm_prefix(enc.get("passage_prefix")) != "query: " and not override:
            errors.append(
                "E5 symmetric datasets (Quora) must use passage_prefix='query: ' "
                "(encode passages as queries). See official e5/mteb_beir_eval.py."
            )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    errors = validate(cfg)
    if errors:
        print("CONFIG INVALID:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("CONFIG OK")
    print(f"  mode={cfg['task'].get('mode')} dataset={cfg['data'].get('dataset')} "
          f"model={cfg['encode'].get('model_name_or_path') or cfg['encode'].get('query_model')} "
          f"pooling={cfg['encode']['pooling']} normalize={cfg['encode']['normalize']} "
          f"score={cfg['encode']['score']} index={cfg['index']['type']} "
          f"primary={cfg['eval']['primary']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
