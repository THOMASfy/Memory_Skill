#!/usr/bin/env python3
"""Encode queries and corpus under the locked contract. Do not change pooling/prefixes here."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, DPRContextEncoder, DPRQuestionEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    contract_from_cfg,
    fingerprint,
    load_config,
    load_ir_data,
    work_paths,
)


def _dtype(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def cls_pool(hidden: torch.Tensor) -> torch.Tensor:
    return hidden[:, 0]


def eos_pool(hidden: torch.Tensor, mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    if padding_side == "left":
        return hidden[:, -1]
    idx = mask.long().sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), idx]


def pool_outputs(arch: str, pooling: str, outputs, mask: torch.Tensor, padding_side: str) -> torch.Tensor:
    if pooling == "cls_pooler":
        if getattr(outputs, "pooler_output", None) is None:
            raise SystemExit("pooling=cls_pooler but model has no pooler_output. DPR requires the official DPR classes.")
        return outputs.pooler_output
    hidden = outputs.last_hidden_state
    if pooling == "cls":
        return cls_pool(hidden)
    if pooling == "mean":
        return mean_pool(hidden, mask)
    if pooling == "eos":
        return eos_pool(hidden, mask, padding_side)
    raise SystemExit(f"Unknown pooling: {pooling}")


class DualEncoder:
    """Not an nn.Module: architecture=auto shares one encoder object on both sides."""

    def __init__(self, query_model, doc_model):
        self.query_model = query_model
        self.doc_model = doc_model

    def eval(self):
        self.query_model.eval()
        if self.doc_model is not self.query_model:
            self.doc_model.eval()
        return self

    def encode_side(self, side: str, **batch):
        model = self.query_model if side == "query" else self.doc_model
        return model(**batch)


def load_models(enc: dict, device: torch.device):
    arch = enc["architecture"]
    dtype = _dtype(enc.get("dtype", "fp32"))
    if arch == "dpr":
        q = DPRQuestionEncoder.from_pretrained(enc["query_model"]).to(device=device, dtype=dtype)
        d = DPRContextEncoder.from_pretrained(enc["doc_model"]).to(device=device, dtype=dtype)
        q_tok = AutoTokenizer.from_pretrained(enc["query_model"])
        d_tok = AutoTokenizer.from_pretrained(enc["doc_model"])
        model = DualEncoder(q, d)
    elif arch == "bi_encoder":
        q = AutoModel.from_pretrained(enc["query_model"]).to(device=device, dtype=dtype)
        d = AutoModel.from_pretrained(enc["doc_model"]).to(device=device, dtype=dtype)
        q_tok = AutoTokenizer.from_pretrained(enc["query_model"])
        d_tok = AutoTokenizer.from_pretrained(enc["doc_model"])
        model = DualEncoder(q, d)
    else:
        inner = AutoModel.from_pretrained(enc["model_name_or_path"]).to(device=device, dtype=dtype)
        tok = AutoTokenizer.from_pretrained(enc["model_name_or_path"])
        q_tok = d_tok = tok
        model = DualEncoder(inner, inner)
    model.eval()
    side = enc.get("padding_side", "right")
    q_tok.padding_side = side
    d_tok.padding_side = side
    if q_tok.pad_token is None:
        q_tok.pad_token = q_tok.eos_token
    if d_tok.pad_token is None:
        d_tok.pad_token = d_tok.eos_token
    return model, q_tok, d_tok


def prepare_texts(texts: list[str], prefix: str, append_eos: bool, tokenizer) -> list[str]:
    out = [f"{prefix}{t}" if prefix else t for t in texts]
    if append_eos:
        eos = tokenizer.eos_token or ""
        out = [x + eos for x in out]
    return out


@torch.no_grad()
def encode_texts(
    model: DualEncoder,
    tokenizer,
    texts: list[str],
    *,
    side: str,
    enc: dict,
    device: torch.device,
) -> np.ndarray:
    max_len = enc["query_max_length"] if side == "query" else enc["doc_max_length"]
    prefix = enc.get("query_prefix") or "" if side == "query" else enc.get("passage_prefix") or ""
    batch_size = int(enc.get("batch_size") or 64)
    pooling = enc["pooling"]
    padding_side = enc.get("padding_side") or "right"
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"encode-{side}"):
        chunk = prepare_texts(texts[i : i + batch_size], prefix, bool(enc.get("append_eos")), tokenizer)
        batch = tokenizer(
            chunk,
            padding=True,
            truncation=True,
            max_length=int(max_len),
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model.encode_side(side, **batch)
        pooled = pool_outputs(enc["architecture"], pooling, outputs, batch["attention_mask"], padding_side)
        if enc.get("normalize"):
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=-1)
        else:
            pooled = pooled.float()
        vectors.append(pooled.cpu().numpy())
    return np.concatenate(vectors, axis=0).astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--work-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = work_paths(args.work_dir)
    corpus, queries, qrels = load_ir_data(cfg)
    enc = cfg["encode"]

    query_ids = sorted(queries.keys())
    # Encode every corpus doc; evaluation uses qrels separately.
    corpus_ids = sorted(corpus.keys())
    query_texts = [queries[i] for i in query_ids]
    if cfg["data"].get("symmetric"):
        # Passages encoded with the query prefix via encode.passage_prefix (validated).
        pass
    corpus_texts = [corpus[i]["composed"] for i in corpus_ids]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, q_tok, d_tok = load_models(enc, device)
    q_vecs = encode_texts(model, q_tok, query_texts, side="query", enc=enc, device=device)
    d_vecs = encode_texts(model, d_tok, corpus_texts, side="doc", enc=enc, device=device)

    np.save(paths["queries"], q_vecs)
    np.save(paths["corpus"], d_vecs)
    paths["query_ids"].write_text(json.dumps(query_ids, ensure_ascii=False), encoding="utf-8")
    paths["corpus_ids"].write_text(json.dumps(corpus_ids, ensure_ascii=False), encoding="utf-8")

    contract = contract_from_cfg(cfg)
    q_norm = float(np.linalg.norm(q_vecs, axis=1).mean())
    d_norm = float(np.linalg.norm(d_vecs, axis=1).mean())
    meta = {
        "fingerprint": fingerprint(contract),
        "contract": contract,
        "n_queries": len(query_ids),
        "n_corpus": len(corpus_ids),
        "n_qrels_queries": len(qrels),
        "dim": int(q_vecs.shape[1]),
        "mean_l2_query": q_norm,
        "mean_l2_corpus": d_norm,
        "device": str(device),
    }
    paths["meta"].write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"ENCODED queries={q_vecs.shape} corpus={d_vecs.shape} "
        f"mean_l2_q={q_norm:.4f} mean_l2_d={d_norm:.4f} fp={meta['fingerprint']}"
    )
    if enc.get("normalize") and (abs(q_norm - 1.0) > 0.02 or abs(d_norm - 1.0) > 0.02):
        print("WARNING: normalize=true but mean L2 is not ~1.0. Pooling/normalize bug.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
