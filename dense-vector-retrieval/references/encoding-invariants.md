# Encoding invariants

这些不变量是脚本必须遵守的实现细节。Agent 不要为了“简化”而改。

## 1. Attention-mask mean pooling

```python
hidden = last_hidden_state  # [B, T, D]
mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
summed = (hidden * mask).sum(dim=1)
denom = mask.sum(dim=1).clamp(min=1e-9)
embedding = summed / denom
```

用 `hidden.mean(dim=1)` 会把 pad 当真实 token，batch 越大、短句越多，向量越歪。

## 2. CLS vs DPR pooler

- `cls`：`last_hidden_state[:, 0]`（BGE）。
- `cls_pooler`：`pooler_output`（Facebook DPR：Linear + Tanh）。两者不可互换。

## 3. EOS / last non-pad

```python
# right padding
idx = attention_mask.sum(dim=1) - 1
embedding = last_hidden_state[torch.arange(B), idx]
```

Decoder-only instruct 模型常用 **left padding**，此时最后一个 token 在 `[:, -1]`，且 tokenizer 必须 `padding_side='left'`。配置里 `encode.padding_side: left` 与 `pooling: eos` 一起出现。

`append_eos: true` 时，在 truncate 前把 eos token id 接到序列末尾（与 E5-Mistral 官方一致）。

## 4. Prefix 在 tokenize 之前拼接

顺序：取 raw 文本 → 按 title_mode 合成 → 加 prefix → tokenizer(`truncation=True`, `max_length=...`)。

错误顺序：先截断到 512 再加一长段 instruction，等于把正文挤掉。

## 5. Title

- `concat_space`：`f"{title} {text}".strip()`（BEIR）。
- `dpr_sep`：`f"{title} [SEP] {text}"`。空 title 则只有 text（不要留下 ` [SEP] `）。
- `text_only`：丢弃 title。仅当论文明确忽略 title。

## 6. 归一化

```python
embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
```

query 与 corpus **同时**开或 **同时**关。检索端：

- 已归一：`score = q @ D.T`（cosine ≡ IP）
- 未归一：`score = q @ D.T`（纯内积）。不要再除以范数。

FAISS：归一化后用 `IndexFlatIP`；未归一化也用 `IndexFlatIP`。不要对未归一化向量用 `IndexFlatL2` 去冒充论文的 IP 排序。

## 7. 精度

复现默认 `dtype: fp32`。仅当论文/官方脚本用 bf16/fp16（大 LLM embedder）才改。先 fp16 再 normalize 会有微小差异；对 7B+ 模型按官方 `torch_dtype`。

## 8. Batch 确定性

- `model.eval()`，关掉 dropout。
- 同一文本单独 encode 与在 batch 里 encode（有 padding）在 **正确 mask pooling** 下应接近 fp32 误差。若差异大，说明 pooling 没 mask。
- 不要 `pad_to_max_length` 到模型上限，除非官方脚本这么做。pad 到 batch 内最长即可。

## 9. 双塔

Query 只用 query encoder，文档只用 ctx encoder。用同一个 BERT 编两边去对 DPR 表，数字会差一截。

## 10. 契约指纹

`encode_meta.json` 写入规范化后的 12 字段。retrieve / evaluate 发现指纹与当前 YAML 不一致必须退出。改了 prefix 却复用旧 `corpus.npy` 是静默污染。
