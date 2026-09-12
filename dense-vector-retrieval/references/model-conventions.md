# Model family conventions

Only use a row when the paper / 模型卡 **没有写全** 12 字段。论文或官方 eval 脚本与本表冲突时，**以论文/官方脚本为准**，并设 `encode.paper_override: true`。

Prefix 必须逐字使用，含末尾空格。

| Family | Match | architecture | pooling | normalize | score | query_prefix | passage_prefix | title_mode | max_length |
|--------|-------|--------------|---------|-----------|-------|--------------|----------------|------------|------------|
| DPR | `dpr-question_encoder` / `dpr-ctx_encoder` / `facebook-dpr-` | `dpr` | `cls_pooler` | false | `ip` | `""` | `""` | `dpr_sep` | 256/256 除非论文另写 |
| Contriever | `facebook/contriever` | `auto` | `mean` | false | `ip` | `""` | `""` | `concat_space` | 512 |
| E5 encoder | `intfloat/e5-small/base/large` 及 `-v2` | `auto` | `mean` | true | `cos_sim` | `query: ` | `passage: ` | `concat_space` | 512 |
| E5 instruct | `e5-mistral` / `e5-*-instruct` | `auto` | `eos` | true | `cos_sim` | 见下方 instruct 模板 | `""` | `concat_space` | 论文值（常 512） |
| BGE v1 / v1.5 | `BAAI/bge-*-en` `bge-*-zh` 非 m3 | `auto` | `cls` | true | `cos_sim` | 见下方 BGE 指令 | `""` | `concat_space` | 512 |
| BGE-M3 dense | `BAAI/bge-m3` | `auto` | `cls` | true | `cos_sim` | 官方 instruction 或 `""` | `""` | `concat_space` | 8192 |
| TAS-B | `msmarco-distilbert-base-tas-b` | `auto` | `cls` | false | `ip` | `""` | `""` | `concat_space` | 512 |
| SBERT symmetric | `all-mpnet-base-v2` / `all-MiniLM-*` | `auto` | `mean` | true | `cos_sim` | `""` | `""` | `concat_space` | 256 或模型卡 |

未知家族：不要填本表。去模型卡 Usage 代码里抄 pooling / normalize / prompt，再设 `paper_override: true`。

## Prefix 原文

### E5（非 instruct）

```
query: 
passage: 
```

`query:` 后面有一个空格。Quora 等对称任务：passage 也用 `query: `（`data.symmetric: true`）。

### E5 instruct / E5-Mistral

论文与官方脚本使用 task instruction。检索默认：

```
Instruct: Given a question, retrieve relevant documents that best answer the question
Query: 
```

passage 不加 prefix。`append_eos: true`，`pooling: eos`。具体 task 以 [E5 utils](https://github.com/microsoft/unilm/blob/master/e5/utils.py) 的 `get_task_def_by_task_name_and_type` 为准。

### BGE v1.5 英文 s2p

```
Represent this sentence for searching relevant passages: 
```

中文：

```
为这个句子生成表示以用于检索相关文章：
```

passage 不加 instruction。v1.5 允许不加 query instruction，但 **复现论文必须与论文一致**（多数 BGE 表用了 instruction）。不要索引无 instruction、查询有 instruction。

### GTE / NV-Embed / GTE-Qwen / SFR / LLM2Vec

各卡不同。常见是 last-token / eos + instruction。**禁止套用 E5 的 `query: `。**

## DPR 专项

- 两个 checkpoint：question encoder + ctx encoder。
- 向量取 `pooler_output`（CLS + Linear + Tanh）。`last_hidden_state[:, 0]` 对不上论文。
- 文档：`title [SEP] text`（`title_mode: dpr_sep`）。
- 相似度：未归一化内积。

## Sentence-Transformers 陷阱

`SentenceTransformer.encode` 可能自动加 prompt 或用 `modules.json` 的 pooling。本 skill 默认走 `scripts/encode.py` 的显式契约，避免双重 prefix。若论文明确用 ST 官方模型且 `1_Pooling` 配置已知，把该 pooling 抄进 YAML，仍不要叠加 ST 默认 prompt。
