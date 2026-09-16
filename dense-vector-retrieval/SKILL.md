---
name: dense-vector-retrieval
description: >-
  Paper-faithful dense vector retrieval with bi-encoders: lock the encoding
  contract, exact search, official IR metrics (nDCG, MRR, Recall, Hit@k). Use
  when reproducing DPR/BEIR/MTEB/MS MARCO/NQ dense retrieval numbers, encoding
  a corpus with a dual encoder, or when retrieval scores must match a paper
  instead of a RAG demo. Do not use for BM25-only, cross-encoder-only rerank,
  ColBERT/late-interaction, or sparse/hybrid fusion unless the paper's dense
  first-stage is the task.
---

# Dense Vector Retrieval

用双塔（bi-encoder）把 query / passage 编成固定维向量，再用内积或余弦做检索。本 skill 的目标不是“能搜”，而是 **按论文的编码契约跑完，指标和论文表同一口径、同一量级**。

默认假设：Agent 会执行仓库里的脚本，而不是重写 pooling / 评测。

## Hard rules

违反任一条，数字都会和论文拉开，常见是 nDCG@10 差 3～15 个点。

1. **先锁契约，再动 GPU。** 12 个字段没齐，禁止 encode。
2. **禁止猜 pooling / prefix / 相似度。** 论文没写就读模型卡和官方 eval 脚本，再查 [references/model-conventions.md](references/model-conventions.md)。仍缺则停下来问，不要用“业界常用 cosine”填空。
3. **复现论文必须精确检索**（`index.type: flat`）。HNSW / IVF / nprobe 会掉 recall，不能和论文表对比，除非论文自己用了 ANN 并公开了参数。
4. **禁止手写评测。** 用 `scripts/evaluate.py`（pytrec_eval）。禁止用 LLM 当 qrels，禁止自己实现 nDCG。
5. **论文报 Hit@20 就报 Hit@20。** 不要把 nDCG@10 拿去对 DPR-NQ 的 Accuracy@20。
6. **ColBERT / MaxSim / 多向量不是本 skill。** 单向量 dense 才走这里。
7. **prefix 字符串必须逐字复制**（含末尾空格、换行）。`query:` 和 `query: ` 不是同一个契约。

## Encoding contract（12 字段）

从论文实验设置 + 官方脚本 + 模型卡抽取，写入 YAML。缺一不可：

| # | 字段 | 典型来源 |
|---|------|----------|
| 1 | `encode.model_name_or_path` 或 query/doc 双 checkpoint | 论文 Table / 附录 |
| 2 | `encode.architecture`: `auto` / `dpr` / `bi_encoder` | DPR 必须 `dpr` |
| 3 | `encode.pooling`: `cls` / `cls_pooler` / `mean` / `eos` | 模型卡；DPR 用 `cls_pooler` |
| 4 | `encode.normalize` | 训练目标；E5/BGE 一般为 true |
| 5 | `encode.score`: `ip` / `cos_sim` | 与 normalize 必须自洽 |
| 6 | `encode.query_max_length` / `doc_max_length` | 常为 32/512、64/256、512/512 |
| 7 | `encode.query_prefix`（逐字） | E5 / BGE / instruct 模型 |
| 8 | `encode.passage_prefix`（逐字） | E5 为 `passage: `；多数为空 |
| 9 | `data.title_mode`: `concat_space` / `dpr_sep` / `text_only` | BEIR 默认 `concat_space` |
| 10 | `data.dataset` + `split` + 官方 qrels | 禁止用 train 对 test 表 |
| 11 | `eval.primary`（如 `nDCG@10` / `MRR@10` / `Hit@20`） | 与论文表头一致 |
| 12 | `index.type` | 复现必须 `flat` |

拷贝 [assets/config.template.yaml](assets/config.template.yaml)，填完后再跑。

`normalize: true` 时 `ip` 与 `cos_sim` **排序等价**，必须两边都 L2 归一。一边归一一边不归一，排序是错的。

`normalize: false` 时必须用 `ip`（DPR、Contriever 原文）。不要擅自改成 cosine。

## Workflow

复制清单并勾选。

```
- [ ] 1. Lock contract into YAML
- [ ] 2. validate_config.py  (exit 0)
- [ ] 3. encode.py
- [ ] 4. retrieve.py         (flat / exact)
- [ ] 5. evaluate.py
- [ ] 6. Compare to paper within tolerance
```

在 skill 根目录执行（把 CONFIG 和 RUN 换成实际路径）：

```bash
python scripts/run_pipeline.py --config path/to/config.yaml --work-dir path/to/run --stage all
```

分步调试用 `--stage validate|encode|retrieve|evaluate`。

### 1. Lock contract

读论文时只采信这些位置，忽略 intro 里的口头描述：

- Experimental Setup / Implementation Details
- 表注、附录超参
- 官方 GitHub eval 脚本（BEIR / MTEB / Tevatron / Pyserini / FlagEmbedding）
- Hugging Face 模型卡的 Usage 代码（pooling 以代码为准，不以散文为准）

对称任务（Quora 类 duplicate）：query 与 passage 用 **同一套 prefix**。E5 官方对 Quora 把 passage 也当成 query 加 `query: `。见 [references/evaluation-protocol.md](references/evaluation-protocol.md)。

### 2. Validate

脚本会：

- 检查 12 字段齐全
- 对照家族默认值；不一致且未设 `encode.paper_override: true` 则失败
- 检查 `score` 与 `normalize` 自洽
- 复现模式拒绝非 `flat` 索引

家族表见 [references/model-conventions.md](references/model-conventions.md)。未知模型且无 override → 失败。这是故意的。

### 3. Encode

只跑 `scripts/encode.py`。实现必须满足 [references/encoding-invariants.md](references/encoding-invariants.md)：

- `model.eval()` + `torch.no_grad()`
- mean pooling **必须**乘 `attention_mask`，padding 不得进平均
- EOS pooling 取 **最后一个非 pad token**；decoder-only 用 left padding
- DPR 用 `pooler_output`，不是 `last_hidden_state[:,0]`
- 截断发生在 **加 prefix 之后** 的 tokenization
- 同一 checkpoint、同一 `max_length`、同一 dtype 编 query 和 corpus（双塔除外）

输出：`work_dir/embeddings/` 下的 `corpus.npy`、`queries.npy`、id 列表、`encode_meta.json`。后续阶段发现 meta 与 config 不一致必须停。

### 4. Retrieve

复现：精确 Inner Product / cosine，全库打分。允许分块矩阵乘，**不允许** IVF 近似。

保存 TREC run：`work_dir/runs/{dataset}.run`。

### 5. Evaluate

`scripts/evaluate.py` 用 pytrec_eval，未标注文档当 0 分（标准 IR）。主指标必须等于 `eval.primary`。

BEIR / MTEB retrieval 主指标默认 **nDCG@10**。  
MS MARCO passage ranking 常报 **MRR@10**。  
DPR on NQ/TriviaQA 常报 **Hit@20 / Hit@100**（Accuracy@k）。

### 6. Compare

配置里填写 `paper.claimed_metrics`。绝对差超过 `paper.tolerance`（默认 nDCG/MRR/MAP 为 `0.01`，Recall/Hit 为 `0.02`）则进程退出码 2，并打印诊断，**禁止**把结果写成“已复现”。

## 对不上时只按这个顺序查

不要调学习率、不要换模型“试试”。先查实现：

1. prefix 漏加 / 加错边 / 末尾空格丢失  
2. pooling 错（尤其 DPR 的 `cls` vs `cls_pooler`；E5-Mistral 的 `eos`）  
3. BEIR title 未拼接（应为 `title + " " + text`）  
4. 单边 L2 归一  
5. 用了 HNSW/IVF  
6. mean pooling 没 mask  
7. `max_length` 与论文不同  
8. split / qrels 错（MS MARCO train vs BEIR 的 MSMARCO；NQ 的 retrieval vs QA 口径）  
9. 对称任务 prefix 策略错（Quora）  
10. checkpoint revision 不同  
11. 语料规模对不上（子集 vs 全库）  
12. 主指标 @k 与论文表头不同  

查完仍超差：把 `encode_meta.json`、metrics、论文表号一起报告，不要改契约去凑数。

## Apply 模式（RAG / 业务检索）

`task.mode: apply` 时仍用同一套编码契约。可以换 ANN，但必须在结果里写清 `index.type` 以及相对 flat 的 recall 损失；**不要**把 ANN 数字和论文表放在一起。

chunk 不是论文默认。只有语料不是现成 passage、或模型 `max_length` 会截断关键内容时才切块；切块后数字不可直接对论文。

Apply 下的 skill 自进化由 `skill_evolution` 在**工作副本**上编译，禁止把检索段落正文写入本文件。基线三个槽位必须保持恒等；运行时只替换标记内部，主干与 Hard rules 只读。

<!-- EVOLVE:QUERY:START -->
恒等：q' = q
<!-- EVOLVE:QUERY:END -->

<!-- EVOLVE:RANK:START -->
恒等：score(d) = sim(q, d)
<!-- EVOLVE:RANK:END -->

<!-- EVOLVE:FILTER:START -->
恒等：keep(d) = true
<!-- EVOLVE:FILTER:END -->

## 脚本

在 skill 根目录安装：`pip install -r scripts/requirements.txt`

| 脚本 | 作用 |
|------|------|
| `scripts/run_pipeline.py` | 编排 validate → encode → retrieve → evaluate |
| `scripts/validate_config.py` | 契约检查 |
| `scripts/encode.py` | 按契约编码 |
| `scripts/retrieve.py` | 精确检索 |
| `scripts/evaluate.py` | pytrec_eval + 对表 |
| `python -m skill_evolution --self-test` | Apply 自进化：工作副本上用 Rocchio / 时间核改槽位 |

Agent **执行**这些脚本，不要复制进对话后改逻辑。改契约只改 YAML。自进化只允许改 `EVOLVE:QUERY` / `EVOLVE:RANK` / `EVOLVE:FILTER` 标记内的公式，禁止把检索正文写入 skill。

## Additional resources

- 模型家族默认值：[references/model-conventions.md](references/model-conventions.md)
- 评测口径：[references/evaluation-protocol.md](references/evaluation-protocol.md)
- 编码不变量：[references/encoding-invariants.md](references/encoding-invariants.md)
- 配置模板：[assets/config.template.yaml](assets/config.template.yaml)
