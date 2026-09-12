# Evaluation protocol

评测口径错了，编码全对也会“对不上论文”。先对齐 **数据集 + split + qrels + 主指标**，再谈模型。

## 用哪个工具

- BEIR 数据集：官方 corpus / queries / qrels，指标用 pytrec_eval（本仓库 `evaluate.py`）。
- MTEB retrieval：主分是 **nDCG@10**；k 默认为 `1,3,5,10,20,100,1000`。
- 原始 MS MARCO passage ranking：开发集 **MRR@10**，不是 BEIR 的 nDCG@10。
- DPR 论文 NQ / TriviaQA：passage retrieval **Hit@k / Accuracy@k**（k=20/100），不是 nDCG。

`eval.primary` 必须等于论文表头。表是 `R@1000` 就报 `Recall@1000`。

## 未标注文档

标准 TREC / BEIR：run 里出现、qrels 没有的文档视为 **不相关（0）**。不要删掉未标注文档再算 nDCG，也不要用 LLM 补标注。

## BEIR 数据约定

- 文档文本：`title_mode: concat_space` → `(title + " " + text).strip()`。title 为空则只用 text。
- 常用 split：多数数据集 `test`；**MS MARCO 在 BEIR 里评 `dev`**。
- `k_values`：至少包含论文出现的 k，建议 `[1, 3, 5, 10, 20, 100, 1000]`。
- 主指标：nDCG@10（除非论文另写）。

## 必须剔除 query 自身的数据集

Query id 与 corpus id 相同且该文档不是“要找的另一条”时，检索结果应去掉自身。

- Quora（重复问题）：`eval.ignore_identical_ids: true`，且 `data.symmetric: true`。
- 其他数据集默认 `false`。MTEB 对部分任务也开了这个开关；以官方任务定义为准。

## MS MARCO 三种口径（不要混）

| 口径 | 集合 | 主指标 |
|------|------|--------|
| 原版 passage ranking | MS MARCO dev | MRR@10 |
| BEIR MSMARCO | BEIR 处理后的 dev | nDCG@10 等 BEIR 指标 |
| TREC DL | 该年 qrels（多级相关） | nDCG@10 |

论文写 “MS MARCO” 时先看引用的是哪张表、哪个官方脚本。

## NQ

- DPR：用 DPR 发布的 Wikipedia 块 + 官方 retrieval qrels，报 Hit@20 / Hit@100。
- BEIR / MTEB NQ：另一套 corpus 与 nDCG 口径。
- 不要用 Natural Questions 的 QA EM 去对 retrieval 表。

## 指标定义（与 pytrec_eval 一致）

- **nDCG@k**：`ndcg_cut_k`，相关等级用 qrels 原值。
- **MAP@k**：`map_cut_k`。
- **Recall@k**：`recall_k`（检索到的相关文档 / 该 query 全部相关文档）。
- **MRR@k**：第一篇相关文档排名 r（r≤k）的 `1/r`，再对 query 平均。
- **Hit@k / Accuracy@k**：top-k 内是否存在任意一篇相关；DPR 用这个。

全部对 query 宏平均（每个 query 等权），不要按相关文档数加权。

## 对表容差

默认：

- nDCG / MAP / MRR：绝对差 ≤ `0.01`
- Recall / Hit：绝对差 ≤ `0.02`

超出即未复现。硬件与框架造成的 0.00x 抖动可以接受；5 个点以上几乎一定是契约错误，不是随机种子。

## 禁止

- 用 cross-encoder 重排后再去对 **dense-only** 表。
- 用 ANN 结果对 **exact search** 表。
- 只在语料子集上评，却去对全库数字。
- 自己实现 nDCG（边界、gain、截断与 pytrec_eval 极易不一致）。
