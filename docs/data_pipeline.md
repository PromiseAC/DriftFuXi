# Stage 1：三个官方数据集的数据流程

本文件基于 `third_party/fuxi-linear` 的固定源码快照。**本机项目 `tmp/`、`data/` 与 本机共享 `AI-data` 目录 未发现三个数据集的原始或官方处理文件；没有下载、预处理或生成它们的统计指标。** 项目 `artifacts/stage1/synthetic_sequences.csv` 仅供 Smoke Test，统计见 `synthetic_stats.json`，不得标为 KuaiRand、KuaiRec 或 MovieLens 数据。

| 数据集 | 原始输入（官方相对路径） | 预处理 | 产物（官方相对路径） | train / validation / test |
|---|---|---|---|---|
| KuaiRand-27K | `tmp/kuairand-27k/log_standard_4_08_to_4_21_27k_part1.csv`、`part2.csv`；`log_standard_4_22_to_5_08_27k_part1.csv`、`part2.csv` | `preprocess_kuairand27k_data.py` 连接四表；迭代保留出现数≥350 的 item 和序列长度≥20 的用户；按 `time_ms` 排序；视频重新映射至 0 开始；rating 固定 4.0 | `tmp/processed/kuairand-27k/video_id_mapping.csv`、`sasrec_format.csv` | `get_reco_dataset()` 从同一 CSV 构建 train (`ignore_last_n=1`) 和 eval (`ignore_last_n=0`)；无独立 validation/test 文件 |
| KuaiRec | `tmp/kuairec/big_matrix.csv` | `preprocess_kuairec_data.py` 迭代保留出现数≥5 的 item / 用户；`timestamp` 秒×1000 转毫秒；时间排序；视频重映射至 0 开始；rating 固定 4.0 | `tmp/processed/kuairec/video_id_mapping.csv`、`sasrec_format.csv` | 同上；无独立 validation/test 文件 |
| MovieLens-20M | `tmp/ml-20m/ratings.csv`、`movies.csv`；`tmp/movielens20m.zip` 不存在时脚本会从 GroupLens 下载它，即使解压后的 CSV 已存在 | `preprocess_public_data.py → get_common_preprocessors()["ml-20m"].preprocess_rating()`；保留原始正整数 movie ID（未压缩），以 Unix 秒时间戳排序并组成用户序列；生成电影信息和 ratings | `tmp/processed/ml-20m/{movies.csv,ratings.csv}`、`tmp/ml-20m/sasrec_format.csv`；另生成 `sasrec_format_by_user_train.csv`、`sasrec_format_by_user_test.csv` | 当前 `get_reco_dataset()` **使用完整 `sasrec_format.csv` 构建 train/eval**，不读取按用户划分的两个文件；无单独 validation 集 |

官方中间 CSV 每位用户一行，字段为 `user_id`, `sequence_item_ids`, `sequence_ratings`, `sequence_timestamps`，快手 CSV 额外有 `index`。`DatasetV2.load_item()` 把字符串序列解析为列表，先取末尾作目标，然后取前面最多 `max_sequence_length` 个交互作历史，并按 `chronological=True` 恢复先旧后新的顺序，不足补零。快手视频 ID 在加载时 `shift_id_by=1`，因为 0 是 padding；MovieLens ID 不加 1。训练例子的目标是倒数第二个交互（排除最后一个）；评估例子的目标是最后一个。模型的自回归训练又将训练目标放回 `past_ids[past_lengths]`，对所有有效位置做 next-item sampled softmax。这个 train/eval 是**同一批用户按时间留一**，不是三份互斥用户或交互文件。官方 `create_data_loader()` 构建 PyTorch `DataLoader`，尽管配置有 `num_workers=8`，实际代码固定 `num_workers=0`。

对于“validation/test”，官方命名里 `eval_dataset` 承担末条事件测试；训练中按 epoch 调用评估以观察模型，但它不是从另一个验证文件构建的独立集合。MovieLens 预处理额外写出的按用户约 90/10 划分文件被标为“not tested yet”，当前训练入口不使用。Stage 1 不改数据划分，也不报告泛化结论。

## 数据统计状态与定义

| 数据集 | users | items | interactions | avg / median / p90 / p95 / max 序列长度 | sparsity |
|---|---:|---:|---:|---|---:|
| KuaiRand-27K | 未测 | 未测 | 未测 | 未测 | 未测 |
| KuaiRec | 未测 | 未测 | 未测 | 未测 | 未测 |
| MovieLens-20M | 未测 | 未测 | 未测 | 未测 | 未测 |

真实 `sasrec_format.csv` 可用时执行：

```bash
.venv/bin/python scripts/data_stats.py third_party/fuxi-linear/tmp/processed/kuairand-27k/sasrec_format.csv
.venv/bin/python scripts/data_stats.py third_party/fuxi-linear/tmp/processed/kuairec/sasrec_format.csv
.venv/bin/python scripts/data_stats.py third_party/fuxi-linear/tmp/ml-20m/sasrec_format.csv
```

路径示例以官方源码目录为当前数据工作根；实际预处理脚本使用进程的工作目录解析 `tmp/`，因此若从项目根目录运行，应传 `tmp/processed/...` 或 `tmp/ml-20m/...`。统计脚本对处理后、截断与 train/eval 留一前的每用户完整序列计算用户数、观察到的 item 数、交互总数、均值/中位数/p90/p95/最大长度；`sparsity = 1 - unique(user,item) / (users × observed_items)`。使用唯一用户物品对是因为同一用户可重复交互同一物品。原始输入的过滤前统计与处理后统计不同，不能混用。长度分布对后续兴趣漂移分桶很关键，需等真实数据到位后再填。

## 代码差异与风险

**Paper says:** KuaiRand 仅使用正反馈，且长序列过滤条件与论文描述一致。**Code does:** `preprocess_kuairand27k_data.py` 构造 `df1=df` 后，在 `df` 上计算并应用 `play_time_ms>0` 且 `is_hate!=1` 的筛选，但后续阈值过滤及输出全部继续使用 `df1`；实际没有应用反馈筛选，代码阈值为 item≥350 / user≥20。**Possible impact:** 处理后的训练分布可能含负反馈，与论文数据协议不一致，真实长序列分布和指标不可直接比较。**Our decision:** Stage 1 只记录，未运行或更改预处理；真实数据验证前先确定协议并修正或明确复现实验版本。

**Paper says:** 三个公开数据集提供论文评测。**Code does:** 训练路径只读每个数据集一份序列 CSV，train/eval 为同用户留一；MovieLens 写出的按用户 split 不参与 `get_reco_dataset()`。**Possible impact:** 将按用户文件当作官方测试集会改变实验协议。**Our decision:** Stage 1 如实记录源码实际切分，不另建划分。
