# Stage 2 — FuXi-Linear baseline 复现记录（准备中，尚未完成正式训练）

本阶段遵守官方源码的 KuaiRec `linear-4b-l1024-b64x2.gin`。**目前没有 NVIDIA CUDA 主机，也没有完整运行，因此所有 Our Run、差值和效率结果均待测。** `artifacts/stage1` 的合成数据 Smoke Test 不能填入本表。源码基准为官方 `5a704061a7ebf0b81afb465f12f93f2747ba4667`；Stage 2 的必要运行修复见下文。运行成功后，以 `outputs/baseline/kuairec/linear-4b-l1024-b64x2/seed-42/` 中配置、元数据、训练日志、`epoch_metrics.jsonl` 和 checkpoint 作为证据。

## 实验协议与来源

- 优先 KuaiRec：作者 [公开数据仓库](https://github.com/chongminggao/KuaiRec) 的 `KuaiRec.zip` 内 `data/big_matrix.csv`；作者 [Zenodo 归档](https://zenodo.org/records/18164998) 大小 431,964,858 字节，MD5 `261550d472c48eff4990fb13c0e5bcf7`。原始数据仅放在忽略的 `tmp/`，不进入 Git。已下载并通过大小、MD5 与解压 CRC 校验；官方预处理完成。结果为 7,176 users、9,958 observed items、12,529,113 interactions，与论文规模一致。
- 官方预处理：`preprocess_kuairec_data.py` 的双阈值迭代过滤、按 timestamp 排序、视频 ID 重映射；官方 `get_reco_dataset()` 以同一用户序列构建 train（排除末条，倒数第二条目标）和 eval（末条目标）。`DatasetV2` 历史最多 1024，`chronological=True`，zero padding，快手 ID +1。**源码没有独立 validation split 和 test split**；`eval_dataset` 每个 epoch 都使用，同一组指标不能同时标作独立 validation 与 test。本报告仅称“官方 eval”。
- 官方模型配置：4 blocks、embedding 128、4 semantic heads、`dqk=dv=32`、chunk 128、时间 heads 8、位置维度 32、dropout 0.5、`gr_output_length` 默认 10。
- 官方训练：101 个 `range(num_epochs)` epoch、每 GPU batch 64，AdamW（β=0.9/0.98）、学习率 1e-3、weight decay 0、warmup 0；128 个本地均匀负采样、temperature 0.05、sampled softmax。一个 GPU 的全局 batch 是 64；两 GPU 才是 128。单 GPU 的结果会明确标为与论文 batch 不同，不改配置追数字。
- 官方评估：`eval_metrics_v2_from_tensors`、全 item `MIPSBruteForceTopK`、过滤用户历史物品，记录 NDCG@10/50、HR@10/50、MRR。验证与训练保持同一候选协议。官方 `full_eval_every_n=1`，每个 epoch 全量 eval。
- 种子：`train_fn.random_seed=42` 显式写入每次保存的 config；官方函数固定 Python `random`、NumPy、PyTorch、CUDA，本项目补 `torch.cuda.manual_seed_all`。当前源码实际强制 DataLoader `num_workers=0`（尽管 Gin 设置为 8），`DistributedSampler(seed=0)` 在每 epoch `set_epoch(epoch)`。启动包装设置 `PYTHONHASHSEED` 和 `CUBLAS_WORKSPACE_CONFIG`。这些措施固定随机来源，但不承诺跨不同 CUDA/PyTorch 版本逐位相同。

## 论文结果对照

以下 Paper 列**只引用论文 Table 3 的 KuaiRec FuXi-Linear**，不是本地测量：[论文原文](https://arxiv.org/html/2602.23671v1)。绝对差定义 `Our Run - Paper`，相对差定义 `(Our Run - Paper) / Paper`。单 seed 42 完整运行后才填写；如资源允许再做 3 seeds 并报告 mean ± sample std。

| Metric | Paper | Our Run | Absolute Difference | Relative Difference |
|---|---:|---:|---:|---:|
| NDCG@10 | 0.1368 | 待测 | 待测 | 待测 |
| NDCG@50 | 0.1851 | 待测 | 待测 | 待测 |
| HR@10 | 0.2242 | 待测 | 待测 | 待测 |
| HR@50 | 0.4486 | 待测 | 待测 | 待测 |
| MRR | 0.1235 | 待测 | 待测 | 待测 |

训练 wall-clock、峰值 GPU allocated memory、训练吞吐（处理的用户序列数/秒）、评估吞吐（候选索引构建、编码、Top-K、指标的端到端用户序列数/秒）均待真实 CUDA 运行。吞吐定义来自 `trainer/train.py` 的同步计时；多卡以全局处理序列数除以 rank 0 所见 wall-clock。`torch.cuda.max_memory_allocated` 是 PyTorch allocated peak，不等于 `nvidia-smi` 中整卡占用。准备脚本不写虚构的数值。

## 与论文/源码的差异以及排查次序

**Paper says:** KuaiRec 报告 7,176 users、9,958 items、12,529,113 interactions，训练默认 100 epochs、batch 128。**Code does:** 预处理输入是作者的 12,530,806 条原始 big matrix，过滤后实际为 12,529,113 条、7,176 users、9,958 items；`get_common_preprocessors()` 为 KuaiRec 配置 `9958+1=9959` 个候选 item，可能包含 1 个没有交互的 phantom item；config 为 101 epochs、每卡 batch 64。**Possible impact:** 预处理版本、候选集大小、epoch 和 GPU 数都会影响与论文的差。**Our decision:** 首跑保持固定上游源码与 Gin 配置，仅做导入阻断修复和日志/种子记录；与论文不一致之处不静默调整。真实运行后按 dataset version → preprocessing → sequence length → model size → batch size → GPU count → optimizer → LR → warmup → candidate set → evaluator → seed 的顺序排查。

**Paper says:** 全物品 top-K 评估。**Code does:** 在 `CandidateIndex` 中过滤历史物品；没有独立 val/test。**Possible impact:** 与其他仓库的候选采样或划分协议不可直接比较；每 epoch 查看同一 eval 集不能当作独立未触碰 test。**Our decision:** 只使用官方 evaluator 并报告其真实名称，不新造 split 或换指标。

## Stage 2 源码与运行准备

1. `encoder_utils.py` 将无关 Mamba、TiM4Rec、TTT 的导入延迟到对应 baseline 构造函数。原实现仅选 FuXi 时也要求 `mamba_ssm`、`transformers` 等未列入 requirements 的依赖；修改不触及 FuXi 的张量计算。
2. `trainer/train.py` 可选读取 `DRIFTFUXI_RUN_DIR`，逐 epoch 保存官方 evaluator 原始五项指标、同步 wall-clock、吞吐与 rank 0 峰值 GPU allocated memory；增加 `torch.cuda.manual_seed_all`。split、模型、负采样、loss、候选集、优化器均未改。原有日志与 checkpoint 继续由官方训练函数生成。
3. `scripts/run_stage2_baseline.py` 固定 config/seed/commit、记录硬件与数据 SHA-256，把训练输出隔离到 `outputs/.../seed-42/`，禁止覆盖已有结果。`--prepare-only` 在无 GPU 时只创建元数据，不产生指标。完整训练要求 `torch.cuda.is_available()` 且 `fbgemm_gpu` 可导入。每次 GPU 运行结束归档官方最终 checkpoint。
4. `scripts/download_kuairec.py` 只下载作者 Zenodo 原始 zip 并校验 MD5。已把归档中的 `data/big_matrix.csv` 解压到 `tmp/kuairec/`，从项目根目录执行官方 `preprocess_kuairec_data.py`，见 `tmp/kuairec_preprocess.log`。`scripts/data_stats.py` 为 KuaiRec 长序列提高 CSV 字段限制，不改变数据。
5. 官方 requirements 锁定 torch 2.2.2，但该版本没有 `torch.nn.functional.rms_norm`，而上游 FuXi MFFN 调用了它。`fuxi_modules/__init__.py` 的 `rms_norm_compat` 在 API 不存在时按 FP32 均方根公式计算并回转原 dtype；API 存在时直调原生函数。已在本地 PyTorch 上比较原生与回退公式的前向、反向结果，尚未在锁定的 Linux/CUDA 环境实测。

本机已通过：KuaiRec 官方下载校验、官方预处理、全量统计、FuXi 训练模块 import、Stage 2 脚本 AST、`--prepare-only`、CPU 合成 smoke 回归、`pip check`。**未通过/未执行：官方 CUDA/FBGEMM 运行、真实训练、全量 eval、GPU 显存与性能。** 无可用 NVIDIA 主机之前，不能将 Stage 2 标记完成。

## 已完成的数据核验

| 指标 | 官方预处理后的实际统计 |
|---|---:|
| Users | 7,176 |
| Observed items | 9,958 |
| Interactions | 12,529,113 |
| 序列平均 / 中位长度 | 1,745.97 / 1,846.5 |
| 序列 p90 / p95 / 最大长度 | 2,912.5 / 3,244.25 / 16,015 |
| 唯一 user-item 对 / 稀疏度 | 10,299,440 / 85.5868% |

长度是截断和 train/eval 构造前的真实每用户交互条数；稀疏度定义为 `1 - unique(user,item)/(users × observed_items)`。官方模型再将历史截断到 1024。

## 算力预算（仅价格与时间情景，不是实测运行时间）

当前本机没有 NVIDIA GPU。AutoDL [官网价格页](https://www.autodl.com/home)显示 RTX 4090 24 GB 约 ¥1.88/卡·小时，RTX 3090 24 GB 约 ¥1.32/卡·小时；实例区、库存、硬盘等费用需在下单前再次核对；[按量计费说明](https://www.autodl.com/docs/price/)规定实例开机到关机按秒计费，环境配置与上传数据也占用付费时长。按公开数据 7,176 users 与官方每卡 batch 64 粗算，1 卡约 113 steps/epoch，2 卡约 57 steps/epoch，101 epochs 对应约 11,413 / 5,757 个同步训练 step；预处理、每 epoch 全量 eval 另耗时。尚无 GPU 实测 step 时间，**不能据此可靠推断总小时数**。

| 租用方案 | 每小时计算费 | 5 小时情景 | 20 小时情景 |
|---|---:|---:|---:|
| 1× RTX 3090 | ¥1.32 | ¥6.60 | ¥26.40 |
| 1× RTX 4090 | ¥1.88 | ¥9.40 | ¥37.60 |
| 2× RTX 3090 | ¥2.64 | ¥13.20 | ¥52.80 |
| 2× RTX 4090 | ¥3.76 | ¥18.80 | ¥75.20 |

建议先给 **1× RTX 4090、2 小时上限** 做环境/显存/单 epoch 计时预检，计算费上限约 ¥3.76（不含可能的存储费用）；预检后再根据实测速率决定完整训练预算。首个 **一 GPU、完整官方 config 的 run** 可先打通 full pipeline，但全局 batch 64 与论文 128 不同。若追求论文 batch 一致，租两卡并保持每卡 64。实际总价应先用短暂 GPU 预检测 step 时间、显存和 eval 时间后更新；预检只用于预算，不填论文复现表。

## Training Flow：一个 batch 在官方代码中如何训练

`DatasetV2.__getitem__/load_item` 返回用户历史、长度、目标和 timestamps → `create_data_loader` 组成 batch → `movielens_seq_features_from_row` 组装 `SequentialFeatures` 并放入目标时间 → `train_fn` 将目标 ID scatter 到有效历史之后 → `model.get_item_embeddings` / `LocalEmbeddingModule.get_item_embeddings` 取 item embedding → `FuXiLinear.forward` / `generate_user_embeddings` 经输入预处理与 `FuXiLinearJagged` 的多个 `FuXiLinearBlockJagged.forward`（`Retention`、`LinearTemporalChannel`、`LinearPositionalChannel`、`MultistageFeedforwardNeuralNetwork`）得每位置 sequence representation → `SampledSoftmaxLoss.forward/jagged_forward` 使用 `LocalNegativesSampler.forward` 和 `model.interaction` 计算正负 logits 与 loss → `loss.backward()` → `AdamW.step()`。每 epoch，`get_eval_state` 建全候选索引，`eval_metrics_v2_from_tensors` 调 `model.encode`、`CandidateIndex.get_top_k_outputs` 给出官方五项指标。

## 本地复核与待执行命令

```bash
# 在项目根目录；已完成，不需要重复下载或预处理
.venv/bin/python scripts/download_kuairec.py
MPLBACKEND=Agg .venv/bin/python third_party/fuxi-linear/preprocess_kuairec_data.py
.venv/bin/python scripts/data_stats.py tmp/processed/kuairec/sasrec_format.csv
.venv/bin/python scripts/smoke_test.py --cpu-reference --output /tmp/driftfuxi-stage2-regression.json
.venv/bin/python scripts/run_stage2_baseline.py --seed 42 --prepare-only

# 仅在 NVIDIA Linux/CUDA 环境、匹配的官方依赖已安装、数据已迁移后执行
python3 scripts/run_stage2_baseline.py --seed 42
```

当前阻断是缺少可用的 NVIDIA CUDA 主机与已经验证匹配的 Linux GPU 依赖。租卡成本须先由用户确认；正式训练、结果表与复现差值仍待真实运行。
