# Stage 1 环境与最小 Smoke Test（2026-09-27 首次运行；2026-10-08 核对）

## 结论与边界

**CPU 逻辑 Smoke Test 通过；官方 CUDA/NCCL 训练未运行。** 项目使用上游固定 commit `5a704061a7ebf0b81afb465f12f93f2747ba4667` 的源码快照，存于 `third_party/fuxi-linear/`；仅对时间通道的一处未初始化属性做必要修复。测试用官方 `DatasetV2`、`create_data_loader`、`FuXiLinear`、`SampledSoftmaxLoss`、`AdamW`、`eval_metrics_v2_from_tensors` 与 `CandidateIndex`。为了在 Apple CPU 上执行，仅在 `--cpu-reference` 运行时注册三个 FBGEMM 变长张量算子的 PyTorch 参考实现。它有前向、零长度、截断、反向梯度检查，但不是 FBGEMM 性能/数值等价认证。

首次运行机器：macOS 26.6.2, Apple M5（10 核 CPU，8 核 Apple GPU，16 GB 内存），arm64。隔离运行环境 Python 3.12.14；复用本机既有环境中的 PyTorch 2.14.0。`torch.version.cuda=None`、`cuda.is_available=False`、CUDA 设备数 0、CUDA Compute Capability **不适用**；MPS 已构建且可用，但本次刻意用 CPU 参考算子。实际版本与 import 结果列在下表。`pip check` 的真实结果：`No broken requirements found.`

| 依赖 | 官方 `requirements.txt` / 源码 | 本次已安装 / import | 处理 |
|---|---|---|---|
| Python | 未锁 Python 版本 | 3.12.14 | 仅 Smoke Test |
| PyTorch | 2.2.2 | 2.14.0 / PASS | 复用已有环境；`F.rms_norm` 可用；未声称版本等价 |
| CUDA / GPU | CUDA 12.1、NCCL、NVIDIA GPU | 无 CUDA；Apple M5 GPU / MPS 可用 | 官方入口不可在本机原样运行 |
| `gin-config` | 0.5.0，直接 import | 0.5.0 / PASS | 项目 `.venv` 中安装 |
| `fbgemm-gpu` | 0.6.0，`main.py` 直接 import，模型调用 3 种算子 | 未安装 / import 失败 | CPU Smoke 时显式注册 3 个参考算子 |
| `apex` | README 建议；本固定版本源码无直接 import | 未安装 / import 失败 | Smoke 路径不需要 |
| `iopath` | 0.1.10；当前源码无直接 import | 0.1.10 / PASS | 项目 `.venv` 中安装 |
| NumPy | 1.26.4 | 2.2.6 / PASS | 与现有 Python 3.12/Pandas/Scipy 可用，未降级全环境 |
| Pandas | 2.2.1 | 2.2.3 / PASS | 最小数据路径需要 |
| SciPy | 1.13.0 | 1.13.1 / PASS | 当前源码无直接 import |
| `einops` | 源码直接 import，requirements 漏列 | 0.8.1 / PASS | 必须补装 |
| `tqdm` | 源码直接 import | 4.67.1 / PASS | 必须补装 |
| TensorBoard | 2.16.2，eval/trainer import | 2.20.0 / PASS | 用于官方 eval import |
| `mamba_ssm` / `transformers` | baseline 源码 import，requirements 漏列 | 未安装 / import 失败 | 独立导入 FuXi 类，不经过无条件导入全部 baseline 的 `encoder_utils.py` |
| `causal_conv1d` | TTT/SSD 源码 import，requirements 漏列 | 未安装 | 非 FuXi Smoke 路径 |
| `seaborn` | 两个快手预处理脚本直接 import，requirements 漏列 | 未安装 / import 失败 | 本次没有真实原始数据，未运行预处理 |
| `matplotlib` / `absl-py` | 预处理 / 入口源码 import | 3.11.1 / 2.5.0，PASS | 已可用 |

本表来自静态 import 扫描和真实 import 检查。上游大型 `requirements.txt` 含 TensorFlow、TensorRT、Triton、NVIDIA CUDA wheels 等，与此 macOS CPU Smoke 路径无关，未执行整表安装或升级。`pip check` 的执行结果保存在 `artifacts/stage1/pip_check.txt`。当前 `.venv` 使用 `.pth` 指向本机已有 PyTorch 环境，**只适用于这台机器**；其它机器应在独立 venv 中安装兼容 PyTorch 后安装 `requirements-stage1.txt` 的最小包。

### 冲突与必要修复

1. **CUDA/FBGEMM** — Expected: `torch==2.2.2`、CUDA 12.1、`fbgemm-gpu==0.6.0`、NCCL。Installed: PyTorch 2.14.0 CPU 构建，CUDA 不可用，`fbgemm_gpu` 不存在。Error: 官方 `main.py` 在导入 `fbgemm_gpu` 时失败，且设备数为 0；模型调用的 FBGEMM op 不存在。Root cause: 官方运行入口面向 NVIDIA CUDA，当前机器为 Apple Silicon。Minimal fix: 不改训练入口，仅 Stage 1 脚本显式启用 CPU 参考版三个 jagged op，测试官方模型/损失/评估的 CPU 计算链。真正官方运行仍需 NVIDIA CUDA 主机与匹配的依赖栈。
2. **非 chunk 时间通道** — Expected: 时间通道可执行多头注意力。Installed: 上游 commit 原码。Error: `AttributeError: 'LinearTemporalChannel' object has no attribute '_no_multihead'`，见 [`nonchunked_before_fix.log`](../artifacts/stage1/nonchunked_before_fix.log)。Root cause: `forward()` 使用 `_no_multihead`，`__init__()` 未初始化；该分支还依赖未定义的 `_ws_t`，实际配置均使用正常多头分支。Minimal fix: 在该类初始化中增加 `self._no_multihead = False`，不改变数学结构。修复后同一非 chunk Smoke Test 通过。
3. **小候选集评估** — Expected: 过滤已看物品后输出 k 个候选。Installed: 上游 `eval_metrics_v2_from_tensors` 固定 `k=min(2500, catalog_size)`；合成 catalog 大小为 64。Error: 若过滤已见物品，部分用户可用候选少于 64，`CandidateIndex.get_top_k_outputs()` 的 `view(-1,k)` 无法成立。Root cause: 当前评估实现假定过滤后仍有 k 个候选。Minimal fix: 本 Smoke Test 用官方指标代码进行未过滤评估，再用官方 `CandidateIndex` 单独检查过滤已见物品的 Top-5；未改官方评估指标定义。真实大数据实验需在对应规模再核验。

**Paper says:** 时间通道与位置通道可用于高效长序列推理。**Code does:** 当前非 chunk 时间通道在未修复前直接报错；chunk 模式可运行，但官方入口和 FBGEMM 仍要求 CUDA。**Possible impact:** 不能把此 CPU Smoke 的通过推断为官方 GPU 训练或论文效率复现。**Our decision:** 只修复已复现的初始化错误，GPU 性能与真实性能留给后续明确授权的阶段。

## 真实 Smoke Test 结果

输入为自动生成的 **6 个合成用户 / 51 条合成交互**，不是任一论文数据集。最大历史 8，2 层 FuXi-Linear，`D=16`，2 个 retention heads，时间 heads=2，位置核维度 4，8 个负采样，batch size=2；随机种子 2026。实际执行 3 个 `DataLoader → forward → sampled softmax loss → backward → AdamW.step`，随后对 6 用户运行官方 eval 函数和单独的已见物品过滤 Top-5。chunk size=4 与 `chunk_size=None` 均完成。chunk 模式的实际 loss 为 `5.84887505, 6.04632378, 8.19381237`，全局梯度范数为 `100.5644, 100.9376, 98.2031`；这只是小样本通路检查，不表明收敛。非 chunk 前两项完全一致，第三项梯度范数约 `98.2031`，见两个 JSON。每步 29 个参数张量实际更新。首批所有模型参数 `grad != None` 且梯度范数非零，包含 item embedding、主 retention 路径、时间与位置通道及 MFFN。前向、loss、梯度、更新后的参数和 eval 输出均 finite，未触发设定的 loss/梯度粗略异常阈值 `1e4`。没有吞吐、延迟、显存或真实推荐质量测量。

关键首批 Tensor（精确全部见 [`smoke.json`](../artifacts/stage1/smoke.json)）：

| Tensor | 实测 Shape | 含义 |
|---|---|---|
| item IDs / timestamps | `[2,12]` / `[2,12]` | 2 用户、12 个历史/预留位置 |
| sequence lengths | `[2]` | 实际历史长度 |
| item embeddings / hidden states | `[2,12,16]` / `[2,12,16]` | `D=16` |
| jagged block input | `[11,16]` | 首批 2 用户共 11 个有效历史 token |
| semantic Q / K / padded V | `[11,16]` / `[11,16]` / `[2,12,16]` | 2 heads × 8 维；V 被 pad |
| temporal Q / K / log-decay | `[2,12,4,2]` / `[2,12,4,2]` / `[2,12,4]` | 时间正余弦及衰减 |
| semantic / temporal / positional output | 各 `[2,12,16]` | 三路输出 |
| MFFN gated input / output | `[11,48]` / `[11,16]` | 三路拼接后作用于有效 token |
| labels / sampled logits / loss | `[2,11]` / `[11,9]` / `[]` | 1 正样本 + 8 负样本；标量 loss |
| final user / candidate embeddings / filtered Top-5 | `[2,16]` / `[1,64,16]` / `[2,5]` | 评估通路 |

`--debug-shapes` 只打印第一次训练 batch；不传时关闭。机器没有完成原生 FBGEMM 与 CUDA、分布式 DDP、原始数据预处理或完整训练，也没有性能/模型指标可报告。

## 运行命令

在项目根目录执行，示例前提是 Python 3.12 环境已装 PyTorch，并在 `.venv` 中装好 `requirements-stage1.txt`。此机器的 `.venv` 已链接现有 PyTorch 环境。

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/inspect_environment.py
.venv/bin/python scripts/smoke_test.py --cpu-reference --debug-shapes
.venv/bin/python scripts/smoke_test.py --cpu-reference --chunk-size 0 --output artifacts/stage1/smoke_nonchunked.json
.venv/bin/python scripts/data_stats.py artifacts/stage1/synthetic_sequences.csv
```

真实数据可用时，统计命令改为官方产出的 `sasrec_format.csv`，结果才可标为该数据集统计；脚本按唯一用户物品对计算稀疏度，序列长度在截断/划分前计算。
