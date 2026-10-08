# DriftFuXi

**面向兴趣漂移的高效长序列推荐系统**  
*Interest-Drift-Aware Efficient Long-Sequence Recommendation*

目标：以 [FuXi-Linear 论文](https://arxiv.org/abs/2602.23671)及其[官方实现](https://github.com/USTC-StarTeam/fuxi-linear)为基准，分阶段完成可用于推荐算法、搜广推算法及序列推荐算法实习展示的复现与创新项目。

## 当前状态

Stage 0 的论文与源码分析已完成，见 [Code Map](docs/codebase_analysis.md)。Stage 1 的 **CPU 合成数据最小 Smoke Test 已通过**，见 [环境报告](docs/environment_report.md)与[数据流程](docs/data_pipeline.md)。Stage 2 已完成 KuaiRec 原始数据校验、官方预处理与 baseline 运行准备，见 [复现记录](docs/reproduction_report.md)；当前无 NVIDIA CUDA 主机，完整训练和真实推荐/效率指标尚未产生。

## 项目结构

```text
DriftFuXi/
├── docs/                       # Stage 0–2 分析、环境、数据与复现记录
├── third_party/fuxi-linear/   # 官方固定 commit 源码快照，必要修复和运行记录已记录
├── scripts/                    # 数据下载、统计、Smoke 与 Stage 2 运行工具
├── artifacts/stage1/           # 合成 fixture、真实执行日志和检查 JSON
├── requirements-stage1.txt     # 已有 PyTorch 环境之上的最小增量依赖
└── AGENTS.md                   # 分阶段执行边界
```

## Stage 1 本机复查

运行环境在 `.venv`，复用本机 现有环境 的 PyTorch。可在项目根目录运行：

```bash
.venv/bin/python -m pip check
.venv/bin/python scripts/smoke_test.py --cpu-reference --debug-shapes
.venv/bin/python scripts/smoke_test.py --cpu-reference --chunk-size 0 --output artifacts/stage1/smoke_nonchunked.json
```

其它机器需先建立隔离 Python 3.12 + PyTorch 环境，再安装 `requirements-stage1.txt`；`--cpu-reference` 仅测试 CPU 计算链。源码来源与 Stage 1 修复见 [upstream.md](docs/upstream.md)；Stage 2 的修复与改动见 [复现记录](docs/reproduction_report.md)。

每次只执行用户明确指定的当前 Stage；完成后汇报核心文件、命令、验证结果与现存问题，并停止。指标和性能数据只来自实际运行。
