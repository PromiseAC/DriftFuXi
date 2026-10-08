# DriftFuXi 工作约束

- Stage 0 已完成；Stage 1 已完成（CPU 合成数据 Smoke Test，官方 CUDA/真实数据实验仍未完成）。不要把后续阶段视为已完成。
- 严格按用户给出的 Stage 顺序工作；每次只执行当前明确授权的 Stage，完成即停止。
- 未获下一 Stage 指令前，不实施论文复现、完整训练、模型创新、drift gate 等下一阶段工作。
- 任何论文与源码不一致之处按 `Paper says / Code does / Possible impact / Our decision` 记录。
- NDCG、HR、MRR、latency、throughput、GPU memory 与提升百分比只能报告真实运行结果；论文数字必须明确标为论文报告。
- 每一 Stage 交付时说明：修改/分析内容、核心文件、运行命令、验证结果、现存问题。
- 官方源码基准 commit 见 `docs/upstream.md`；引入或更新上游源码必须显式记录版本，避免静默漂移。
