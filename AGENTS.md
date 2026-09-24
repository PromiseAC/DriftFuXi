# DriftFuXi 工作约束

- 当前已完成 Stage 0：论文与官方源码分析。不要把未来阶段视为已完成。
- 严格按用户给出的 Stage 顺序工作；每次只执行当前明确授权的 Stage，完成即停止。
- 未获下一 Stage 指令前，不修改模型源码、不安装依赖、不下载数据、不训练、不增加 drift gate 或其他创新模块。
- 任何论文与源码不一致之处按 `Paper says / Code does / Possible impact / Our decision` 记录。
- NDCG、HR、MRR、latency、throughput、GPU memory 与提升百分比只能报告真实运行结果；论文数字必须明确标为论文报告。
- 每一 Stage 交付时说明：修改/分析内容、核心文件、运行命令、验证结果、现存问题。
- 官方源码分析固定基准 commit 见 `docs/upstream.md`；引入或更新上游源码必须显式记录版本，避免静默漂移。
