# DriftFuXi

**面向兴趣漂移的高效长序列推荐系统**  
*Interest-Drift-Aware Efficient Long-Sequence Recommendation*

目标：以 [FuXi-Linear 论文](https://arxiv.org/abs/2602.23671)及其[官方实现](https://github.com/USTC-StarTeam/fuxi-linear)为基准，分阶段完成可用于推荐算法、搜广推算法及序列推荐算法实习展示的复现与创新项目。

## 当前状态

**Stage 0 已完成：论文与官方源码分析。** 当前项目只有分析与项目约束，没有模型源码、依赖环境、数据、训练结果或创新模块。完整 Code Map 见 [docs/codebase_analysis.md](docs/codebase_analysis.md)。

## 项目结构

```text
DriftFuXi/
├── README.md                    # 项目定位与当前状态
├── AGENTS.md                    # 分阶段执行边界
├── .gitignore                   # 本地数据、环境与实验产物忽略规则
└── docs/
    ├── codebase_analysis.md    # Stage 0 交付：论文与官方源码 Code Map
    └── upstream.md             # 官方论文、源码与分析基准版本
```

## 阶段边界

每次只执行用户明确指定的当前 Stage；完成后汇报修改/分析内容、核心文件、运行命令、验证结果、存在问题，并停止。后续阶段的实现与实验只有在用户发出下一阶段任务后才开始。指标和性能数据必须来自实际运行，不使用论文报告值冒充本地复现值。

## 查看 Stage 0

无需安装依赖。直接阅读 [Code Map](docs/codebase_analysis.md) 和 [上游版本说明](docs/upstream.md)。如需对照源码，可按 `docs/upstream.md` 的只读命令检出对应官方 commit。
