# 官方来源与版本

- 论文：[FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation](https://arxiv.org/abs/2602.23671)，Stage 0 分析基于 [HTML v1](https://arxiv.org/html/2602.23671v1)。
- 官方仓库：[USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear)。
- Stage 0 / Stage 1 固定源码基准：[`5a704061a7ebf0b81afb465f12f93f2747ba4667`](https://github.com/USTC-StarTeam/fuxi-linear/tree/5a704061a7ebf0b81afb465f12f93f2747ba4667)。
- Stage 1 用该 commit 的 `git archive` 建立 `third_party/fuxi-linear/` 快照，保留上游 LICENSE 与 NOTICE。仅 `generative_recommenders/modeling/sequential/fuxi_modules/attn.py` 有一处必要改动：初始化 `self._no_multihead = False`，修复非 chunk 时间通道已复现的 AttributeError。CPU 参考 jagged 算子在项目 `scripts/cpu_jagged.py`，不属于官方源码。

Stage 0 代码地图见 [codebase_analysis.md](codebase_analysis.md)，Stage 1 的运行边界和日志见 [environment_report.md](environment_report.md)。
