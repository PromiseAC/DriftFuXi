# 官方来源与版本

- 论文：[FuXi-Linear: Unleashing the Power of Linear Attention in Long-term Time-aware Sequential Recommendation](https://arxiv.org/abs/2602.23671)，Stage 0 阅读的 HTML 版本为 [v1](https://arxiv.org/html/2602.23671v1)。
- 官方仓库：[USTC-StarTeam/fuxi-linear](https://github.com/USTC-StarTeam/fuxi-linear)。
- Stage 0 源码固定 commit：[`5a704061a7ebf0b81afb465f12f93f2747ba4667`](https://github.com/USTC-StarTeam/fuxi-linear/tree/5a704061a7ebf0b81afb465f12f93f2747ba4667)。
- 该源码只在 `/tmp/driftfuxi-stage0-source` 做过只读检查；临时目录可能被系统清理，项目目录不依赖它存在。

只读核对上游版本的命令（会从网络获取官方仓库到临时目录；不安装依赖、不运行模型）：

```bash
git clone https://github.com/USTC-StarTeam/fuxi-linear.git /tmp/driftfuxi-stage0-source
git -C /tmp/driftfuxi-stage0-source checkout 5a704061a7ebf0b81afb465f12f93f2747ba4667
git -C /tmp/driftfuxi-stage0-source rev-parse HEAD
```

若临时目录已经存在，直接运行最后一条命令即可核对。Stage 0 的源码分析、差异与静态风险记录见 [codebase_analysis.md](codebase_analysis.md)。
