# FuXi-Linear Stage 0：论文与官方源码 Code Map

分析基准：[论文 arXiv:2602.23671v1](https://arxiv.org/html/2602.23671v1)；[官方仓库](https://github.com/USTC-StarTeam/fuxi-linear) commit [`5a704061a7ebf0b81afb465f12f93f2747ba4667`](https://github.com/USTC-StarTeam/fuxi-linear/tree/5a704061a7ebf0b81afb465f12f93f2747ba4667)。本文中的源码相对路径均以该 commit 的仓库根目录为准。工作区原本只有 `LongTail-ETEGRec`，官方源码仅在临时目录 `/tmp/driftfuxi-stage0-source` 只读检查；没有将它改造成项目源码。

**范围与证据**：检查了 `README.md`、`main.py`、`requirements.txt`、三个 `preprocess_*.py`、全部 24 个 Gin 配置，以及 `generative_recommenders/` 下的 data、trainer、modeling、indexing 文件。没有安装依赖、下载训练数据或执行模型；以下形状与复杂度为代码推导，论文结果不作为本地验证结果。

## 1. 仓库入口与目录地图

| 路径 | 实际作用 |
| --- | --- |
| `README.md` / `requirements.txt` | 数据目录、启动方式与依赖清单；README 的安装示例和锁定的 requirements 不相同。 |
| `main.py` | 解析 Gin，建立或加载 `tmp/loaded/{dataset}-l{L}.pt` 缓存，按 GPU 数量 `mp.spawn` 调用 `train_fn`。 |
| `configs/{kuairand-27k,kuairec,ml-20m}/*.gin` | 每个数据集均有 alpha、beta、hstu、linear、mamba、sasrec、tim4rec、ttt 共 8 个配置；`linear-*.gin` 对应 FuXi-Linear。 |
| `preprocess_public_data.py` | 实际启用 ML-20M 预处理，其余调用被注释。 |
| `preprocess_kuairand27k_data.py` / `preprocess_kuairec_data.py` | 读原始 CSV、迭代过滤、按用户时间排序、重映射视频 ID、写 `sasrec_format.csv`。 |
| `generative_recommenders/data/{preprocessor,dataset,reco_dataset,eval}.py` | 公共预处理、样本切分与缓存、数据集构造、全物品评估。`item_features.py` 是 MovieLens 侧信息结构。 |
| `generative_recommenders/trainer/{data_loader,train}.py` | DataLoader、模型/损失/优化器构造、训练与评估循环。 |
| `generative_recommenders/modeling/sequential/` | `encoder_utils.py` 模型分发；`features.py` 批次特征；`embedding_modules.py` item embedding；`input_features_preprocessors.py` 绝对位置 embedding；`output_postprocessors.py` 输出归一化；`utils.py` 最后有效位置抽取；`autoregressive_losses.py` 采样损失；六类基线与 FuXi 家族源码。 |
| `generative_recommenders/modeling/sequential/fuxi_modules/` | `attn.py` 实现三个通道；`linear_attn.py`、`linear_attn_fn.py` 实现固定块大小的块间状态与块内注意力；`__init__.py` 实现 MFFN；`rab.py` 是其他相对偏置辅助实现，不是 FuXi-Linear 的三通道入口。 |
| `generative_recommenders/modeling/{similarity_module,similarity_utils,ndp_module}.py`、`modeling/similarity/` | DotProduct/MoL 的打分接口；当前 Gin 使用 DotProduct。`initialization.py` 负责 embedding 初始化。 |
| `generative_recommenders/indexing/` | `mips_top_k.py` 全物品矩阵乘加 `topk`；`candidate_index.py` 过滤已见物品。 |

其他源码：`fuxi_alpha.py` 用分离的时间/位置平方注意力；`fuxi_beta.py` 用函数化时间偏置、仍构建平方矩阵；`hstu.py` 用含相对时间和位置偏置的平方注意力；`sasrec.py` 用 `MultiheadAttention`；`mamba4rec.py` 调用外部 Mamba；`tim4rec.py` 和 `ssd.py` 调用时间调制的 SSD；`ttt4rec.py` 包装 `ttt.py` 的 TTT-MLP。`ttt.py` 自带 TTTCache、TTTLinear、TTTMLP、TTTModel，但推荐包装调用没有传 cache。

## 2. FuXi-Linear 精确定位

| File | Class / Function | Purpose |
| --- | --- | --- |
| `modeling/sequential/encoder_utils.py` | `fuxi_linear_encoder`, `get_sequential_encoder` | Gin 配置注入和 `Fuxi-Linear` 分发。 |
| `modeling/sequential/fuxi_linear.py` | `FuXiLinear`：`forward`, `encode`, `generate_user_embeddings` | 端到端序列模型、输入预处理、堆叠 block 与输出。 |
| 同上 | `FuXiLinearJagged.jagged_forward` | 循环执行多个 block；用 FBGEMM jagged/dense 转换。 |
| 同上 | `FuXiLinearBlockJagged.forward` | 投影出 `u,q,k,v`，调用语义、时间、位置三个通道，拼接/门控，再调用 MFFN。 |
| `modeling/sequential/fuxi_modules/attn.py` | `Retention.forward`, `chunkwise_forward` | 语义 Q/K/V 衰减 retention。 |
| 同上 | `LinearTemporalChannel.forward`, `_get_query_key`, `chunkwise_forward` | 时间戳生成周期 Q/K，间隔生成衰减，对语义 value 单独聚合。 |
| 同上 | `LinearPositionalChannel.forward`, `chunkwise_forward` | 可学习位置核 `_emb`，对位置 value 聚合。 |
| `modeling/sequential/fuxi_modules/linear_attn_fn.py` | `chunkwise_parallel_forward`, `ChunkwiseHiddenStateFunction` | 分块内部局部注意力，块间 KV 状态递推及手写 backward。 |
| `modeling/sequential/fuxi_modules/linear_attn.py` | `chunkwise_forward` | 时间通道使用的变间隔衰减版本。 |
| `modeling/sequential/fuxi_modules/__init__.py` | `MultistageFeedforwardNeuralNetwork.forward` | 三通道合并后两阶段 FFN、残差和 RMSNorm。 |

## 3. 从单条原始 interaction 到推荐结果

下面以 KuaiRand-27K 为例；ML-20M 由 `preprocess_public_data.py` 调用 `MovielensDataProcessor.preprocess_rating`，后续公共流程相同。

```text
raw user_id/video_id/time_ms/play_time_ms/is_hate
  └─ preprocess_kuairand27k_data.py（脚本顶层）
     合并四份日志、迭代频次过滤、视频 ID 重映射、按 time_ms 排序
     → tmp/processed/kuairand-27k/sasrec_format.csv
       [user_id, sequence_item_ids, sequence_ratings, sequence_timestamps]
  └─ data/reco_dataset.py:get_reco_dataset
     → data/dataset.py:DatasetV2.load_item
       train 忽略最后 1 个事件，eval 保留；各自用末事件为 target，
       之前事件为 chronological history，截断/补 0；KuaiRand ID +1，0 是 padding
  └─ main.py:_main/mp_train_fn → trainer/train.py:train_fn
     → trainer/data_loader.py:create_data_loader → PyTorch DataLoader
     → modeling/sequential/features.py:movielens_seq_features_from_row
       past_ids、past_lengths、timestamps、ratings、target_ids/target_timestamps
  └─ trainer/train.py:train_fn
     → embedding_modules.py:LocalEmbeddingModule.get_item_embeddings
     → input_features_preprocessors.py:LearnablePositionalEmbeddingInputFeaturesPreprocessor.forward
     → fuxi_linear.py:FuXiLinear.forward/generate_user_embeddings
     → FuXiLinearJagged.jagged_forward → FuXiLinearBlockJagged.forward × num_blocks
     → attn.py:Retention / LinearTemporalChannel / LinearPositionalChannel
     → fuxi_modules/__init__.py:MultistageFeedforwardNeuralNetwork
     → output_postprocessors.py:L2NormEmbeddingPostprocessor.forward
  ├─ TRAIN: autoregressive_losses.py:SampledSoftmaxLoss.forward/jagged_forward
  │         → similarity_module.py:GeneralizedInteractionModule.interaction
  │         → similarity/dot_product.py:DotProductSimilarity.forward → sampled logits → loss
  └─ EVAL: fuxi_linear.py:FuXiLinear.encode → sequential/utils.py:get_current_embeddings
            → data/eval.py:eval_metrics_v2_from_tensors
            → indexing/mips_top_k.py:MIPSBruteForceTopK.forward
            → indexing/candidate_index.py:CandidateIndex.get_top_k_outputs
            → NDCG/HR/MRR
```

训练的细节：`features.py` 在 `past_lengths` 位置写入 target timestamp；`train_fn` 又在同一位置写入 target ID，但传给 jagged encoder 的有效长度仍是旧 `past_lengths`。这样该 ID 不作为当前前缀输入，`supervision_ids[:,1:]` 却可用它监督最后一个有效位置。训练目标是序列内逐位置 next-item，评估目标是该用户被保留序列的末 item。当前脚本没有独立的 validation/test 用户拆分：同一用户序列的倒数第二个事件用于 train target，最后一个事件用于 eval target。

## 4. Tensor Shape Map

约定：`B`=批量用户数；`L`=样本设定的历史最大长度；`N=L+gr_output_length+1` 是 `features.py` 附加输出槽后的密集长度；`N′=ceil(N/C)C` 是分块时补齐后的长度；`C`=chunk size；`D`=item/模型 embedding 维；`H`=语义 head 数；`d_k`=`dqk`；`d_v`=`dv`；`V=H d_v`；`P`=位置核维；`H_t`=时间周期 head 数，实际拆成 `2H_t` 正弦/余弦 head；`J=sum_b past_lengths[b]`=jagged 有效 token 数；`I`=候选物品数；`R`=采样负例数；`K`=输出推荐个数；`M=3V` 是启用三通道时的拼接维。官方 KuaiRand linear Gin 给 `L=1024,D=128,H=4,d_k=32,d_v=32,P=32,H_t=8,C=128,R=128`；ML-20M 给 `L=200,D=256,H=8,d_k=32,d_v=32,P=32,H_t=8,C=None`。

| Tensor | Shape；每个维度 | Produced at | Consumed at |
| --- | --- | --- | --- |
| `historical_ids` / `historical_timestamps` | `[B,L]`；用户 × 历史槽位 | `DatasetV2.load_item`、DataLoader | `movielens_seq_features_from_row` |
| `target_ids`, `target_timestamps` | `[B,1]`；用户 × 一个待预测事件 | 同上 | `features.py`、`train_fn` / `eval.py` |
| `past_ids`, `past_payloads['timestamps']` | `[B,N]`；用户 × 含预留输出槽位 | `features.py`，训练时 `train_fn` scatter ID | `LocalEmbeddingModule`、`FuXiLinear.generate_user_embeddings` |
| `past_lengths` | `[B]`；每个用户有效历史长度；`x_offsets` 为 `[B+1]` 累积长度 | `DatasetV2.load_item` → `features.py`；FBGEMM cumsum | jagged 转换、当前向量提取、损失 |
| `item embeddings` | `[B,N,D]`；用户 × 槽位 × item 维 | `LocalEmbeddingModule.get_item_embeddings` | 输入特征预处理 |
| `absolute position indices` / embeddings | `[B,N]` / `[B,N,D]`；用户 × 位置（及维） | `LearnablePositionalEmbeddingInputFeaturesPreprocessor.forward` | 加到 item embeddings |
| `user_embeddings` / `combined hidden states` | `[B,N,D]`，分块临时 `[B,N′,D]`；用户 × 序列位置 × 模型维 | 预处理 / 每个 MFFN / 输出归一化 | `FuXiLinearJagged`、loss 或 `get_current_embeddings` |
| `normed_x`, `u`, `q`, `k`, `v`（jagged） | `[J,D]`、`[J,M]`、`[J,H d_k]`、`[J,H d_k]`、`[J,V]`；有效 token × 对应特征维 | `FuXiLinearBlockJagged.forward` 的 `_uvqk` 投影 | 三通道与门控 |
| semantic `Q/K/V`（padded） | `[B,N′,H,d_k]`、`[B,N′,H,d_k]`、`[B,N′,H,d_v]`；用户 × token × head × head 内维 | `Retention._input_preprocess` | `Retention.chunkwise_forward` 或平方分支 |
| `semantic state`（块间） | `[N′/C,B,H,d_k,d_v]`；块 × 用户 × head × key 维 × value 维；单块当前状态 `[B,H,d_k,d_v]` | `linear_attn_fn.ChunkwiseHiddenStateFunction.forward` | 块间输出项 `q·state`；未作为推理 cache 返回 |
| `time intervals` / `theta_t` | `[B,N′-1]` / `[B,N′,H_t]`；用户 × 相邻间隔 / 周期相位 | `LinearTemporalChannel.chunkwise_forward`, `_get_query_key` | 时间衰减与周期 Q/K |
| temporal `Q/K/V` | `[B,N′,2H_t,2]` / `[B,N′,2H_t,2]` / `[B,N′,2H_t,V/(2H_t)]`；用户 × token × 正弦/余弦 head × 相位或 value 维 | `LinearTemporalChannel._get_query_key`, `forward` | `linear_attn.chunkwise_forward` |
| `temporal state` / channel output | `[B,2H_t,2,V/(2H_t)]`（单块递推状态）；`[B,N′,V]` 输出 | `linear_attn.chunkwise_forward`；`LinearTemporalChannel.forward` | 时间输出并入 block；状态没有跨请求保存 |
| positional kernel `E_p` / position `Q/K` | `[N_max,P]` 可学习参数；分块时实际 Q/K 广播自 `[1,N_max,1,P]`；`N_max` 是配置最大长度按 C 补齐 | `LinearPositionalChannel.__init__`, `chunkwise_forward` | 位置通道 |
| `positional state` / channel output | `[B,1,P,V]`（单块状态）；`[B,N′,V]` 输出 | `linear_attn_fn.chunkwise_parallel_forward` / `LinearPositionalChannel.forward` | 位置输出并入 block；状态没有跨请求保存 |
| `combined channels` / gated LMCA | `[B,N′,M]` / `[J,M]`；用户 × token × 三通道维 / 有效 token × 三通道维 | `FuXiLinearBlockJagged.forward` 的 concat、`dense_to_jagged`、`u*` | MFFN |
| `final user representation` | `[B,D]`；每用户最后一个有效位置 | `FuXiLinear.encode` → `get_current_embeddings` | eval top-K |
| `candidate IDs` / `candidate embeddings` | `[1,I]` / `[1,I,D]`；一套全物品候选 | `data/eval.py:get_eval_state` | `MIPSBruteForceTopK` |
| training positive / negative logits | `[J,1]` / `[J,R]`；有效 token × 正/负候选 | `SampledSoftmaxLoss.jagged_forward` → DotProduct | sampled softmax loss |
| evaluation all-item logits / top-K IDs | `[B,I]` / `[B,K]`；用户 × 候选 / 排名 | `MIPSBruteForceTopK.forward` | `CandidateIndex.get_top_k_outputs`、指标 |
| labels / `loss` | `supervision_ids[:,1:]` 为 `[B,N-1]`，jagged 后 `[J]`；`loss` 为标量 `[]` | `train_fn`、`SampledSoftmaxLoss` | `loss.backward` |

`N_max` 与运行时 `N′` 在代码里并非始终同一符号；位置核及因果 mask 按构造时上限建立。当前配置下维度碰巧匹配（如 KuaiRand 的 `N=1035,N′=1152`），扩展序列长度时必须核对索引和 broadcast。非分块时间/位置分支会创建 `[B,heads,N,N]` 或 `[N,N]` 矩阵。

## 5. FuXi-Linear Block：三个通道与 MFFN

### Semantic / Retention Path

`FuXiLinearBlockJagged.forward` 对 `x∈[J,D]` 做 LayerNorm，再经 `_uvqk` 和 SiLU 获得门控 `u` 以及语义 `q,k,v`。`Retention` 每个 head 学习 `_gamma`，`_get_gamma` 以 `exp(-cumsum(softplus(_gamma)))` 给正衰减参数；实际一步因子又用 `exp(-gamma)`。非分块路径显式构造 `QKᵀ` 与因果衰减矩阵，复杂度平方。分块路径把 token 划成长度 C 的块：块内计算 `C×C` 局部注意力，块间通过 KV 状态累积；`ChunkwiseHiddenStateFunction` 用全零 `[B,H,d_k,d_v]` 初始化，递推形如 `S_c = decay_c · S_{c-1} + Σ_i (decay_{i→end} k_iᵀv_i)`，每个位置再把先前块的状态乘以对应 q，合并块内项。该状态是一次 forward 的临时计算，**没有被保存为下一次请求的 recurrent cache**；`Retention.forward` 返回 `(raw_output, None)`，传入 `cache`、`return_cache_states` 也未改变此行为。

### Temporal Retention Channel

`features.py` 把原始 timestamp 放入 `past_payloads['timestamps']`，并把目标时刻放在历史长度位置；block 将其送入 `LinearTemporalChannel`。`_get_query_key` 用 `timestamp % period` 后乘 `2π/period`，产生每周期的 `cos/sin` key，以及利用下一时刻生成的正弦、余弦 query，合计 `2H_t` 个 head。相邻时间间隔在 `chunkwise_forward` 中以 `clamp(t_{i+1}-t_i,min=0)` 得到，乘周期尺度与 `-log(sigmoid(_gamma))` 成为 log 衰减；固定周期来自 `_intervals = base^(start_index + k·base_stride)`。Kuai 脚本时间单位为毫秒，ML-20M 预处理保留 Unix 秒；源码未统一换算。value 由 `normed_x` 经 `proj_v` 投影，输出 `[B,N′,V]`，配置 `aug_current=True` 时使用每头 `alpha·history + beta·current`。时间 Q/K 只由时间戳生成、语义 Q/K 只由 item 特征生成；两个通道输出在 block 末合并，避免把时间因子直接乘入语义 QK 权重。时间并非完全独立于语义，因为 value 仍来自 item 特征。源码还提供 `_get_query_key_inference`，但主 `forward` 没有调用它。

### Linear Positional Channel

单靠线性衰减，邻近位置只体现为粗粒度衰减，难表达灵活的相对位置效应。论文令 `f(i-j)≈k(i)ᵀk(j)`，把位置加权和变为累积 `S_p←S_p+k(j)ᵀv_j`，输出 `k(i)S_p`。源码在 `LinearPositionalChannel.__init__` 以正弦/余弦表初始化可训练 `_emb∈[N_max,P]`，`forward` 投影 `v`，分块函数把 `_emb` 分别用作 q、k，调用共享块状态算子；`aug_current=True` 时再加 `alpha·history+beta·v`，输出 `[B,N′,V]`。固定 C、P、V 时随 L 的主要算量为线性；未分块则显式 `E_p E_pᵀ`，是 `O(L²P+L²V)`。位置通道与输入预处理器的绝对位置 embedding 是两个不同参数族，两者都在当前模型中使用。

### Multi-stage FFN

三通道各经 block 的 `_norm_input`（实际为 **LayerNorm**）后，按源码顺序 `semantic → temporal → positional` 拼成 `[B,N′,M]`，转回 jagged 后乘 `u`。`MultistageFeedforwardNeuralNetwork.forward(X,X0)` 第一阶段 `lin0(dropout(X))+X0` 把三通道压回 D，并与原始 block 输入残差相加；第二阶段对该和做 RMSNorm、dropout，再以 `SiLU(lin1(z))*lin3(z)` 做门控 feature interaction，经 `lin2` 回到 D 并再次残差相加。最后返回 float32。`past_length` 参数传入但 MFFN 没有使用。

## 6. 对序列长度 L 的复杂度

普通 self-attention 需要 `Q∈[L,d_k]` 与 `Kᵀ∈[d_k,L]` 相乘，形成 `[L,L]` 的 `QKᵀ`；每个 query 看全部历史 key，故注意力时间 `O(L²d_k + L²d_v)`、attention map 内存 `O(L²)`，外加投影 `O(LD²)`。`sasrec.py` 的 `MultiheadAttention` 与 HSTU/FuXi-α/β 的显式矩阵属于此类。

对 FuXi-Linear 的**分块通道核心**，若 C、head 数与维度固定，块内 `L/C` 个 `C×C` 局部矩阵：时间 `O(L C (d_k+d_v))`、局部图存储 `O(B H L C)`；块间状态的计算与存储分别随块数 `L/C` 线性增长，典型状态项 `O((L/C) B H d_k d_v)`。再加逐 token 投影 `O(LD²)`，整体对 L 为 `O(L)`；随 D、C 增大，常数并不小。位置状态 `P×V`，时间状态 `2H_t×2×V/(2H_t)`，都与历史 L 无关。单 token 的**理论**递推更新对 L 为 `O(1)`，固定维度下仅需要上述几个状态。

必须区分官方**完整源码路径**：`FuXiLinear.__init__` 总会分配 `_attn_mask∈[N_max,N_max]`，所以静态模型 buffer 内存 `O(L²)`；分块时该 mask 转 float 后未用于三通道主要计算，但仍被建立。`chunk_size=None` 会走三个平方路径，时间/内存都是 `O(L²)`。完整评估还含全物品打分 `O(B I D)` 和已见物品过滤的候选比较，复杂度不能算进序列编码的 `O(L)` 结论。推理 cache 缺失也意味着现有 `encode` 每次重算完整序列，无法据此宣称实际 decode 对 L 常数。

## 7. Training 与 Inference 的真实调用

```text
main.py:_main → mp_train_fn → train_fn
  batch = DataLoader(DatasetV2)
  → movielens_seq_features_from_row
  → model.module.get_item_embeddings(past_ids)
  → DDP(model).forward → FuXiLinear.forward → [B,N,D]
  → SampledSoftmaxLoss.forward → 正例 [J,1]、负例 [J,R] logits → 标量 loss
  → loss.backward()
  → 手工 warmup（若 num_warmup_steps>0；否则恒定 LR）
  → AdamW.step()
```

`train_fn` 先设随机种子、NCCL DDP 和 AdamW；默认 sampled softmax + local negative sampler，每位置正例为下一个 item，重复采到正例的负样本 logits 设为极小值。没有独立 scheduler 对象。训练中每 `eval_interval` 个 batch 用当前 batch 做一次指标检查；每 epoch 遍历 eval DataLoader，`full_eval_every_n` 决定部分评估；`save_ckpt_every_n` 名称像“步数”，实际代码按 epoch 判断且要求 `epoch>0`，训练结束另存一次 `model_state_dict` 和 optimizer state。没有源码里的 checkpoint 加载/续训入口。

**评估推理**：`eval_metrics_v2_from_tensors` 用完整历史调用 `model.encode`，`get_current_embeddings` 取 `past_lengths-1` 位置作为 `[B,D]` 用户向量。`get_eval_state` 建立 `[1,I,D]` 候选 embedding；`MIPSBruteForceTopK.forward` 计算 `[B,D]@[D,I]` 得 `[B,I]` logits，再取 `min(2500,I)` 候选，经 `CandidateIndex.get_top_k_outputs` 排除已见 ID，计算 NDCG/HR/MRR。这是当前真实 top-K 路径，无单独服务/在线推理脚本。

**Prefill / decode / state cache**：论文的加速依据是三通道的 KV/位置/时间递推状态；分块 prefill 可以块内并行加块间递推，理论 decode 每个新 token 更新固定形状状态。源码的 `generate_user_embeddings` / `encode` 虽有 `cache`、`target_timestamps`、`return_cache_states` 形参，`FuXiLinearJagged` 也收集 block 的 cache tuple，但三个通道的实际返回状态均为 `None`；`linear_attn_fn` 的状态只在本次 forward 的局部张量中，未跨调用保留；评估从不传 cache。因此**不能从这份公开训练/评估源码复现论文的增量 decode 加速**。`target_timestamps` 在 block 传到时间通道，但常规 `forward` 未使用它；未分块时间通道还访问未定义的 `_no_multihead` 属性，ML-20M linear Gin 的 `chunk_size=None` 路径存在运行阻断风险（本阶段仅静态发现，未执行验证）。

## 8. 官方基线支持矩阵

下表复杂度仅指模型序列混合机制随 L 的主项，固定模型维度和块长；不包含全物品 top-K。每个配置名字以仓库现有文件为准，`{ds}` 分别为 `kuairand-27k`、`kuairec`、`ml-20m`。

| Model | Config | Main source file | Attention/state mechanism | Complexity wrt L / 实际支持 |
| --- | --- | --- | --- | --- |
| SASRec | `configs/{ds}/sasrec-*.gin` | `modeling/sequential/sasrec.py` | `torch.nn.MultiheadAttention` causal self-attention | `O(L²)`；分发已接入。 |
| HSTU | `configs/{ds}/hstu-*.gin` | `modeling/sequential/hstu.py` | QK 全矩阵 + 时间/位置相对偏置 | `O(L²)`；分发已接入。 |
| Mamba4Rec | `configs/{ds}/mamba-*.gin` | `modeling/sequential/mamba4rec.py` | 外部 `mamba_ssm.Mamba` 选择性状态空间层 | 理论 `O(L)`；分发已接入，依赖需另行核对。 |
| TiM4Rec | `configs/{ds}/tim4rec-*.gin` | `modeling/sequential/tim4rec.py`, `ssd.py` | 时间间隔调制 SSD chunk scan | 固定块长时序列混合约 `O(L)`；分发已接入，另有按完整长度参数化的时间 MLP。 |
| TTT4Rec | `configs/{ds}/ttt-*.gin` | `modeling/sequential/ttt4rec.py`, `ttt.py` | TTT-MLP 小批在线权重更新 | 固定 mini-batch、维度时约 `O(L)`；分发已接入，实际调用未用 TTTCache。 |
| FuXi-Linear | `configs/{ds}/linear-*.gin` | `modeling/sequential/fuxi_linear.py`, `fuxi_modules/attn.py` | retention + 独立时间/位置核，chunkwise KV 状态 | Kuai 两配置 `chunk_size=128` 时核心 `O(L)`；ML 配置 `chunk_size=None` 为 `O(L²)` 且有未定义属性风险。 |
| FuXi-α / FuXi-β | `alpha-*.gin` / `beta-*.gin` | `fuxi_alpha.py` / `fuxi_beta.py` | 分离的相对时间/位置 full matrix；β 采用连续时间函数 | `O(L²)`；分发已接入。 |

论文另列 RecBLR、RetNet 等对比，但本仓库没有对应模型类或 Gin 配置，不能视为官方代码已支持。

## 9. Paper–Code 差异与复现风险台账

### 9.1 完整线性内存与 decode

- **Paper says:** 固定 C 时训练内存随 L 线性；递推形式支持对 L 为常数的 decode。
- **Code does:** 分块通道确实避免全长 attention map，但模型构造保留 `[N_max,N_max]` mask；三个通道不返回可复用状态，eval 每次全量 `encode`。
- **Possible impact:** 峰值模型内存仍有平方项，公开路径无法直接验证论文的 decode latency/throughput 结论。
- **Our decision:** Stage 0 仅注明理论与代码路径边界；未来复现阶段须分别测编码核心、完整模型和真正 cache decode，不把论文数字当本地实测。

### 9.2 ML-20M linear 配置

- **Paper says:** FuXi-Linear 对长序列保持线性复杂度。
- **Code does:** `configs/ml-20m/linear-8b-b256x4.gin` 设 `chunk_size=None`，三个通道走平方并行分支；时间通道该分支访问未定义的 `self._no_multihead`。
- **Possible impact:** 这一公开配置与线性效率主张不对应，而且可能在前向时抛 `AttributeError`。
- **Our decision:** 标为静态阻断风险；本阶段不运行/修复，后续按实际执行验证后再决定配置或源码处理。

### 9.3 KuaiRand 过滤规则

- **Paper says:** 保留没有负反馈的事件，并保留交互次数大于 100 的 item。
- **Code does:** 脚本计算了 `pos_index=(play_time_ms>0)&(is_hate!=1)` 和 `df=df[pos_index]`，但此前 `df1=df`，之后全部过滤及输出仍用 `df1`；视频阈值是 `>=350`，用户阈值是 `>=20`。
- **Possible impact:** 论文所述样本定义与当前输出不一致，训练集组成与统计口径会改变。
- **Our decision:** 保留原脚本作证据；后续复现需明确选论文口径还是公开代码口径并分别记录结果，本阶段不改。

### 9.4 时间周期与衰减初始化

- **Paper says:** Kuai 数据周期底数 B=8；各时间衰减 `r_k` 按周期初始化，并可有多尺度参数。
- **Code does:** Kuai linear Gin 为 `base=2,start_index=10,base_stride=3`，周期依次为 `2^(10+3k)`；相邻周期的**比例确实是 8**，但起始周期为 `2^10`，不能只看 `base=2` 就断言与论文的 B=8 冲突。`LinearTemporalChannel` 用 `sigmoid(_gamma)`，默认非 learnable 为 0.5，仅第 0 层在配置要求时设 learnable gamma；其余层为固定 gamma。`_get_query_key_inference` 实现未接入主路径。
- **Possible impact:** 周期起点、每层学习自由度及衰减初值的细节不能仅从论文的 B=8 描述还原，可能影响跨数据集时间建模与推理等价性。
- **Our decision:** 以配置/代码为实际复现基线，记录论文差异；不在 Stage 0 调参。

### 9.5 归一化与 MFFN 公式

- **Paper says:** FuXi-Linear block 的 Norm 采用 RMSNorm；MFFN 公式只写门控投影项。
- **Code does:** block `_norm_input` 与三个通道输出用 `F.layer_norm`；MFFN 内用 `F.rms_norm`，且第二阶段输出还加上第一阶段残差。
- **Possible impact:** 按论文公式手写可能无法复现当前实现的数值行为。
- **Our decision:** Code Map 以真实源码为准；论文公式作设计解释，不把两者视为逐算子等价。

### 9.6 数据、评估与依赖口径

- **Paper says:** 报告全物品 HR/NDCG/MRR，并描述使用 Accelerate 多卡。
- **Code does:** `main.py` 使用 `mp.spawn` + NCCL DDP；评估先取最多 2500 个原始 top 候选，再过滤已见物品；未命中者被设为 rank 2501，故源码 MRR 对这些样本给非零倒数。`data_loader.py` 虽暴露 `num_workers`、`prefetch_factor`，实际构造 DataLoader 时固定 `num_workers=0`。`requirements.txt` 未列出部分实际 import（如 `mamba_ssm`、`transformers`、`einops`、`tqdm`、`seaborn`），README 安装命令也不是完整锁定清单。
- **Possible impact:** 训练并行、排名和依赖的可复现性与论文/文档简述存在差距；HR/NDCG 截断后未命中是 0，但 MRR 定义有近似。
- **Our decision:** 本阶段不装包、不执行；后续先按精确源码检查依赖与评价协议，所有报告标明评估口径。

## 10. Stage 0 静态核验与边界

- 已读取官方 commit 与 24 个配置路径，并定位所有要求的类/函数与数据流；只读 `ast.parse` 静态检查覆盖 45 个 Python 文件，语法解析失败数为 0（Python 对部分既有字符串转义给出 `SyntaxWarning`）。未产生训练指标、延迟、吞吐或显存实测。
- 可做的只读复核命令（不触发训练）：

```bash
git -C /tmp/driftfuxi-stage0-source rev-parse HEAD
rg --files /tmp/driftfuxi-stage0-source/generative_recommenders
rg -n 'class (FuXiLinear|FuXiLinearBlockJagged|Retention|LinearTemporalChannel|LinearPositionalChannel|MultistageFeedforwardNeuralNetwork)' /tmp/driftfuxi-stage0-source/generative_recommenders
rg -n 'chunk_size|main_module' /tmp/driftfuxi-stage0-source/configs/kuairand-27k/linear-4b-l1024-b64x2.gin /tmp/driftfuxi-stage0-source/configs/ml-20m/linear-8b-b256x4.gin
```

- `/tmp` 源码快照可能被系统清理；永久证据是顶部的 commit permalink。官方源码没有被加入当前工作区；这个 Stage 的唯一交付文件是本文。上述未定义属性、依赖缺失等是**静态风险**，尚无运行错误日志。Stage 0 到此停止。

## Interview Critical Files

1. `preprocess_kuairand27k_data.py`：讲原始交互怎样变成按时序的用户列表，并能指出过滤变量没有贯穿输出；重点看 `pos_index`、`df1`、`groupby` 与 `sasrec_format.csv` 写入。
2. `generative_recommenders/data/dataset.py`：讲 train/eval 末事件切分、反转、截断/补齐和 ID 偏移；重点看 `DatasetV2.load_item`。
3. `generative_recommenders/trainer/train.py`：讲训练入口、采样损失、反传、评估与 checkpoint；重点看 `train_fn` 主循环。
4. `generative_recommenders/modeling/sequential/fuxi_linear.py`：讲模型入口、block 三通道拼接与 cache 现状；重点看 `FuXiLinearBlockJagged.forward`、`FuXiLinear.generate_user_embeddings`、`encode`。
5. `generative_recommenders/modeling/sequential/fuxi_modules/attn.py`：讲语义 retention、时间周期 Q/K、位置可学习核；重点看 `Retention`、`LinearTemporalChannel._get_query_key`、`LinearPositionalChannel.chunkwise_forward`。
6. `generative_recommenders/modeling/sequential/fuxi_modules/linear_attn_fn.py`：讲块间 KV 状态与块内 `C×C` 注意力；重点看 `ChunkwiseHiddenStateFunction`、`chunkwise_parallel_forward`。
7. `generative_recommenders/modeling/sequential/autoregressive_losses.py`：讲逐位置 sampled softmax、负采样形状与正例掩码；重点看 `SampledSoftmaxLoss.jagged_forward`。
8. `generative_recommenders/data/eval.py`：讲用户向量如何送入全物品 top-K 并计算指标；重点看 `get_eval_state`、`eval_metrics_v2_from_tensors`。

## Critical Code Path

```text
Data
  DatasetV2.load_item → movielens_seq_features_from_row
→ Model Entry
  train_fn → get_sequential_encoder → FuXiLinear.forward / encode
→ FuXi Block
  FuXiLinearJagged.jagged_forward → FuXiLinearBlockJagged.forward
→ Linear State
  Retention.chunkwise_forward → linear_attn_fn.chunkwise_parallel_forward
→ Temporal Channel
  LinearTemporalChannel._get_query_key → linear_attn.chunkwise_forward
→ Positional Channel
  LinearPositionalChannel._emb → chunkwise_forward
→ Output
  MultistageFeedforwardNeuralNetwork.forward → L2NormEmbeddingPostprocessor
→ Loss
  SampledSoftmaxLoss.jagged_forward → DotProductSimilarity
→ Evaluation
  FuXiLinear.encode → get_current_embeddings → MIPSBruteForceTopK
  → CandidateIndex.get_top_k_outputs → eval_metrics_v2_from_tensors
```
