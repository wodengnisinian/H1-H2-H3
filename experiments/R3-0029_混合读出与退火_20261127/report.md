# R29 逐轮报告：混合读出与退火

2026 年 11 月 27 日。针对 R28 确诊的「稀疏度↔泛化」权衡（位于训练目标层面），本轮把 sparsemax 与 entmax(1.5) 在分布层面混合、训练 α 退火、叠加可学习温度与历史 w 初始化。预注册见 [configs/protocol.json](configs/protocol.json)。

## 主结果（600 图冻结队列，三题型等权，配对 Bootstrap 10,000 次）

| 变体 | H1 | H1 判定 | H2 | H2 判定 | 未见句式 H1 / H2 |
| --- | ---: | --- | ---: | --- | --- |
| learned（冻结原公式） | 70.16% | 未过 | 72.20% | 过 | 43.44% / 50.50% |
| relmask_cand（R22） | 78.61% | 未过 | 75.37% | 过 | 34.89% / 45.89% |
| entmax15（R26） | 79.41% | 未过 | 75.85% | 过 | 35.67% / 44.33% |
| sparsemax（R27） | 81.19% | 过 | 78.77% | 过 | 27.67% / 42.78% |
| **mix（主变体）** | **80.04%** | **过** | **77.03%** | **过** | **30.28% / 44.78%** |
| mix_anneal | 77.01% | 未过 | 76.69% | 过 | 35.94% / 45.61% |
| sparsemax_anneal | 78.03% | 未过 | 76.82% | 过 | 34.00% / 45.89% |
| sparsemax_tau | 80.69% | 过 | 78.95% | 过 | 27.72% / 42.33% |
| sparsemax_winit | 70.44% | 未过 | 75.25% | 过 | **44.44%** / 49.89% |

## 判读

1. **mix 是首个以主变体身份同时通过 H1（80.04%）与 H2（77.03%）开发门槛的无词表读出**：H1 相对词面顺序 +30.04pp（下界 +27.38pp），H2 相对均匀 +7.86pp（下界 +7.17pp）。λ 学到的混合比例三种子为 0.51 / 0.33 / 0.49，即 sparsemax 与 entmax 大致对半混合。
2. **泛化哨兵仍未达标**：mix 未见句式 H1 30.28%，未达协议要求的 35%，按预注册决策规则本轮不冻结、不进入 H3 准备。
3. **关键诊断 sparsemax_winit**：用历史 learned w 初始化后，H1 掉到 70.44%，但未见句式升到 44.44%（与 learned 43.44% 相当）——泛化主要来自初始化/权重本身，稀疏读出只是放大器。这提示下一步应把「winit 的泛化」与「sparsemax/mix 的 H1」结合。
4. 逐种子：mix 为 78.31 / 74.69 / 87.11；sparsemax_tau 为 78.19 / 78.03 / 85.83（跨种子较稳）。α 退火（anneal）系列反而低于直接 α=2.0，退火没有带来预期收益。

## 训练与实现

- mix 末轮 NLL 与 λ 见 [results/prediction/train-*.json](results/prediction/)；sparsemax_tau 学到 τ=1.140 / 1.609 / 1.227。
- α 退火路径用加宽二分括号的 EntmaxRobust（仅 r29 内部，r26 库未改动），小数 α 的浮点边界问题已处理。
- 连续性校验通过：relmask_cand 78.61%、entmax15 79.41%、sparsemax 81.19%、learned 70.16%。
- 云端 predictions SHA-256 `abae0c54…`；冻结依赖 model.py `f85636ff…`、队列 `cf7d3ba2…` 核对通过。

## 结论与下一步

主变体通过 H1/H2 但未见句式未达 35% 哨兵，按协议继续循环。R30 方向：

1. **mix_winit**：混合读出的两个分量都用历史 learned w 初始化（结合 R29 两条证据）；
2. mix_winit 与可学习温度叠加；
3. λ 初值偏向 entmax 侧的受控变体。

累计轮次 R29。开发 H1/H2 双过 + 未见句式 ≥35% 才冻结进入新 H3 验收。

## 归档清单

- [预注册协议](configs/protocol.json) · [训练与推理脚本](code/r29_mix.py) · [统计脚本](code/r18_stats.py) · [汇总脚本](code/r29_summary.py)
- [逐题预测](results/prediction/predictions.jsonl.gz) · [统计结果](results/stats/metrics.json) · [图级指标 CSV](results/stats/graph_metrics.csv) · [云端日志](logs/)
