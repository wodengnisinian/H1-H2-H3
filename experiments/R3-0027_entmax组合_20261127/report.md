# R27 逐轮报告：entmax 组合（角色监督、多任务、α 扫参、对比归一化）

2026 年 11 月 27 日。在 R26 最佳诊断（entmax α=1.5，H1 79.41%）基础上组合升级。硬约束延续：不得预先写死任何时序/标记字词表。预注册见 [configs/protocol.json](configs/protocol.json)，执行于 tdklgrqctvo9g2techo.funhpc.com（torch 2.10.0）。

## 主结果（600 图冻结队列，三题型等权，图内平均三种子与问法，配对 Bootstrap 10,000 次）

| 变体 | H1 | H1 判定 | H2 | H2 判定 | 未见句式 H1 / H2 |
| --- | ---: | --- | ---: | --- | --- |
| learned（冻结原公式） | 70.16% | 未过 | 72.20% | 过 | 43.44% / 50.50% |
| relmask_cand（R22） | 78.61% | 未过 | 75.37% | 过 | 34.89% / 45.89% |
| entmax15（R26 复现） | 79.41% | 未过 | 75.85% | 过 | 35.67% / 44.33% |
| **entmax_role（主变体）** | **79.02%** | **未过** | **76.55%** | **过** | **38.44% / 44.50%** |
| entmax_multi | 78.98% | 未过 | 76.52% | 过 | 38.11% / 44.50% |
| entmax_tau | 78.24% | 未过 | 75.83% | 过 | 34.44% / 44.44% |
| entmax_alpha12 | 79.88% | 未过 | 75.93% | 过 | 34.39% / 45.44% |
| **entmax_alpha20（sparsemax）** | **81.19%** | **过** | **78.77%** | **过** | **27.67% / 42.78%** |
| entmax_contrast | 79.41% | 未过 | 75.85% | 过 | 35.67% / 44.33% |

## 判读

1. **sparsemax（α=2.0）首次在词典无关约束下通过 H1 门槛（81.19%）**，H2 78.77% 同步通过；H1 相对词面顺序 +31.19pp，下界 +28.76pp。这证实「精确归零的稀疏质量」比 α=1.5 与 softmax 更能消除软分配稀释。
2. **代价是泛化**：sparsemax 未见句式 H1 只有 27.67%，比 learned（43.44%）低 15.8pp——稀疏化放大了对模板特征的过拟合。R24 的教训（cue 开发过门槛、独立 H3 失败）要求我们在这条路上继续加固，而不是直接冻结验收。
3. 主变体 entmax_role（角色监督）79.02%，未见句式 38.44% 为 entmax 族内最高，但未过门槛；角色监督没有带来 H1 突破，只改善了未见句式约 3pp。
4. 跨种子波动依旧：主变体逐种子 71.81 / 70.42 / 94.83；sparsemax 逐种子 H1 为 78.83 / 80.14 / 84.58。seed 1701/2701 在主变体上远低于 3701，而 sparsemax 的种子间差异较小且稳定在 80% 附近。

## 训练与实现

- 三种子 entmax_role 末轮损失：1.86e-05 / 9.53e-05 / 5.48e-06；entmax_multi：8.65e-04 / 1.15e-03 / 7.17e-04；entmax_tau 学到 τ=1.153 / 1.339 / 1.174。
- α-entmax 二分法求解，α=2.0 为 sparsemax（支持集精确置零）。
- 连续性校验通过：relmask_cand 78.61%（R22）、entmax15 79.41%（R26）、learned 70.16%。
- 云端 predictions SHA-256 `d334a6e0…`；依赖冻结 model.py `f85636ff…`、队列 `cf7d3ba2…` 核对通过。

## 结论与下一步

主变体未过 H1（79.02%），但诊断变体 sparsemax 首次通过 H1+H2 开发门槛。R28 围绕 sparsemax 加固泛化：

1. sparsemax_role / sparsemax_multi（角色与多任务损失，目标改善未见句式）；
2. sparsemax_contrast（t_j = w·(h_j − q̄) 词典无关对比归一化）；
3. α=1.8 中间点与 sparsemax_tau；
4. 逐种子诊断：若 1701/2701 持续偏低，评估种子级波动来源。

累计轮次 R27。开发 H1/H2 双达标且未见句式 H1 ≥ 40% 才冻结进入新 H3 验收。

## 归档清单

- [预注册协议](configs/protocol.json) · [训练与推理脚本](code/r27_entmax_combo.py) · [R26 库](code/r26_sharp_cue.py) · [统计脚本](code/r18_stats.py) · [汇总脚本](code/r27_summary.py)
- [逐题预测](results/prediction/predictions.jsonl.gz) · [统计结果](results/stats/metrics.json) · [图级指标 CSV](results/stats/graph_metrics.csv) · [云端日志](logs/)
