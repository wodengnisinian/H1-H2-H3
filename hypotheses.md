
## R01  2026-11-24T09:44:01.892591+00:00

归档完整，历史训练与开发图的人物集合分离。候选首位可能与题型相关。

85 个输入文件逐字节匹配。检查图的实体集合、训练开发交集和金标位置分布；报告直接首位的分题型基数与命中。

## R02  2026-11-24T09:44:02.523453+00:00

同一数据、冻结检查点与旧协议可在独立服务器重放原问与四问。

三个种子的六组封存模型预测及直接首位逐条匹配。配对下界和历史数值按相同抽样实现核对。

## R03  2026-11-24T09:45:40.791842+00:00

Frozen representations encode first-relation identity on held-out wording. A matched linear question-only readout exceeds its randomized-label control.

Constrained two-relation probe accuracy exceeds shuffled-label accuracy by at least 5pp in fresh, template and combination original questions. Same ridge penalty and same training examples in both fits. Learned word mass identifies the first relation only when its mass exceeds the competing relation; exact ties are reported as 0.5.

## R04  2026-11-24T09:46:25.573092+00:00

The closed gate and per-token margin extrema impose an empirical ceiling on attainable first-hop accuracy.

Before training a new head, compare reachable ceiling with max(first,pool)+5pp. Separate negative strict upper bounds, exact zero ties, uncovered candidates and open/closed gates. Label-informed oracle and mathematical ceiling are diagnostics.

## R05  2026-11-24T09:46:54.298451+00:00

Position-specific learned word weights contribute to actual ranking beyond uniform, random and shuffled alternatives.

Learned beats all three weighting controls on fresh original and fresh quartet. Report all seeds. Random vector is Gaussian with learned norm. Shuffled p deterministically permutes valid positions in each example. Candidate swap must preserve p and permute logits within 2e-5.

## R06  2026-11-24T09:47:39.338591+00:00

相同的64维权重训练预算下，四个冻结查询的方向评分优于单个冻结查询。

在新开发原问和四问上，四查询均严格优于单查询。单查询以四个冻结查询的算术平均作为一个查询，只做一次词位置 softmax。

## R07  2026-11-24T09:49:14.585775+00:00

Old context-only weights overfit original syntax. A question-only clause-order feature helps rank relations across unseen wording with fixed H2.

Compare replacing the 64th hidden coordinate with an annotated ordinal cue feature on precisely the same examples and loss; retain 64 parameters. Fresh original >=5pp gain over each baseline, fresh paraphrase >0 against each baseline and fresh quartet >first, >=pool. Report separate combination and template challenge metrics.

## R08  2026-11-24T09:52:01.926441+00:00

R07 对首位仅提高0.28个百分点。把共享权重限制在题面的关系词上后，直接监督首步关系与候选损失的训练效果是否不同？

主候选 cue_role 和 cue_candidate 同结构、同参数量、同数据和批顺序。在新开发原问的关系识别与最终排序两者上均严格更高才支持训练目标假设。进入 H3 的标准另由总协议规定。

## R13  2026-11-24T09:55:35.235689+00:00

R08 在首步角色识别上取得的增益可在新进程从零初始化重新训练得到。

复跑全部三个种子，核验12份检查点、参数、逐题预测和错误记录（压缩文件比较解压字节），并核对两个指标体系及配对区间。

## R12  2026-11-24T09:57:50.297591+00:00

R08 选中首步关系达到100%，而软分配仍限制最终纠错。仅改变软分配到最高质量关系硬分配能否提升同题表现？

主候选只使用 R08 冻结、无标签的关系选择。原问相对同权重软对照的配对点估计和单侧下界均大于零。H3开发门槛维持原设定。

## R09  2026-11-24T09:59:44.537784+00:00

R12 达到开发门槛后，检验冻结模型在新图和新句式上的首跳增益。

抽取 4,800 张独立合成图，三个固定种子、每类 1,600 图；相对池化与直接首位均至少提高 5 个百分点，题型内按图配对重抽 10,000 次的两个单侧 95% 下界均大于零。每个图及四问同组，只有一次正式开封。

## R10  2026-11-24T10:00:59.834072+00:00

当候选同起点、同关系时，读取后继边有助于首跳选择；移除后继信息使配对孪生图不可区分。

按 challenge.json 预设的准确率、四问、仅换第二关系的双对率和按孪生组重抽的区间同时判定。四候选的独立挑战不参与经典 H3 模型选择。

## R11  2026-11-24T10:01:11.051844+00:00

当两种相反的计划同时出现，仅交换已采用与已放弃的状态词时，冻结读题器是否随实际采用的方案改变首跳？

每对的图、词袋、候选顺序相同而正确边互异。检验神经模型实际输入碰撞，按图配对估计双正确率；适用范围及阈值来自 scope.json。仅使用开发集。

## R14 开发对照

在新开发原问中，仅改变证据条目的输入位置并映回原事实编号后，冻结 R12 的首跳选择保持不变。将候选槽位和检索前事实顺序作为两个独立干预分别检验。

逐种子原序对照须复现已封存 R12 预测、金标和池化及首位基线，评分容差 2e-5。在不改变候选成员的干预中，如任一 R12 选中的事实发生变化，即反驳严格的槽位不变性。在倒置全部事实并重建候选后，如出现任一改变，即反驳端到端严格不变性。个体不变性和平均准确率效应分开报告，准确率升降均保留。

## R15 开发对照

R14 的位置敏感性集中在检索边界同分处。若在两项既有语义检索键之后，用事实内容的稳定键取代原行号，则即使输入行序改变，也应保留同一对有序候选、同一条 R12 最终选择。

主判据为对全部 600 张开发图，在原序、倒序和预先固定的逐图随机排列中，有序候选映回相同事实；全部 1800 个图–种子组合的 R12 选择也须跨三个排列完全相同。任一例失败即否定严格不变性。报告旧规则、两个基线、金标覆盖和准确率的代价；准确率仅作描述，无新验收或优越性判决。

## R16 新开发队列

论文原始共享词权重在一组全新开发图上同时达到 H1 和 H2 的预置门槛。 事前主判据见本轮 configs/protocol.json。

## R17 新开发队列

R08 的已训练关系读头配合 R12 冻结硬选择策略，在同一批开发图上同时达到两个门槛。 事前主判据见本轮 configs/protocol.json。
