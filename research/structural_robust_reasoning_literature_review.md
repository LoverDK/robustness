# 大模型推理的结构鲁棒性：文献版图、证据边界与 EACL 研究切口

## 摘要

本调研围绕三个问题展开：第一，哪些结构变化会导致大模型推理性能下降；第二，现有工作如何区分任务结构、推理轨迹、上下文位置和表面扰动；第三，结构感知的对抗训练与 LoRA 是否构成一个尚未被直接覆盖的研究切口。检索范围覆盖 2020--2026 年的 ACL Anthology、arXiv、Crossref、OpenAlex 记录，并补充了一个奠基性的 2017 年工作。证据形成了四条相互关联但不能混为一谈的路线：组合泛化与结构外推、任务/上下文干扰、CoT 或 rationale 鲁棒性、参数高效与对抗训练。综合结果显示，已有研究已经充分说明标准随机测试集会高估推理能力，且模型对无关子句、结构复杂度、位置变化和 rationale 噪声存在脆弱性；但本次检索没有发现一篇已核验工作同时完成“标签保持的推理结构威胁模型、未见结构组合划分、结构感知的 adversarial LoRA 训练”三者的直接组合。因此，项目可以围绕 **reasoning under structural distribution shift** 收敛，但“新任务构造 + LoRA + 对抗训练”本身不足以构成贡献。真正需要建立的是不改变答案的结构变换、可审计的结构元数据、严格的结构 OOD 测试，以及相对于随机增强和普通 LoRA 的机制性比较。

## 1. 引言与研究问题

大模型在 GSM8K、MATH 和常规多步问答上的高分，并不自动说明模型学会了可迁移的推理结构。等价实例、前提顺序、推理深度、无关信息位置和中间 rationale 的变化，可能在不改变问题答案的情况下改变模型输出。GSM-Symbolic、GSM-Plus、GSM-IC 以及 compositional generalization 文献共同把问题从“模型能否答对一批题”推进为“模型能否在结构分布变化下保持正确”。

本报告冻结以下研究问题：

* **RQ1：失效机制。** 哪些可控的结构变化会产生 reasoning generalization gap？这些下降来自任务图结构、上下文干扰、rationale 噪声，还是表面统计和记忆捷径？
* **RQ2：评估协议。** 如何构造标签保持的结构扰动、训练/测试结构划分和指标，使结构鲁棒性不被长度、数字分布、模板泄漏或语义改变混淆？
* **RQ3：修复机制。** 结构增强、错误导向的 adversarial mining、LoRA/QLoRA 或全量微调，哪一种能改善未见结构组合上的鲁棒性，并保留 clean accuracy？

本报告的核心判断是：**已有证据支持“结构分布偏移下的推理能力”作为问题；尚未支持某个具体 LoRA 训练方案一定有效。** 后者必须通过对照实验验证。

## 2. 检索方法

### 2.1 检索视角

检索分为五条线：

1. **主流方向：** compositional generalization、systematicity、OOD reasoning、task structure。
2. **批评与失败案例：** GSM-Symbolic、GSM-Plus、irrelevant context、distractor、rationale noise、CoT intervention。
3. **邻近方法：** self-ask、symbolic CoT、counterfactual tasks、computation-graph reasoning。
4. **训练方法：** adversarial training、targeted adversarial training、self-play、LoRA、QLoRA、AdaLoRA。
5. **评估方法：** long-context position effects、side-question 相关的多问题和干扰上下文。

关键词组合包括 `compositional generalization + reasoning/LLM`、`structural OOD + semantic parsing`、`GSM-Symbolic/GSM-Plus`、`noisy rationale/CoT intervention`、`adversarial reasoning`、`LoRA reasoning`、`irrelevant context/distractor` 和 `self-ask follow-up questions`。

### 2.2 核验与证据等级

每条核心引用先用完整标题和作者在 ACL Anthology、Crossref、OpenAlex 或 arXiv 记录中核验存在性，再用摘要、论文元数据或公开正文核对研究对象。没有正文支持的地方只陈述标题和元数据层面的结论，不延伸为方法细节。2025--2026 年的部分工作仍是预印本，因此只把它们作为边界证据，不据此宣称领域共识。

本次检索得到数十条候选记录；去重和核验后，以约 30 篇核心工作进行综合，另有若干只用于发现关键词的记录未进入结论。检索受 arXiv 和 OpenAlex 的限流影响，部分晚近预印本只接受元数据级使用；因此“未检索到直接组合”应理解为本次范围内的结果，而不是严格的优先权证明。

## 3. 分类框架

现有文献可按“被扰动的对象”划分为四个互斥主分支，再按“泛化关系”分析每个分支：

| 分支 | 被扰动的对象 | 典型问题 | 与本项目的关系 |
|---|---|---|---|
| A | 任务/推理图结构 | 已见组件能否组成未见结构，长度和深度能否外推 | 核心主线 |
| B | 输入上下文和干扰 | 无关子句、位置、side question 是否改变主答案 | 核心威胁模型候选 |
| C | rationale/CoT 轨迹 | 中间步骤有噪声、被篡改或不忠实时，模型是否仍能推理 | 重要邻近方向，不能与 A 混称 |
| D | 训练与适配机制 | 对抗样本、LoRA、量化和结构先验能否修复上述缺陷 | 方法层 |

泛化关系则分为：同分布实例变化、语义保持的结构变换、未见组件组合、长度/复杂度外推和跨任务迁移。一个实验必须明确自己属于哪一类；否则“robustness”会同时包含多个来源不同的性能下降。

## 4. 分支 A：组合泛化与结构外推

### 4.1 组合泛化提供了最成熟的结构语言

Hupkes 等人的综述式框架把 compositionality 分解为 systematicity、productivity、substitutivity 等属性，为“结构”提供了比模板改写更严格的词汇。Oren 等人和 Herzig 与 Berant 则在 semantic parsing 中显式构造组合泛化划分：训练集包含基础组件，测试集包含新的结构组合；后者通过 span/tree 结构表示显著改善 compositional split，而随机划分的差异较小。这两条证据共同说明，随机测试集主要检验组件识别，不能充分检验结构组合。

Qiu 等人进一步表明，扩大模型规模并不会自动解决 compositional OOD：在其 semantic parsing 设置中，fine-tuning 的 scaling 曲线可能平坦或变差，而 prompt-tuning 和 in-context learning 的趋势不同。Press 等人对 single-hop 与 multi-hop QA 的比较也指出，模型规模提高并不能消除 compositionality gap，CoT 和 self-ask 可以缩小部分差距。换言之，规模是一个条件变量，而不是结构泛化的机制解释。

### 4.2 结构复杂度和长度是可控但容易混淆的变量

Dziri 等人的 Faith and Fate 用 computation graph 和中间子程序分析多步推理，指出模型常把复杂推理压缩为线性化子图匹配，性能随任务复杂度快速下降。Jelassi 等人研究 arithmetic Transformer 的长度外推，Press 等人提出 Position Coupling 以利用任务结构改善外推；这些工作支持“深度、长度和位置是结构变量”，但它们主要是合成算术或语义解析，不等价于自然语言推理中的全部结构鲁棒性。

Xu 与 Wang 的 SPOR 将 compositional generalization 划分为 systematicity、productivity、order invariance 和 rule learnability。对你的项目而言，**order invariance** 和 **productivity** 最有用：前者可定义不改变答案的顺序变换，后者可定义训练中未出现的深度或组件组合。BIG-Bench Hard 则提醒我们，任务族选择和推理格式本身会改变结论，因此不能只依赖一个数学任务。

### 4.3 代表性工作比较

| 工作 | 结构对象 | 泛化划分/干预 | 主要启示 | 主要限制 |
|---|---|---|---|---|
| Hupkes et al., 2020 | compositionality 属性 | systematicity、productivity 等概念分解 | 为结构泛化提供统一词汇 | 不是 LLM 推理训练方法 |
| Oren et al., 2020 | 语义解析结构 | 新结构组合的 compositional split | 结构划分比随机划分更能揭示问题 | 任务是 semantic parsing |
| Herzig & Berant, 2021 | span/tree 表示 | compositional generalization split | 显式结构表示可以改善 OOD | 不是自由形式 reasoning |
| Qiu et al., 2022 | 模型规模与训练方式 | CFQ 等 compositional OOD | 规模本身不是充分解法 | 结论依赖 semantic parsing 设置 |
| Press et al., 2023 | single-hop/multi-hop | 组合复杂度与 self-ask | follow-up question 可缩小组合差距 | 主要是 prompting |
| Dziri et al., 2023 | computation graph | 子程序和复杂度变化 | 应用计算图而非模板计数 | 合成/受控任务占主导 |
| Xu & Wang, 2024 | 四类组合泛化 | systematicity、productivity、order invariance、rule learnability | 可直接借鉴评估维度 | data-to-text 任务 |

**分支结论：** 你的项目不能只说“构造不同结构的题”。更精确的贡献对象应是一个结构变换族，以及训练/测试之间明确的结构组合关系。

## 5. 分支 B：输入结构、干扰信息与 side question

### 5.1 GSM 系列把结构脆弱性变成了可测量现象

Shi 等人的 GSM-IC 通过插入无关上下文测试数学推理，展示了模型对干扰信息的敏感性。Mirzadeh 等人的 GSM-Symbolic 使用符号模板生成等价实例，发现仅改变数字实例就可能产生显著方差，随着子句数量增加性能下降，加入一个看似相关但对答案无用的子句也会导致大幅下降。Li 等人的 GSM-Plus 扩展了数学问题鲁棒性评估。它们共同支持一个判断：标准 GSM8K 式静态测试集可能把模板记忆和真正的推理混在一起。

但这条证据不能直接等同于“结构缺陷”。2026 年对 GSM-Symbolic 的批评性再分析指出，数字分布、统计显著性和模型特定的变量绑定/算术失败可能解释部分结果。MATH-Perturb 也区分了简单表面扰动和改变解法性质的困难扰动。因此你的数据构造必须把数字范围、token 长度、词汇、模板和结构图分层控制，并对结构效果做配对检验。

### 5.2 side question 的现有位置

`side question` 不是一个已统一命名的 benchmark 类别。其最近邻是三种工作：Jia 与 Liang 在阅读理解中插入保持答案不变的干扰句；GSM-IC/GSM-Symbolic 插入无关或看似相关的子句；Press 等人的 self-ask 使用显式 follow-up question 缩小多跳组合差距。前两类把辅助信息当作干扰，后一类把问题当作必要的分解步骤，不能混为同一现象。

本次检索没有发现一个已核验的 2022--2026 benchmark，专门把“在推理链不同位置插入保持主答案不变的辅助问题”作为独立轴系统评估。因此，side question 可以成为你的一个聚焦切口，但应使用保守表述：**本次检索未发现该精确定义下的 benchmark**，而不是宣称“首个”。

### 5.3 位置和长上下文会混淆 side question 效果

Liu 等人的 Lost in the Middle 发现模型对长上下文中的信息位置存在 U 型敏感性，LooGLE 则从更广泛的长上下文理解和推理角度提供基准。这些工作提示，side question 不能只改变数量，还要控制位置：开头、中间、结尾，以及与主问题的实体和词汇相似度。否则观察到的下降可能只是位置效应或上下文长度效应。

建议把 side question 定义为：**插入到主任务上下文中的、对主答案非必要且保持主标签不变的辅助问题**。至少记录数量、位置、与主问题的语义相似度、是否共享实体、是否需要同时回答。主指标可定义为：

\[
\text{Interference Cost}=\text{Acc}_{clean}-\text{Acc}_{side-question}.
\]

### 5.4 代表性工作比较

| 工作 | 干扰形式 | 是否保持主答案 | 研究价值 | 需要控制的混淆 |
|---|---|---|---|---|
| Jia & Liang, 2017 | 干扰句 | 是 | side-question/distractor 的概念祖先 | 阅读理解与推理任务不同 |
| Shi et al., 2023 | 无关上下文 | 是 | 证明无关信息会伤害数学推理 | 长度和语义相似度 |
| Mirzadeh et al., 2024 | 等价实例、额外子句 | 设计上是 | 结构/实例变化的强基线 | 数字分布与统计方法 |
| Liu et al., 2024 | 证据位置 | 是 | 提供位置控制依据 | 长上下文检索而非多步推理 |
| Press et al., 2023 | self-ask follow-up | 通常服务主答案 | 说明“问题”也可能是必要分解 | 不能当作干扰问题 |

**分支结论：** side question 具有新颖的评估潜力，但必须先将其与 distractor、必要的 sub-question 和 prompt injection 分开定义。

## 6. 分支 C：CoT/rationale 鲁棒性与忠实性

### 6.1 推理轨迹不是任务结构

Zhou 等人的 NoRa 研究在上下文中的 rationale 引入无关或错误思路，发现模型性能下降，并提出 CD-CoT 进行去噪。R3 Prompting 也把 noisy context 作为 prompt-time 处理对象。Turpin 等人和 Lanham 等人的工作则说明，CoT 可能受到选项顺序、提示干预和后验合理化影响；模型生成的解释不一定是产生答案的真实因果路径。

von Recum 等人对 reasoning LLM 的 CoT 进行直接干预，进一步把“轨迹被改变后是否仍能答对”和“模型是否从轨迹恢复”分开。FRODO（即 Paul 等人的 *Making Reasoning Matter*）、Faithful CoT 和 SymbCoT 通过因果中介、符号链或验证器研究忠实性与推理监督。Tanneru 等人关于 faithful CoT 难度的分析则提醒，简单 ICL、微调或 activation editing 的跨 benchmark 泛化可能有限。

这些工作与任务结构鲁棒性相关，却回答不同问题：

| 问题 | 被改变的对象 | 你的项目中应否合并 |
|---|---|---|
| noisy rationale | 中间解释/示例轨迹 | 可作邻近基线，不应作为主威胁模型 |
| CoT intervention | 模型已经生成的推理过程 | 可作机制分析，但需独立实验 |
| task-structure shift | 输入任务的推理图和组织方式 | 建议作为主线 |
| distractor context | 输入中的无关信息 | 可作为结构威胁模型的一个分支 |

### 6.2 为什么要把 faithfulness 放在机制分析中

如果结构对抗训练提高了最终答案准确率，却只是让模型记住某种格式，不能据此声称“推理能力提高”。至少需要比较：答案正确率、结构 OOD gap、rationale 与答案的一致性，以及在结构干预后的恢复行为。否则模型可能只学会了防御模板，而没有获得结构不变的推理能力。

**分支结论：** CoT 鲁棒性文献支持“推理表面形式和真实机制可能分离”，因此适合用来设计诊断实验；它不能直接替代任务结构泛化实验。

## 7. 分支 D：Adversarial training、LoRA 与参数高效适配

### 7.1 对抗训练的可借鉴部分

Pereira 等人的 Targeted Adversarial Training 按模型当前错误选择训练重点，提供了“错误导向 hard-example mining”的直接方法先例。ANLI 通过人和模型参与的对抗式数据收集构造更难的 NLI benchmark，说明 adversarial data 可以暴露 shortcut，但其扰动不一定保持语义。Cheng 等人的 Self-playing Adversarial Language Game 把对抗者和求解者放进自博弈，属于最接近推理能力提升的相邻路线；它优化的是对抗游戏和难度，而不是你需要的标签保持结构变换。

因此，项目中的 adversarial training 不应定义为“随机增加困难题”，而应定义为：

\[
\min_\theta\;\mathbb{E}_{(x,y)}\left[\max_{\delta\in\Delta_{LP}}\mathcal{L}(f_\theta(T_\delta(x)),y)\right],
\]

其中 \(\Delta_{LP}\) 只包含 label-preserving 的结构变换。可以先做离线候选生成和当前模型错误筛选，再考虑在线 adversary；后一种方案成本和稳定性风险更高。

### 7.2 LoRA 的合理角色

Hu 等人的 LoRA 冻结基础模型，只训练低秩更新，是最基本的参数高效基线。QLoRA 进一步把基础模型量化，适合在有限硬件上并行训练多个结构适配器；AdaLoRA 则按参数重要性分配低秩预算。它们都没有直接证明结构鲁棒推理，因此在论文中应首先作为实验平台和对照条件。

Tina 以及 LoRA 用于 reasoning/safety alignment 的近期工作说明，低秩更新可以用于推理或后训练，但目标分别是低成本 reasoning 和 safety tax，而不是结构分布偏移。Rank-1 LoRA 的可解释性工作也只能说明低秩更新可能承载一部分推理信号，不能替代你的结构泛化证据。

### 7.3 必须的训练对照

| 训练条件 | 要回答的问题 |
|---|---|
| frozen/base model | 结构脆弱性是否本来就存在 |
| ordinary LoRA | 低秩微调本身是否足够 |
| random structure augmentation + LoRA | 提升是否只是数据量效应 |
| targeted hard-structure mining + LoRA | 错误导向是否有额外作用 |
| adversarial structural LoRA | 结构威胁模型是否带来机制性收益 |
| full fine-tuning | LoRA 是否牺牲鲁棒性或只是低成本替代 |
| clean-only 与 mixed clean/robust | 鲁棒性与 clean accuracy 的权衡 |

**分支结论：** LoRA 是可行的执行工具，也可能成为方法贡献，但只有当 rank、target modules、训练 token、步数和量化状态严格匹配时，才能把增益归因于结构对抗机制。

## 8. 跨分支综合：文献已经支持什么

第一，**标准随机评估不足以测量结构推理。** 组合泛化工作通过新结构组合划分证明了这一点；GSM-Symbolic 和 GSM-Plus 则在数学推理上用等价实例和干扰子句展示了同一问题。两类工作任务不同，但都指向“组件正确不等于组合正确”。

第二，**扰动对象决定机制解释。** NoRa 改变的是 rationale，Lost in the Middle 改变的是信息位置，GSM-IC/GSM-Symbolic 改变的是任务上下文或实例，SPOR 关注的是组合属性。若将它们统一称作“reasoning robustness”，论文会失去可证伪性。

第三，**规模和通用微调不是充分解法。** Qiu 等人的规模研究和 Dziri 等人的计算图分析都对“更大的模型自然学会结构”提出限制；LoRA 文献只说明高效适配可行，不能推出结构鲁棒性会自动提升。

第四，**side question 是一个有文献邻接、但尚未标准化的轴。** 其概念基础来自 distractor、位置敏感性和 self-ask，但必要子问题与无关问题的功能相反。这个差异本身可以成为论文中的 threat-model 设计，而不是把 side question 当作一个模糊关键词。

## 9. 研究空白与 EACL 可行切口

在本次检索范围内，尚未发现已核验工作直接同时具备以下三项：

1. **语义/标签保持的推理结构威胁模型**，明确哪些结构变换不改变答案；
2. **未见结构组合的训练/测试划分**，而非随机实例划分；
3. **结构感知的 adversarial PEFT/LoRA 训练**，并与随机增强、普通 LoRA 和 full fine-tuning 做等预算比较。

这构成的是一个“组合式空白”，不是“每个组件都没人做过”。因此论文主张应写成：

> We study reasoning under structural distribution shift and test whether error-guided, label-preserving structural perturbations can improve unseen-composition robustness through parameter-efficient adaptation.

中文可表述为：

> 本文研究结构分布偏移下的推理泛化，并验证标签保持的错误导向结构扰动能否通过参数高效适配提升未见结构组合上的鲁棒性。

不建议把贡献写成“提出一个新 benchmark、一个新 adversary、一个新 LoRA、一个理论界和一个通用推理框架”。对于 EACL，聚焦两个贡献更稳妥：

* 一个可验证的结构威胁模型和结构 OOD 评估协议；
* 一个错误导向的结构增强/LoRA 方法，以及跨结构、跨任务的受控实验。

## 10. 建议的实验设计

### 10.1 结构变换

第一版只选三类：

1. **顺序变换：** 重排不依赖顺序的前提或推理步骤；
2. **复杂度变换：** 增加推理深度、分支或嵌套，但保持标签可由符号求解器验证；
3. **side-question 变换：** 在开头、中间、结尾插入非必要辅助问题，控制数量和语义相似度。

每个样本保存推理图和结构元数据：depth、branching、operator sequence、side-question position、token length、surface template 和 answer。

### 10.2 数据划分

至少使用四类测试集：

| 划分 | 含义 |
|---|---|
| IID | 训练和测试共享结构分布，只改变实例内容 |
| Surface-OOD | 表达、名字或模板变化，但推理结构相同 |
| Structure-OOD | 测试结构类型训练中未出现 |
| Composition-OOD | 基础组件已见，但组件组合训练中未出现 |

测试集还应按结构组报告 worst-group accuracy，避免平均值掩盖某一种结构完全失效。

### 10.3 指标

核心指标建议包括：

\[
\text{Structural Gap}=R_{OOD}-R_{IID},
\]

以及 clean accuracy、robust accuracy、worst-group accuracy、Interference Cost、不同扰动强度下的性能曲线和训练/推理成本。若使用 rationale，另报告 rationale-answer consistency，但不要把语言流畅度当作 faithfulness。

### 10.4 最小可行实验顺序

1. **无训练诊断：** 在两个数学任务和一个非数学推理任务上测量 clean、结构变换和未见组合差距。
2. **离线 hard mining：** 生成多个合法变体，用当前模型错误率筛选困难结构。
3. **LoRA 对照：** ordinary LoRA、random augmentation + LoRA、hard mining + LoRA。
4. **归因与稳健性：** 加 full fine-tuning、不同 rank、不同 seed、不同基础模型和 clean/robust trade-off。
5. **机制分析：** 检查提升是否只出现在训练过的结构，是否迁移到完全未见组合，side-question 位置是否影响结果。

## 11. 仍未解决的问题

### 11.1 标签保持如何形式化

顺序重排并不总是安全：自然语言中的代词、因果顺序和量词可能让重排改变语义。必须使用符号层生成、规则验证和人工抽查，而不能直接用 LLM 判断“答案应该不变”。

### 11.2 结构鲁棒与难度提升如何分离

更深的推理图通常也更难。实验需要匹配 clean 难度，或报告结构变化相对于难度变化的增量效果。否则 adversarial training 可能只是增加了训练难度。

### 11.3 真实语言任务的迁移性

合成任务能提供干净的结构控制，但不能自动代表自然语言。建议使用“合成任务发现机制 + 自然语言任务验证”的两阶段设计，并承认二者之间的外部有效性边界。

### 11.4 训练收益是否来自更多数据

结构对抗训练通常会产生更多样本。没有等 token、等步数和 random augmentation 对照，就不能把收益归因于 adversary。这个问题必须在实验设计阶段固定，而不是论文写作阶段补充。

### 11.5 2026 年新推理模型的稳定性

近期 reasoning LLM、CoT intervention 和 LoRA 后训练工作变化很快。论文提交前应重新检查基础模型版本和最新结构鲁棒性结果，尤其不要把 2024 年的 GSM 结论直接外推到所有 2026 年模型。

## 12. 结论：对你当前项目的建议

### RQ1 的回答

已有证据一致提示：大模型在等价实例、无关子句、结构复杂度、上下文位置和 rationale 噪声下会出现不同形式的性能下降。但这些证据来自不同对象，不能合并成一个未经定义的“推理鲁棒性”。

### RQ2 的回答

最可靠的评估方式是把任务表示为可验证的结构图，构造 label-preserving transforms，并使用 structure-OOD 与 composition-OOD 划分。SPOR 的四类属性、Faith and Fate 的 computation graph、GSM-Symbolic 的符号实例化和 Lost in the Middle 的位置控制可以共同提供设计依据。

### RQ3 的回答

LoRA/QLoRA 能显著降低多组适配实验的成本，Targeted Adversarial Training 和 Self-play 提供 hard-example 选择的相邻先例；但本次检索没有直接证据表明 adversarial LoRA 会提升未见结构泛化。这个结论必须由等预算、跨结构、跨任务实验获得。

### 最终研究切口

建议把项目收敛为：

> **Reasoning under Structural Distribution Shift: Label-Preserving Structural Adversarial Training with Parameter-Efficient Adaptation**

第一阶段不要同时做新的理论、完整 benchmark 和复杂 online adversary。先证明三件事：结构差距真实存在；它不是数字、长度或模板泄漏；错误导向的结构 LoRA 能改善未见组合，并且不以明显损失 clean accuracy 为代价。

## 参考文献

[1] Hupkes, D., Dankers, V., Mul, M., and Bruni, E., “Compositionality Decomposed: How Do Neural Networks Generalise?,” Journal of Artificial Intelligence Research, 2020.

[2] Oren, J., Herzig, J., Gupta, K., Gardner, M., and Berant, J., “Improving Compositional Generalization in Semantic Parsing,” Findings of EMNLP, 2020.

[3] Herzig, J. and Berant, J., “Span-based Semantic Parsing for Compositional Generalization,” ACL-IJCNLP, 2021.

[4] Qiu, L., Shaw, P., Pasupat, P., Shi, T., Herzig, J., Pitler, E., Sha, F., and Toutanova, K., “Evaluating the Impact of Model Scale for Compositional Generalization in Semantic Parsing,” EMNLP, 2022.

[5] Press, O., Zhang, M., Min, S., Schmidt, E., Smith, N. A., and Lewis, M., “Measuring and Narrowing the Compositionality Gap in Language Models,” Findings of EMNLP, 2023.

[6] Dziri, N. et al., “Faith and Fate: Limits of Transformers on Compositionality,” arXiv preprint, 2023.

[7] Suzgun, M. et al., “Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them,” Findings of ACL, 2023.

[8] Jelassi, S., d’Ascoli, S., Domingo-Enrich, C., Wu, Y., Li, Y., and Charton, F., “Length Generalization in Arithmetic Transformers,” arXiv preprint, 2023.

[9] Xu, H. and Wang, W., “SPOR: A Comprehensive and Practical Evaluation Method for Compositional Generalization in Data-to-Text Generation,” ACL, 2024.

[10] Cho, J., Cha, J., Awasthi, P., Bhojanapalli, S., Gupta, A., and Yun, S.-Y., “Position Coupling: Improving Length Generalization of Arithmetic Transformers Using Task Structure,” NeurIPS, 2024.

[11] Mirzadeh, I., Alizadeh, K., Shahrokhi, H., Tuzel, O., Bengio, S., and Farajtabar, M., “GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models,” arXiv preprint, 2024.

[12] Li, Q., Cui, L., Zhao, X., Kong, L., and Bi, W., “GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers,” ACL, 2024.

[13] Shi, F., Chen, X., Misra, K., Scales, N. B., Dohan, D., H. Ed, Schärli, N., and Zhou, D., “Large Language Models Can Be Easily Distracted by Irrelevant Context,” arXiv preprint, 2023.

[14] Jia, R. and Liang, P., “Adversarial Examples for Evaluating Reading Comprehension Systems,” EMNLP, 2017.

[15] Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., and Liang, P., “Lost in the Middle: How Language Models Use Long Contexts,” TACL, 2024.

[16] Li, J. et al., “LooGLE: Can Long-Context Language Models Understand Long Contexts?,” ACL, 2024.

[17] Zhou, Z., Tao, R., Zhu, J., Luo, Y., Wang, Z., and Han, B., “Can Language Models Perform Robust Reasoning in Chain-of-thought Prompting with Noisy Rationales?,” NeurIPS, 2024.

[18] Turpin, M., Michael, J., Perez, E., and Bowman, S. R., “Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting,” NeurIPS, 2023.

[19] Lanham, T. et al., “Measuring Faithfulness in Chain-of-Thought Reasoning,” arXiv preprint, 2023.

[20] Paul, D., West, R., Bosselut, A., and Faltings, B., “Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning,” Findings of EMNLP, 2024.

[21] Lyu, Q. et al., “Faithful Chain-of-Thought Reasoning,” arXiv preprint, 2023.

[22] Xu, J., Fei, H., Pan, L., Liu, Q., Lee, M.-L., and Hsu, W., “Faithful Logical Reasoning via Symbolic Chain-of-Thought,” ACL, 2024.

[23] Tanneru, S. H. et al., “On the Hardness of Faithful Chain-of-Thought Reasoning,” arXiv preprint, 2024.

[24] Cheng, P. et al., “Self-playing Adversarial Language Game Enhances LLM Reasoning,” NeurIPS, 2024.

[25] Pereira, J. et al., “Targeted Adversarial Training for Natural Language Understanding,” NAACL, 2021.

[26] Nie, Y. et al., “Adversarial NLI: A New Benchmark for Natural Language Understanding,” ACL, 2020.

[27] Wu, T. et al., “Reasoning or Reciting? Exploring the Capabilities and Limitations of Language Models Through Counterfactual Tasks,” NAACL, 2024.

[28] Hu, E. J. et al., “LoRA: Low-Rank Adaptation of Large Language Models,” ICLR, 2022.

[29] Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L., “QLoRA: Efficient Finetuning of Quantized LLMs,” NeurIPS, 2023.

[30] Zhang, Q. et al., “AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning,” ICLR, 2023.

[31] Wang, S. et al., “Tina: Tiny Reasoning Models via LoRA,” arXiv preprint, 2025.

[32] Xue, Y. and Mirzasoleiman, B., “LoRA is All You Need for Safety Alignment of Reasoning LLMs,” arXiv preprint, 2025.

[33] von Recum, A., Girrbach, L., and Akata, Z., “Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?,” arXiv preprint, 2026.

[34] Press, O., Smith, N. A., and Lewis, M., “Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation,” arXiv preprint, 2021.

[35] Długosz, D. A., Oliveira, A., and Díaz-Rodríguez, N., “The Importance of Being Statistically Earnest: A Critical Re-evaluation of GSM-Symbolic,” arXiv preprint, 2026.
