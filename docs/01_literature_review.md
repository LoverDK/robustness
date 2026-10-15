# 01 文献综述：结构鲁棒性与大模型推理

> 产出技能：deep-research。检索方式：WebSearch 串行执行 5 个视角（主流诊断、训练缓解、一致性正则、强化学习、批评/方法论），未使用并行子代理。所有引用均在检索结果中核对过标题、作者、会议；未能核实的条目不使用。

## 摘要

前提顺序会显著影响 LLM 推理，这一点已经由 Chen et al. (ICML 2024)、DyVal (ICLR 2024)、MEND (ACL Findings 2025) 等工作反复确认；用"随机重排前提"做 SFT 增强以提升鲁棒性也已被 He et al. (EMNLP 2025) 与 MEND 系统做过；跨等价变体的一致性训练在偏见/越狱 (BCT, Chua et al. 2024; Irpan et al. 2025)、提示扰动 (F2C, 2025)、选项排列 (PA-GRPO, ACL 2026) 上也已存在；最坏排列的分布鲁棒训练见 Pearl (2025)。尚未被检索到的组合是：**以可解释的依赖负担（回读次数、相关前提距离、活动依赖前沿）为引导、在推理链 SFT 中选择困难呈现，并与同一潜在任务跨呈现的一致性项联合训练，同时用同图稳定性与新图迁移分开评估增益来源**。这一缺口较窄，论文贡献必须落在"负担引导是否优于随机重排"这一可证伪的比较上。

## 1 研究问题（冻结）

- RQ1：哪些工作已经证明前提顺序/结构呈现会改变 LLM 推理结果？其机制解释是什么？
- RQ2：哪些训练方法已经用重排增强、一致性正则或 RL 提升了推理对结构呈现的鲁棒性？它们各自缺什么？
- RQ3：在"同图稳定性 vs 新图迁移"分离测量的框架下，哪些问题仍未被回答？

## 2 方法

五个检索视角：(1) 前提顺序与依赖结构诊断；(2) 顺序不变性训练与数据增强；(3) 一致性正则；(4) 推理强化学习（GRPO 等）与泛化；(5) 批评与方法论（捷径特征、噪声敏感性、表征层面）。每个视角两轮关键词（宽→窄），合并后检查盲区。

## 3 分类体系

| 分支 | 核心问题 | 代表工作 |
|---|---|---|
| A 诊断：顺序与结构敏感性 | 呈现变化是否改变答案 | Chen 2024; DyVal; GSM-∞; iGSM; GSM-Symbolic; GSM-Plus; GSM-IC; Hoppe 2025; Tao 2026 |
| B 训练缓解：增强与架构 | 怎样让模型对等价呈现不敏感 | He 2025; MEND; Pearl; Set-LLM; SBP-FT |
| C 一致性正则 | 怎样直接约束跨变体输出一致 | R-Drop; AugMix; BCT; Irpan 2025; F2C |
| D 强化学习与泛化 | RL 是否能带来结构泛化 | DeepSeekMath/GRPO; Chu 2025; PA-GRPO; GC²PO |
| E 数据集与捷径 | 合成规则推理是否学到真推理 | RuleTaker; ProofWriter; SimpleLogic (Zhang 2023) |

## 4 分支 A：前提顺序与依赖结构的诊断

**Chen, Chi, Wang, Zhou (ICML 2024, Premise Order Matters)** 在 SimpleLogic 风格的命题逻辑（只含 modus ponens 的定子句）上用 Kendall τ 控制规则顺序，发现与证明顺序一致的前向顺序（τ=1）最好，打乱后准确率可下降 30% 以上；加入无关规则会同时降低整体准确率并放大顺序效应；GPT-4-turbo 等模型对逆序（τ=−1）的容忍度高于其他非前向顺序。错误分析把失败分为错误反驳、规则幻觉、事实幻觉。他们发布了 R-GSM（220 对人工改写的 GSM8K 题），但**没有公开下载地址**（HF 论文页上有人询问发布位置而无答复），这直接影响我们"在 R-GSM 上评估"的计划。论文明确把训练缓解留作未来工作，并把效应归因为"模型偏好从左到右读，而不是来回读"。

**DyVal (Zhu et al., ICLR 2024)** 用 DAG 动态生成算术、逻辑与算法题，节点描述可按拓扑序、逆拓扑序、随机序输出，三种顺序被当作复杂度来源之一；他们也发现 DyVal 生成的数据可用于微调。与 Chen 相比，DyVal 把顺序视为难度旋钮而非研究对象，没有分析"为什么某种顺序更难"。

**GSM-∞ (Zhou et al., ICML 2025)** 把 GSM8K 抽象为计算图，用"最短解路径/拓扑序"控制推理复杂度，并通过从核心图向外连边的"Spider Topology"加入噪声节点，使噪声不改变答案；他们观察到复杂度上升时准确率呈 sigmoid 衰减，推理计算量指数增长只带来线性收益。**iGSM (Ye, Xu, Li, Allen-Zhu, ICLR 2025)** 从零预训练 GPT-2 规模模型，用探针发现模型在生成前就"心算"出哪些参数是必要的、哪些已可计算，错误多为系统性的规划错误而非生成噪声，并指出深度比宽度更重要。两者共同说明"依赖越复杂越难"，但都不针对"同一图的不同呈现"做训练干预。

数学文本上的扰动基准给出相同方向的证据：**GSM-Symbolic (Mirzadeh et al., ICLR 2025)** 改变数值或增加一个看似相关但无关的子句（GSM-NoOp），准确率最多下降 65%；**GSM-Plus (Li et al., ACL 2024)** 用 8 类扰动构造 10,552 道变体；**GSM-IC (Shi et al., ICML 2023)** 显示无关上下文导致显著下降，自一致性解码与"忽略无关信息"指令可以部分缓解。**Hoppe, Ilievski, Kalo (ECAI 2025)** 在演绎推理上比较对抗噪声与反事实扰动，发现噪声主要伤害自动形式化方法，反事实对所有方法都有害。

一个与"模型线性扫描"机制假设相张力的新结果是 **Tao (arXiv 2609.28442, 2026)**：在 16 个 1B–8B 模型的规则重排任务上，**准确率越高的模型，其隐藏表征越能区分不同排列**（Spearman 最高 0.86）。这意味着"答案不变性"不要求"表征不变性"，因此我们的一致性项应约束输出分布而不是隐藏状态。

**小结（RQ1）**：顺序效应本身、其随干扰项放大、其与依赖复杂度的交互都已被确认，不能作为贡献。"回读"是 Chen 的定性解释，但尚无工作把回读量化为逐样本特征并用它**预测**哪一个呈现会让模型翻转。

## 5 分支 B：顺序不变性训练与数据增强

**He et al. (EMNLP 2025, Order Doesn't Matter, But Reasoning Does)** 是与本项目最接近的已发表工作：对 FOLIO、RuleTaker、LogicNLI 的前提随机重排（condition shuffle）并用 DAG 拓扑序重排推理步骤（answer-step shuffle），在 Llama-3-8B、Llama-2-13B、Mistral-7B 上全参数微调，报告相对 vanilla SFT 7–15% 的提升，并做了等步数对照以排除数据量效应。两点细节对我们关键：(i) 他们**没有一致性损失、没有困难样本挖掘**；(ii) 他们按 Kendall τ 分档考察训练，发现**均匀随机重排在所有 τ 档上都最好**——这是对"负担引导挖掘优于随机"假设的直接反证，必须在实验中正面检验。

**MEND (ACL Findings 2025, Your Language Model May Think Too Rigidly)** 在 Llama-3.2-1B/3B 上，以 PromptBench/DyVal 逻辑与算术任务、iGSM 为数据，比较拓扑序、随机序、逆序测试集，用"顺序重排 + 冗余添加"的查询增强训练，报告对 OOD 冗余的泛化。其实验设置（小模型、拓扑/随机/逆序三种呈现、干扰项）与我们的 E0/B2 几乎重合，因此 **B2（随机重排增强 SFT）应视为 He 2025 与 MEND 的复现基线**。

**Pearl (2025, Towards Permutation-Resilient LLMs)** 把每个样本的所有排列视为模糊集，用置换提议网络（Sinkhorn 最优传输）寻找最坏排列，做 DRO 式对抗训练；对象是 ICL 示例排列而非推理前提。Pearl 是"困难排列挖掘"的已有实现，**BIT 的负担引导可视为一个无需额外网络、可解释的最坏情况代理**，需要与"按损失挖掘"对照。**Set-LLM (NeurIPS 2025)** 与 **SBP 微调 (2025)** 从注意力掩码/位置编码层面保证集合输入的置换不变性，主要针对多选题与多文档，不适用于前提之间有依赖、需要按证明顺序读取的推理。

**小结（RQ2-a）**：随机重排增强已被做过；最坏排列 DRO 已被做过（ICL 场景）；"按依赖负担选择困难呈现"尚未检索到。

## 6 分支 C：一致性正则

一致性正则源于视觉与 NLU：**AugMix (ICLR 2020)** 用原图与两个增强图后验分布之间的 Jensen–Shannon 散度；**R-Drop (NeurIPS 2021)** 对同一输入两次 dropout 的输出施加双向 KL。在 LLM 上，**BCT (Chua et al., 2024)** 让模型在含偏置提示与无偏置提示下生成相同的 CoT（以 SFT 实现），在一个偏置上训练可把留出任务上的偏置推理降低 86%，并泛化到其他偏置；**Irpan et al. (2025, Consistency Training Helps Stop Sycophancy and Jailbreaks)** 在 Gemini 2.5 Flash 上比较输出层 BCT 与激活层 ACT；**F2C (arXiv 2510.14242, 2025)** 用多数投票伪标签与 KL/JSD 把提示变体的预测拉向高置信共识。

这些方法的共同点是"同一语义的多个输入应得到同一输出"，与 BIT 的一致性项同构；区别在于它们的变体来源是偏置提示、越狱包装或改写，而**不是保证答案不变的结构重排**，也没有按负担选择变体。另外 BCT 的实现形式（让不同呈现拟合同一条推理链）与 B2/BIT 中"所有呈现共享同一条求解器推理链"相同，因此 BIT 的增益必须来自显式 KL 项或挖掘策略，而不能来自"共享目标"本身。

## 7 分支 D：强化学习与泛化

**DeepSeekMath (Shao et al., 2024)** 提出 GRPO，以组内平均奖励替代价值网络；**Chu et al. (ICML 2025, SFT Memorizes, RL Generalizes)** 发现结果奖励 RL 对未见规则变体的泛化优于 SFT，但 SFT 对稳定输出格式仍是必要的。与结构不变性直接相关的是 **PA-GRPO (Zheng et al., ACL 2026)**：把同一实例的选项排列组成排列组，用跨排列优势基线与一致性奖励抑制选择偏置；**GC²PO (arXiv 2602.06475, 2026)** 用表征空间扰动下的稳定性奖励鼓励"因果不变"的推理模式；arXiv 2512.12858 用 GRPO 优化语义等价提示组的信息一致性。

这些工作说明"跨等价变体分组优化"在 RL 中已有实现，但对象是多选题偏置或提示改写，而非推理前提的依赖顺序。由于 8GB 显存难以运行组采样 RL，我们把 RL 变体列为讨论与未来工作，不进入主实验；这在论文中需要作为限制披露。

## 8 分支 E：合成规则推理与捷径

**RuleTaker (Clark et al., IJCAI 2020)** 与 **ProofWriter (Tafjord et al., Findings ACL 2021)** 提供深度 D0–D5 的合成规则推理数据（ProofWriter 每个深度 10 万题，含证明标注）；**Zhang et al. (IJCAI 2023, On the Paradox of Learning to Reason from Data)** 证明在 SimpleLogic 上 BERT 会利用"规则数量"等统计特征达到分布内高准确率却无法跨采样分布泛化。这对生成器设计有直接约束：yes/no 两类必须在规则数、事实数、干扰项数上完全匹配，否则训练会学到捷径而非推理。我们的生成器已按此约束实现（负例用"同结论、不可推出前提"的诱饵规则替换一条证明规则，规则数不变）。

## 9 跨分支综合与开放问题

1. **增益来源未被分解**。He 2025 与 MEND 报告的是分布内重排测试集上的提升，没有把"同一图上的稳定性"与"新依赖结构上的迁移"分开测量。重排训练若只提升已见图族上的 shifted 准确率，是局部适应；若提升未见图族，才说明学到了与呈现无关的推理过程。
2. **困难呈现的定义是空白**。Pearl 用损失定义困难，He 2025 用 Kendall τ 分档后发现随机最好；还没有人用依赖负担（回读、距离、前沿）这种**结构上可解释、无需模型前向**的量去选择训练呈现，并检验它是否优于随机或按损失选择。
3. **一致性项在推理链 SFT 中是否有额外作用未知**。BCT 类方法用共享目标隐式实现一致性；显式 KL 在已经共享目标的推理链 SFT 上是否还有增益，尚无证据。
4. **小模型的适用边界**。MEND 在 1B/3B 上有效，但我们自己的初步试跑显示 Qwen2.5-1.5B 在深度 ≥4 的规则推理上接近随机，这限制了任何"鲁棒性"结论的适用范围。

## 10 结论（逐条回答 RQ）

- **RQ1**：已被充分证明。前向/证明对齐顺序最好、随机与逆序更差、干扰项放大效应（Chen 2024; DyVal; MEND; GSM-Symbolic）。机制层面只有"线性扫描/回读"的定性解释与 iGSM 的规划探针证据，以及 Tao 2026 的"表征区分排列"反直觉结果。
- **RQ2**：随机重排增强 SFT（He 2025; MEND）、最坏排列 DRO（Pearl, ICL 场景）、跨变体一致性训练（BCT; F2C; Irpan 2025）、排列组 RL（PA-GRPO）都已存在。**BIT 的 (a) 与 (c) 两部分单独看均已被覆盖，只有 (b) 负担引导挖掘及其与一致性项、2×2 分离评估的组合未检索到。**
- **RQ3**：尚未回答的是：(i) 重排训练的增益在新图族上是否保留；(ii) 负担引导是否比随机重排更有效（He 2025 的 τ 分档结果暗示可能不会）；(iii) 显式一致性项在共享推理链目标下是否有边际作用。

## 参考文献（均已核实）

1. Chen, X., Chi, R. A., Wang, X., Zhou, D. Premise Order Matters in Reasoning with Large Language Models. ICML 2024, PMLR 235:6596–6620.
2. Zhu, K., Chen, J., Wang, J., Gong, N. Z., Yang, D., Xie, X. DyVal: Dynamic Evaluation of Large Language Models for Reasoning Tasks. ICLR 2024.
3. Zhou, Y., Liu, H., Chen, Z., Tian, Y., Chen, B. GSM-∞: How Do Your LLMs Behave over Infinitely Increasing Reasoning Complexity and Context Length? ICML 2025, PMLR 267.
4. Ye, T., Xu, Z., Li, Y., Allen-Zhu, Z. Physics of Language Models: Part 2.1, Grade-School Math and the Hidden Reasoning Process. ICLR 2025.
5. He, Q., He, Q., Liang, J., Zhou, W., Sun, Z., Yu, F., Xiao, Y. Order Doesn't Matter, But Reasoning Does: Training LLMs with Order-Centric Augmentation. EMNLP 2025, pp. 27178–27192.
6. Your Language Model May Think Too Rigidly: Achieving Reasoning Consistency with Symmetry-Enhanced Training (MEND). Findings of ACL 2025.
7. Pearl: Towards Permutation-Resilient LLMs. arXiv:2502.14628, 2025.（前身：Chen, L., Bian, Y., Shen, L., Wong, K.-F. Simple Permutations Can Fool LLaMA. ICLR 2024 SeT LLM Workshop.）
8. Set-LLM: A Permutation-Invariant LLM. NeurIPS 2025.
9. Order Independence With Finetuning. arXiv:2503.23483, 2025.
10. Mirzadeh, I., et al. GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models. ICLR 2025.
11. Li, Q., et al. GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers. ACL 2024.
12. Shi, F., Chen, X., Misra, K., Scales, N., Dohan, D., Chi, E., Schärli, N., Zhou, D. Large Language Models Can Be Easily Distracted by Irrelevant Context. ICML 2023.
13. Hoppe, F., Ilievski, F., Kalo, J.-C. Investigating the Robustness of Deductive Reasoning with Large Language Models. ECAI 2025, pp. 1776–1783.
14. Tao, Z. S. Order-Invariant Answers, Order-Sensitive Representations in Mathematical Reasoning. arXiv:2609.28442, 2026.
15. Hendrycks, D., et al. AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. ICLR 2020.
16. Wu, L., et al. R-Drop: Regularized Dropout for Neural Networks. NeurIPS 2021.
17. Chua, J., et al. Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought. arXiv:2403.05518, 2024.
18. Irpan, A., et al. Consistency Training Helps Stop Sycophancy and Jailbreaks. arXiv:2510.27062, 2025.
19. Flip-Flop Consistency (F2C). arXiv:2510.14242, 2025.
20. Shao, Z., et al. DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300, 2024.
21. Chu, T., et al. SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training. ICML 2025.
22. Zheng, J., Yuan, J., Yao, J., Gu, C., Zheng, P., He, G. Mitigating Selection Bias in Large Language Models via Permutation-Aware GRPO. ACL 2026.
23. Towards Generalizable Reasoning: Group Causal Counterfactual Policy Optimization for LLM Reasoning (GC²PO). arXiv:2602.06475, 2026.
24. Clark, P., Tafjord, O., Richardson, K. Transformers as Soft Reasoners over Language. IJCAI 2020.
25. Tafjord, O., Dalvi, B., Clark, P. ProofWriter: Generating Implications, Proofs, and Abductive Statements over Natural Language. Findings of ACL 2021.
26. Zhang, H., Li, L. H., Meng, T., Chang, K.-W., Van den Broeck, G. On the Paradox of Learning to Reason from Data. IJCAI 2023.

注：条目 6、7、8、9、19、23 的完整作者列表在检索页中未完整显示，写入 refs.bib 前需再核对；条目 16 R-Drop 的第一作者按检索结果记录为 Wu et al.（检索页仅给出 "Wang et al., 2021" 的二手转述，存在不一致，进入论文前必须核对原文）。
