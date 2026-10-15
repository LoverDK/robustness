# MVP 实验协议：结构稳定性、结构迁移、错误规律和边界

## 实验目的

第一轮真实模型实验按下面顺序推进：

1. 在同一依赖图的等价变体上测结构稳定性；
2. 在完全独立的未见图池上测结构迁移；
3. 在发现集上选出错误规律，在确认集上做一次样本外检验；
4. 在第二任务族和匹配训练干预中检查规律的边界。

StructShift 只负责生成和验证数据。LoRA 要等前三步稳定以后再做。toy
pilot 只能检查代码和指标，不能当作模型证据。

## 一、先把三个数据池分开

### 1. 稳定性池

一对稳定性样本必须共享：

- 同一个 world；
- 同一张依赖图；
- 同一个目标问题；
- 同一个 solver proof 和答案；
- 相同的深度、分支数和算子组合。

允许的变化只有：

- 拓扑合法的前提或规则顺序；
- 语义等价的文本模板；
- 相关前提之间的文本距离，在总 token 尽量匹配时改变。

增加深度、增加分支、改变算子组合不能放进稳定性定义，因为这已经可能改变解题复杂度。

数学上，先固定潜在任务 \(z=(G,q,y)\)，再生成 canonical 和 shifted 两个呈现：

\[
x^{can}=R(z,r_0),\qquad x^{shift}=R(z,r_1).
\]

主指标是条件稳定率

\[
\widehat{SS}=
\frac{\sum_i \mathbf 1\{f(x_i^{can})=y_i,\ f(x_i^{shift})=y_i\}}
{\sum_i \mathbf 1\{f(x_i^{can})=y_i\}},
\]

条件翻转率为 \(1-\widehat{SS}\)。另外报告不条件化的一致率，防止模型只是稳定地犯错。

### 2. 迁移池

迁移样本必须来自稳定性池和发现数据中没有出现过的图族或算子组合。迁移池不能共享 world、base query、图哈希、图族和改写 lineage。

数学上固定 canonical 呈现规则，改变图分布：

\[
R_S=\mathbb E_{G\sim P_S}[\mathbf 1\{f(x_G^{can})\neq y_G\}],\qquad
R_U=\mathbb E_{G\sim P_U}[\mathbf 1\{f(x_G^{can})\neq y_G\}],
\]

\[
\Delta_T=R_U-R_S.
\]

因此迁移不是“同一题换写法”，而是“新图支持上的风险增量”。\(SS\) 和 \(\Delta_T\) 必须在不同数据池上估计。

### 3. 干扰池

disconnected decoy、长度控制句和 side-question 单独分析。它们可以帮助区分解释，但不能悄悄并入稳定性分数或迁移分数。

## 二、数据怎么切

在自然语言渲染前，先切符号对象：

1. 开发 smoke set：只检查生成、求解和指标；
2. discovery set：选择错误特征和模型规格；
3. confirmatory stability set：新的 world 和 base query；
4. confirmatory transfer set：新的图族或算子组合；
5. independent task set：第二个任务族、新模板和新词汇。

任何 base query、图哈希、图族或 paraphrase lineage 都不能跨保护边界。确认集打开前，要冻结模型版本、特征算法、排除规则、系数方向和比较模型。

## 三、Stage 0：生成器和元数据检查

第一版先做 ProofWriter 风格规则推理。每条记录保存：

- world ID、base-query ID、图哈希、图族 ID、渲染 lineage；
- proof trace、答案、深度、分支、图规模和算子序列；
- 证明中相关前提的位置和相互距离；
- 从 solver trace 计算出的 active frontier；
- token 长度、目标位置、模板、标签和变换序列。

每个候选都重新求解。答案改变、稳定性池中的依赖图改变、引用错误、问题有歧义、数据泄漏和重复样本全部拒绝。模型实验前做分层人工抽查，并记录拒绝原因和人工一致率。

## 四、Stage 1：结构稳定性

用一个冻结的开源指令模型和确定性解码。对每个同图样本对报告：

- 原题准确率；
- 变体准确率；
- 条件翻转率：原题答对、变体答错；
- 条件稳定率：原题答对、变体仍答对；
- 恢复率：原题答错、变体答对；
- 按 base query 配对的 bootstrap 区间。

主统计量是：

    Pr[变体预测 != y | 原题预测 = y]

在分析中控制 token 长度、目标位置、模板、相关前提距离和标签。如果控制后条件翻转消失，就不能保留“该变换造成独立稳定性损失”的强结论。

## 五、Stage 2：结构迁移

在保护的迁移池上评估未见图族和未见算子组合。报告：

- 规范呈现下的迁移准确率；
- 最差图族准确率；
- 按图族 bootstrap 或置信区间；
- 单独测得的 clean competence。

稳定性和迁移的关系只能跨多个模型、checkpoint 或受控训练条件估计：

    transfer_score ~ stability_score + clean_competence
                     + parameter_count + model_or_training_condition

两个分数不能来自同一批 base query 或同一批图族。相关不稳定或不显著，本身就是结果，不能事后重定义迁移分数。

## 六、Stage 2.5：专门区分两者的正交干预实验

只靠定义和二乘二评估还不够。要再做一个“双重分离”实验，检验两个指标是否能被不同干预分别改变。

四个训练条件：

1. Base：不做结构增强；
2. Render-Aug：只加入已见图的合法重排和等价模板，不能加入新的图族；
3. Graph-Aug：只加入 canonical 呈现下的新结构组合，呈现规则固定；
4. Joint：同时加入两种增强。

四个条件都匹配基座、训练 token、训练步数、LoRA rank、目标模块、随机种子和显存预算。评估固定使用四格：

- seen/canonical；
- seen/shifted；
- unseen/canonical；
- unseen/shifted。

相对 Base 定义：

\[
G_S(c)=SS_c-SS_{\mathrm{Base}},
\qquad
G_T(c)=\Delta_{T,\mathrm{Base}}-\Delta_{T,c}.
\]

预注册等价界限 \(\delta_S,\delta_T\)：

- Render-Aug：要求 \(G_S>\delta_S\)，且 \(|G_T|<\delta_T\)；
- Graph-Aug：要求 \(G_T>\delta_T\)，且 \(|G_S|<\delta_S\)；
- Joint：允许两个收益都超过对应界限。

用图族或 base query 为 bootstrap 单位。不能用“一个显著、另一个不显著”证明双重分离，必须使用等价检验或置信区间判断交叉效应是否足够小。

如果 Render-Aug 和 Graph-Aug 的效果完全相同，就不能宣称稳定性和迁移在该任务上已经行为解耦；这会成为需要报告的负结果。

## 七、Stage 3：发现和确认错误规律

### 候选特征

- relevant_distance：证明必需前提之间的距离；
- active_frontier：同时保持活动状态的未完成依赖数量；
- operator_coordination：最终答案需要联合的关系数量；
- 控制项：总 token、图规模、目标位置、模板、标签、深度和分支数。

### 先冻结一个具体规律

主确认模型保留三个标准化特征：

\[
\operatorname{logit}\Pr(e_{m,x}=1)
=\alpha_m+\beta_D\widetilde D_{\mathrm{rel}}
+\beta_F\widetilde F_{\mathrm{active}}
+\beta_K\widetilde K_{\mathrm{op}}
+\gamma^\top Z.
\]

只有在 discovery split 中冻结权重后，才构造可视化摘要：

\[
B(x)=w_D\widetilde D_{\mathrm{rel}}(x)
+w_F\widetilde F_{\mathrm{active}}(x)
+w_K\widetilde K_{\mathrm{op}}(x),
\qquad w_D,w_F,w_K\geq 0.
\]

候选规律写成：

\[
\Pr(e_{m,x}=1\mid B(x),Z(x))
=\sigma\left(\alpha_m+\phi(B(x))+\gamma^\top Z(x)\right),
\]

其中 \(\phi\) 先用线性函数，再用预先固定的单调样条做形状检查。这里的科学主张是：在长度、图规模、模板等条件匹配后，依赖特征向量或冻结后的 \(B\) 更高的题应该有更高的错误风险。不能在确认集重新学习权重。

只用 discovery set 选择特征、交互项和随机效应。确认集打开前冻结模型规格。

至少比较四类模型：

1. 只用 token 长度和目标位置；
2. 只用图规模、深度和分支；
3. 只用依赖负担特征；
4. 完整的依赖特征加干扰变量模型。

在确认集报告 held-out log loss 或 Brier score、校准、排序相关和预先划定的负担区间错误率。bootstrap 的单位用 base query 或图族，不能把同一图的多个渲染当成独立样本。

规律只有同时满足以下条件才写成确认性结果：

1. 确认集上系数方向与 discovery 预期一致；
2. 依赖负担模型优于最强的长度/图规模对照；
3. 负担分箱错误率单调增加；
4. 正确依赖提示优于格式和 token 数匹配的 placebo 提示。

方向反转、预测优势消失、分箱不单调或 placebo 同样有效，都只能报告为探索性趋势或当前规律失效。

### 区分解释的附加测试

1. 询问 solver-derived 的局部子问题。如果局部步骤和完整证明一样容易错，不能把错误主要归因于依赖整合。
2. 加入正确的依赖标记，再加入 token 数量和格式都匹配的 placebo 标记。两者都有效，说明可能只是提示格式或额外 token 的作用。
3. 总长度匹配，只改变相关前提距离。
4. 图规模和深度固定，只改变 active frontier。

规律只有在方向和预测优势都能通过这些比较时，才可以写成确认性结果；否则只能写成探索性分析。

## 八、Stage 4：边界实验

### 1. 第二个任务族

把已经冻结的特征提取、标准化参数、预测模型和系数方向搬到第二个可求解推理任务族，换模板、词汇和渲染规则。方向和样本外预测都保持，说明有跨任务支持；只在第一任务成立，就写成任务边界。

每个边界条件写成

\[
\tau=(\mathcal G,\mathcal R,\mathcal Q,m,d),
\]

分别表示图分布、呈现分布、任务族、模型和解码条件。定义：

\[
V(\tau)=
\bigl(
V_{\mathrm{direction}}(\tau),
V_{\mathrm{prediction}}(\tau),
V_{\mathrm{calibration}}(\tau)
\bigr).
\]

实验中一次只改变一个主要因素，并分别记录三项 \(V(\tau)\)。只有三项都通过时，才称为完整适用；这样可以区分方向、预测优势和校准分别在哪一步失效。

### 2. 匹配训练干预

前三阶段稳定后，再比较：

1. 冻结基座；
2. clean-only LoRA；
3. 随机合法结构增强 LoRA；
4. 可选的错误引导结构 LoRA。

匹配 base model、LoRA rank、目标模块、训练 token、步数、标签比例、随机种子和计算量。评估 clean competence、同图稳定性、见过的变体、未见图迁移、第二任务迁移，以及规律预测错误和实际错误的变化。

只有保护迁移集也提升，才能说出现了可迁移的鲁棒性变化。只在见过的变体上提升，应报告为局部适应。LoRA 结果本身也不能直接证明某个隐藏神经机制。

## 九、停止和解释规则

- 答案或依赖图不能稳定保持，先修生成器；
- 稳定性在控制后消失，缩小稳定性结论；
- 稳定性不能超出 clean competence 预测独立迁移，放弃“稳定性是通用迁移代理”的说法；
- 长度模型和依赖模型预测一样好，放弃当前错误规律；
- 规律在确认集失败，只能作为探索性趋势；
- 规律只在第二任务失败，报告任务边界；
- LoRA 只提升见过的变体，报告局部适应。

## 十、资源和执行顺序

符号生成、求解和元数据计算可以先用 CPU。第一轮先跑小规模真实模型，再决定正式样本量。3B 模型零样本评估需要模型缓存和足够批处理内存；QLoRA 在扩大规模前先用 16--24 GB GPU 做 profiling。保存所有符号对象、预测、配置、模型版本、随机种子、训练日志、排除记录和资源测量。

严格顺序是：生成器验证 -> 稳定性 pilot -> 独立迁移 pilot -> 正交干预区分 -> 错误规律发现/确认 -> 第二任务边界 -> 可选 LoRA。
