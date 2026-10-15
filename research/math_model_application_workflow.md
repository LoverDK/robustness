# 数学模型在后续实验流程中的用法

这份说明回答一个实际问题：定义了 \(SS\) 和 \(\Delta_T\) 以后，后续实验到底怎样使用它们，而不是只把公式放在第三节。

## 1. 数据生成阶段：让公式对应到可验证字段

每条样本先保存潜在任务和呈现：

\[
z_i=(G_i,q_i,y_i),\qquad x_i^p=R(z_i,r_i^p),
\]

其中 \(p\in\{\mathrm{can},\mathrm{shift}\}\)。数据表至少包含：

- graph_id、graph_family、base_query_id；
- presentation_type；
- seen_unseen；
- answer 与 model_prediction；
- token_length、目标位置、模板；
- relevant_distance、active_frontier、operator_coordination。

这样后面每个指标都能从原始预测重新计算，不依赖手工抄分数。

## 2. 第一层：估计结构稳定性

对每个同图样本对 \((x_i^{can},x_i^{shift})\)，保存两个指示变量：

\[
c_i=\mathbf 1\{f(x_i^{can})=y_i\},\qquad
s_i=\mathbf 1\{f(x_i^{shift})=y_i\}.
\]

条件稳定率和条件翻转率直接计算为

\[
\widehat{SS}=\frac{\sum_i c_i s_i}{\sum_i c_i},\qquad
\widehat{SF}=1-\widehat{SS}.
\]

这一步的用途有三个：

1. 先判断模型是否有足够 clean competence；
2. 测量同一图的呈现变化是否导致已有能力丢失；
3. 为后面比较 Render-Aug 的训练收益提供基线。

bootstrap 时以 base_query_id 为单位，不能把同一任务的多个改写当成独立任务。

## 3. 第二层：估计结构迁移

对 canonical 呈现下的已见图和未见图，分别计算：

\[
\widehat R_S=\frac{1}{n_S}\sum_{i\in S}
\mathbf 1\{f(x_i^{can})\neq y_i\},
\]

\[
\widehat R_U=\frac{1}{n_U}\sum_{i\in U}
\mathbf 1\{f(x_i^{can})\neq y_i\}.
\]

迁移差距为

\[
\widehat\Delta_T=\widehat R_U-\widehat R_S.
\]

这一步的用途是测量换到未见图族后新增了多少错误。bootstrap 时以 graph_family 为单位，并报告每个图族的结果，避免大图族支配总体均值。

## 4. 二乘二模型：把两个效应放进一次误差分析

将每条预测写成：

\[
\operatorname{logit}\Pr(e_i=1)
=\alpha_m+\eta_S I_{\mathrm{shift},i}
+\eta_T I_{\mathrm{unseen},i}
+\eta_{ST}I_{\mathrm{shift},i}I_{\mathrm{unseen},i}
+\beta^\top X_i.
\]

其中 \(X_i\) 包含长度、图规模、深度、分支、目标位置和模板等控制项。

解释方式：

- \(\eta_S>0\)：在相同图熟悉度下，shifted 呈现增加错误；
- \(\eta_T>0\)：在 canonical 呈现下，未见图增加错误；
- \(\eta_{ST}>0\)：未见图会额外放大呈现变化的损失。

该模型的用途不是替代 \(SS\) 或 \(\Delta_T\)，而是检验二者是否对应不同的误差来源，并控制长度和图规模。

## 5. 第三层：建立可被样本外推翻的错误规律

在 discovery split 上用求解器 trace 计算：

- \(D_{\mathrm{rel}}\)：必需前提之间的距离；
- \(F_{\mathrm{active}}\)：同时保持活动的未完成依赖数量；
- \(K_{\mathrm{op}}\)：需要协调的算子数量。

主确认模型保留三个标准化特征：

\[
\operatorname{logit}\Pr(e_{m,x}=1)
=\alpha_m+\beta_D\widetilde D_{\mathrm{rel}}
+\beta_F\widetilde F_{\mathrm{active}}
+\beta_K\widetilde K_{\mathrm{op}}
+\gamma^\top Z.
\]

只有在 discovery split 中冻结权重后，才构造标准化的依赖负担摘要：

\[
B(x)=w_D\widetilde D_{\mathrm{rel}}(x)
+w_F\widetilde F_{\mathrm{active}}(x)
+w_K\widetilde K_{\mathrm{op}}(x),
\qquad w_D,w_F,w_K\geq 0.
\]

再拟合：

\[
\operatorname{logit}\Pr(e_{m,x}=1)
=\alpha_m+\beta_1D_{\mathrm{rel}}
+\beta_2F_{\mathrm{active}}
+\beta_3K_{\mathrm{op}}
+\gamma^\top Z.
\]

然后冻结特征算法、模型规格和系数方向，在 confirmatory split 上只做一次检验。

比较结果至少包括：

1. 长度和位置模型；
2. 图规模、深度和分支模型；
3. 依赖负担模型；
4. 完整模型。

主判据是 held-out log loss、Brier score、校准和预注册负担区间错误率。这样数学模型真正承担“预测未见样本错误”的任务，而不是只解释已经看过的结果。

规律被视为成立，需要同时满足：系数方向保持、预测优于最强长度/图规模对照、负担分箱错误率单调，以及正确依赖提示优于匹配的 placebo 提示。任意一项失败，都要缩小结论。

## 6. 第四层：用正交干预检验两个量能否分离

对四个训练条件 \(c\) 计算：

\[
G_S(c)=SS_c-SS_{\mathrm{Base}},
\qquad
G_T(c)=\Delta_{T,\mathrm{Base}}-\Delta_{T,c}.
\]

判定逻辑：

- Render-Aug：\(G_S\) 超过 \(\delta_S\)，同时 \(G_T\) 落在 \([-\delta_T,\delta_T]\)；
- Graph-Aug：\(G_T\) 超过 \(\delta_T\)，同时 \(G_S\) 落在 \([-\delta_S,\delta_S]\)；
- Joint：两个收益都超过界限。

这里的 \(\delta_S,\delta_T\) 必须在看结果前根据测量噪声、pilot 方差或最小科学效应预注册。使用等价检验或置信区间，不能把“一个显著、另一个不显著”当成分离证据。

## 7. 适用边界如何计算

把每个测试条件写成

\[
\tau=(\mathcal G,\mathcal R,\mathcal Q,m,d),
\]

分别表示图分布、呈现分布、任务族、模型和解码条件。冻结 discovery 得到的特征标准化、模型规格和系数方向，在每个新条件上计算：

\[
V(\tau)=
\mathbf 1\{\text{方向一致}\}
\mathbf 1\{\text{预测优于最强对照}\}
\mathbf 1\{\text{校准通过}\}.
\]

\[
V(\tau)=
\bigl(
V_{\mathrm{direction}}(\tau),
V_{\mathrm{prediction}}(\tau),
V_{\mathrm{calibration}}(\tau)
\bigr).
\]

只有三项都为 \(1\) 才把该条件纳入完整适用域；改变一次一个主要因素并记录三项结果，就能得到边界表。

## 8. 第六层：LoRA 结果如何解释

LoRA 不直接证明内部机制，只提供受控干预结果。对每个 LoRA 条件同时报告：

- clean accuracy；
- \(SS\) 和 \(SF\)；
- \(\Delta_T\)；
- 第二任务族迁移；
- 错误规律在干预前后的预测能力。

解释规则：

- 只提高 \(SS\)，不改变 \(\Delta_T\)：呈现适应；
- 只降低 \(\Delta_T\)：结构组合迁移改善；
- 两者都改善，且未见图族也保持：支持更广泛的结构改进；
- 只在训练中见过的变体上改善：只能称为局部适应。

## 9. 从公式到论文结论的最短路径

论文结果表按以下顺序组织：

1. 报告 \(\widehat{SS}\)、\(\widehat{SF}\) 和 clean competence；
2. 报告 \(\widehat\Delta_T\)、各图族风险和二乘二交互；
3. 报告错误规律在 confirmatory split 上的样本外预测；
4. 报告正交干预的 \(G_S\) 和 \(G_T\)；
5. 报告第二任务族和 LoRA 的边界。

这样每个公式都有对应实验用途：\(SS\) 测同图保持，\(\Delta_T\) 测新图风险增量，回归模型预测错误，\(G_S/G_T\) 检验两者能否被不同训练因素分别改变。
