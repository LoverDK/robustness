# C1 诊断：同图稳定，还是换图也能跟上

这份稿给项目负责人看。论文现在以 C1 为主。C2（按负担挑选是否优于随机打乱）在两轮试跑里都写成负结果。下面的准确率和区间都来自本稿打开的 metrics 文件。E0 的区间在文件里是三个数一行，没有 `lo` / `hi` 这两个字段名，稿中按「点估计、下端、上端」来读。E1 的区间用文件里的 `lo` 和 `hi`。

## 1. 这一问在问什么

C1 要分开两件事。同图稳定问的是：依赖图还是训练里见过的那一类，只把规则的呈现换掉，准确率还在不在。换图迁移问的是：依赖结构换成训练里没出现过的图族之后，准确率还在不在。两件事可以一个成立、另一个不成立。同一张图的逆序变稳，只说明这种写法难不倒它；没见过的图族上变差，说明它还没有把这件事带到新的依赖结构上。一个只报「打乱之后的准确率」的数字，会把这两种变化加在一起。

测量上，这件事落在四格里：见过的图族（seen）或没见过的图族（unseen），再乘规范呈现（aligned）或偏移呈现（shifted）。文件里 shifted 的 `presentation` 是 `reversed`，也就是核心规则倒序。同图稳定看 seen 上 aligned 和 shifted 是否一起变。换图迁移看 unseen 上的两格，尤其是 unseen/shifted：图族和呈现都离开了训练时最熟的那一格。

现有的顺序训练在这轮里有三档：只练规范顺序（B1）、随机打乱（B2）、随机打乱再加一致性项（B3）。C1 比较的是这三档把哪一格抬高。按负担挑选（ours）是另一问，C2：这样挑呈现，是否比随机打乱更好。两轮试跑都没有得到一份和随机打乱分开的优势，本稿按负结果写，不把 ours 写成方法上的胜利。

## 2. 题和测量

先把一份更早的试点单独放在这里，后面的格子不和它比。`experiments/results/mvp_qwen25_1.5b/metrics.json` 里四个格子都是 0.76（`cell_accuracy.seen_canonical`、`cell_accuracy.seen_shifted`、`cell_accuracy.unseen_canonical`、`cell_accuracy.unseen_shifted`）。标签是 yes 38、no 12（`label_distribution_per_cell`）。同文件的 `SS` 是 1.0，`SF` 是 0.0，`Delta_T` 是 0.0。`interpretation` 写的是：这只是探索性的流程检查，多数类行为使它不能支持结构鲁棒性的说法。格子名字还是 `canonical`，后面的文件用的是 `aligned`。0.76 只说明当时的输出贴着多数类，不能拿来当顺序训练的证据。

E0 没有训练。模型是 Qwen/Qwen2.5-1.5B-Instruct（`build/e0/round1/metrics.json`，`config.model`），深度是 2、4、6、8（`config.depths`），每一档 200 个潜在任务（`config.n_per_level`，以及 `levels.2.n_latent` 等），种子 0（`config.seed`），图族只有 chain 和 conj_fact（`config.families`）。门禁写在 `gate.criteria`：规范顺序准确率落在 0.55 到 0.9，区间下端高于 0.5，并且从规范顺序到随机顺序的 SF 至少 0.1、区间下端高于 0。`gate.any_level_pass` 是 true，因为只有深度 2 的 `levels.2.gate_pass` 是 true。

深度 2 的规范顺序准确率是 0.65，区间 0.58 到 0.715（`levels.2.acc.aligned`）。随机顺序是 0.735，区间 0.675 到 0.795（`levels.2.acc.random`）。逆序是 0.69，区间 0.625 到 0.755（`levels.2.acc.reversed`）。到随机顺序的 SF 是 0.2692307692307692，区间 0.1937921980495124 到 0.34558823529411764（`levels.2.SF.random`）。三个门禁条件这一档都满足，所以后面的训练留在深度 2。

深度 4、6、8 的 `gate_pass` 都是 false。深度 4 的规范顺序是 0.56，区间 0.49 到 0.63（`levels.4.acc.aligned`）：点估计还在 0.55 到 0.9 里，区间下端 0.49 没有高于 0.5。深度 6 是 0.51，区间 0.44 到 0.58（`levels.6.acc.aligned`）。深度 8 是 0.495，区间 0.425 到 0.565（`levels.8.acc.aligned`）。后两档的点估计已经低于 0.55，区间下端也低于 0.5。深度 4 到随机顺序的 SF 仍然是 0.4017857142857143，区间 0.31302989130434783 到 0.4952403204272363（`levels.4.SF.random`），所以这三档被留下的原因是规范顺序已经靠近一半对一半，翻转率不再好解释，不是因为文件里没有顺序差异。

深度 2 上，随机顺序和逆序的点估计都高于规范顺序。规范顺序减去随机顺序的配对差是 -0.085，区间 -0.175 到 0.005（`levels.2.diff_vs_aligned.random`）。减去逆序是 -0.04，区间 -0.14 到 0.05（`levels.2.diff_vs_aligned.reversed`）。两个区间的上端都大于 0，点估计的方向没有和 0 分开。项目文献笔记里已有「按证明顺序最好」这一说法，本稿没有再次核实引用。这一次的点估计方向和那一说法相反，配对区间还跨着 0。带干扰的随机呈现是 0.46，区间 0.395 到 0.535（`levels.2.acc.random_distract`），规范顺序减去它是 0.19，区间 0.09 到 0.29（`levels.2.diff_vs_aligned.random_distract`），这一格的区间在 0 的一侧。

顺序标签没有写反。E0 这份 metrics 里没有回读次数字段。后面打开的每一份 E1 metrics 里，`presentation` 为 aligned 的 `back_reads_mean` 都是 0.0，seen 上 reversed 的 `back_reads_mean` 都是 1.0，unseen 上 reversed 的 `back_reads_mean` 都是 3.0。生成代码把 reversed 定义成核心规则整段倒序。深度 2 的 seen 图族每条任务有 2 条核心规则，倒序的逆序对上限是 1；unseen 图族每条任务有 3 条核心规则，上限是 3。评估里记到的均值等于这个上限，所以 reversed 是满额倒序。metrics 里没有 `back_reads_norm`，也没有一个叫「满额」的字段。

E0 这一轮有 SF，没有 SS，也没有同图稳定和换图迁移的交互项。E0 的图族只有 chain 和 conj_fact，四格里的 unseen 还没出现。

## 3. 不训练时的四格

不训练的四格在 `experiments/e1/pilot/b0/metrics.json`。条件是 b0（`condition`），种子 0（`seed`），深度 2（`depth`），模型仍是 Qwen/Qwen2.5-1.5B-Instruct（`model`）。评估量化写的是 bf16-eval（`quant`）。后面训练过的条件写的是 nf4。和 B0 比的时候，中间还夹着这个差别；同一轮里 B1、B2、B3、ours 彼此都是 nf4。每一格 200 题，yes 和 no 各 100（各格的 `n`、`n_yes`、`n_no`）。解码是 `do_sample` false、`max_new_tokens` 512（`decoding`）。

seen/aligned 的准确率是 0.645，区间 0.58 到 0.71（`cells.seen_aligned.acc`、`lo`、`hi`）。seen/shifted 是 0.685，区间 0.62 到 0.75（`cells.seen_shifted`）。unseen/aligned 是 0.64，区间 0.57 到 0.705（`cells.unseen_aligned`）。unseen/shifted 是 0.55，区间 0.48 到 0.62（`cells.unseen_shifted`）。见过的图上，逆序的点估计高于规范顺序，两格区间重叠。没见过的图上，unseen/shifted 的点估计最低，它的区间和 unseen/aligned 仍然重叠。不训练时，四格还不是「一换呈现就掉、一换图再掉」的整齐台阶。

抽取失败率在这四格是 0.01、0.01、0.04、0.005（对应格的 `extract_fail_rate`）。预测为 yes 的比例是 0.545、0.445、0.515、0.46（`pred_yes_rate`），没有贴在 0 或 1 上。GSM8K 是 0.688，区间 0.65 到 0.728，500 题（`gsm8k.acc`、`lo`、`hi`、`n`）。混入轮没有重跑 B0，后面普通数学题的对照就用这个 0.688。

这一份文件里没有 SS、SF 和交互项。这一轮没有这个数。

## 4. 第一轮试跑

第一轮是 `experiments/e1/pilot`，训练里没有混入 GSM8K。四份训练条件都是种子 0、深度 2、nf4、63 步、1000 条训练序列（各文件的 `seed`、`depth`、`quant`、`train.steps`、`train.consumed_sequences`）。B1 是只练规范顺序，B2 是随机打乱，B3 是同一套随机打乱再加一致性项，ours 是按负担挑选。每一格仍是 200 题、yes/no 各 100。这一轮四份训练文件和 B0 一样，都没有 SS、SF 和交互项。这一轮没有这个数。

只练规范顺序（`experiments/e1/pilot/b1/metrics.json`）把两个规范呈现格抬高了，两个 shifted 格没有从 B0 的区间里分开。seen/aligned 是 0.97，区间 0.945 到 0.99（`cells.seen_aligned`）。这个区间的下端高于 B0 的上端 0.71。unseen/aligned 是 0.88，区间 0.835 到 0.925（`cells.unseen_aligned`），下端高于 B0 的上端 0.705。seen/shifted 是 0.73，区间 0.665 到 0.79（`cells.seen_shifted`），和 B0 的 0.62 到 0.75 重叠。unseen/shifted 是 0.645，区间 0.58 到 0.71（`cells.unseen_shifted`），下端 0.58 仍低于 B0 的上端 0.62。seen/shifted 和 unseen/shifted 这次是一起停在重叠里，一起被抬高并和 B0 分开的是两个 aligned 格。这一档的 shifted 格预测为 yes 的比例是 0.27 和 0.225（`cells.seen_shifted.pred_yes_rate`、`cells.unseen_shifted.pred_yes_rate`），比 B0 更偏向 no。GSM8K 是 0.51，区间 0.468 到 0.554（`gsm8k`）。

随机打乱（`experiments/e1/pilot/b2/metrics.json`）四个格的区间都和 B0 分开。seen/aligned 是 0.985，区间 0.965 到 1.0（`cells.seen_aligned`）。seen/shifted 是 0.895，区间 0.855 到 0.935（`cells.seen_shifted`），下端高于 B0 的上端 0.75。unseen/aligned 是 0.79，区间 0.735 到 0.845（`cells.unseen_aligned`）。unseen/shifted 是 0.705，区间 0.645 到 0.765（`cells.unseen_shifted`），下端高于 B0 的上端 0.62。seen/shifted 和 unseen/shifted 一起离开了 B0，两格彼此却不重叠：同一张图的逆序在 0.855 以上，换图的逆序在 0.765 以下。和 B1 比，分开得清楚的是 seen/shifted：B1 的上端是 0.79，B2 的下端是 0.855。unseen/shifted 上，B1 的 0.58 到 0.71 和 B2 的 0.645 到 0.765 仍重叠。unseen/aligned 上，B1 的 0.835 到 0.925 和 B2 的 0.735 到 0.845 也重叠。规范顺序训练抬高的是规范呈现，随机打乱额外把同一张图的逆序抬到了另一段区间里。GSM8K 是 0.544，区间 0.496 到 0.586（`gsm8k`）。

加一致性（`experiments/e1/pilot/b3/metrics.json`）的 `lambda_comparison.lambda` 是 0.1。seen/aligned 是 0.975，区间 0.95 到 0.995。seen/shifted 是 0.9，区间 0.855 到 0.94，和 B0 分开，也和 B2 的 seen/shifted 重叠。unseen/aligned 是 0.81，区间 0.755 到 0.865，和 B0 分开。unseen/shifted 是 0.68，区间 0.615 到 0.745，下端 0.615 仍低于 B0 的上端 0.62，也和 B2 的 0.645 到 0.765 重叠。这一档里 seen/shifted 离开了 B0，unseen/shifted 没有。两格 shifted 没有一起动。验证集平均准确率 B2 是 0.9375，B3 是 0.9325，差是 0.0050000000000000044，阈值是 0.05（`lambda_comparison.b2_val_mean_acc`、`b3_val_mean_acc`、`b2_minus_b3`、`threshold_mean_acc`）。`lambda_0_5_ran` 是 false，一致性系数没有再加大。GSM8K 是 0.544，区间 0.5 到 0.588（`gsm8k`）。

按负担挑选（`experiments/e1/pilot/ours/metrics.json`）的挖掘记录里，选中负担均值是 2.2286666666666664 和 2.1401666666666666，候选负担均值两轮都是 1.6714583333333333（`train.mine_stats` 的 `selected_burden_mean`、`candidate_burden_mean`）。挑选确实偏向负担更高的呈现。seen/aligned 是 0.885，区间 0.84 到 0.93。seen/shifted 是 0.91，区间 0.87 到 0.95，和 B0 分开，和 B2 的 0.855 到 0.935 重叠。unseen/aligned 是 0.745，区间 0.685 到 0.805，和 B0 的 0.57 到 0.705 重叠。unseen/shifted 是 0.77，区间 0.715 到 0.8251249999999992，和 B0 分开。点估计 0.77 高于 B2 的 0.705，也高于 B3 的 0.68，但区间下端 0.715 低于 B2 的上端 0.765，也低于 B3 的上端 0.745。文件里没有 Ours 减 B2 的差值，也没有这个差值的区间。seen/shifted 和 unseen/shifted 都离开了 B0，两格彼此不重叠。这不能写成已经超过随机打乱。GSM8K 是 0.514，区间 0.468 到 0.56（`gsm8k`）。

这一轮四个训练条件的 GSM8K 点估计是 0.51、0.544、0.544、0.514，都比 B0 的 0.688 低超过 1 个百分点。四个区间的上端最高是 0.588，低于 B0 的下端 0.65。

## 5. 混入约 15% GSM8K 训练题之后

第二轮是 `experiments/e1/pilot_mix15`。B1、B2、B3、ours 都在，没有 B0。请求比例是 0.15（`gsm_mix.fraction_requested`），实际比例是 0.14965986394557823（`gsm_mix.gsm_fraction`）。合成序列 1000 条，GSM8K 训练序列 176 条、88 道题，合计 1176 条（`gsm_mix.n_synthetic_sequences`、`n_gsm_sequences`、`n_gsm_questions`、`n_total_sequences`）。与已评测的 500 道题文本重叠数是 0（`gsm_mix.selected_text_overlap_with_held_out`，`held_out_test_n`）。四份都是种子 0、深度 2、nf4、74 步。`experiments/e1` 下面除此以外没有另一份 metrics。没有更大混入比例的结果文件，也没有全量训练的结果文件。一致性系数同样没有加大：B3 的 `lambda_comparison.lambda_0_5_ran` 仍是 false。这一轮也没有 SS、SF 和交互项。这一轮没有这个数。

同一张图上的分割还在。只练规范顺序（`experiments/e1/pilot_mix15/b1/metrics.json`）的 seen/aligned 是 0.965，区间 0.935 到 0.99；seen/shifted 是 0.775，区间 0.715 到 0.83。两格区间不重叠，规范呈现仍高于同一张图的逆序。unseen/aligned 是 0.8，区间 0.74 到 0.855。unseen/shifted 是 0.625，区间 0.56 到 0.69，和 B0 的 0.48 到 0.62 重叠。seen/shifted 的区间也和 B0 的 0.62 到 0.75 重叠。随机打乱（`experiments/e1/pilot_mix15/b2/metrics.json`）的 seen/shifted 是 0.905，区间 0.865 到 0.94，下端高于这一轮 B1 的上端 0.83，也高于 B0 的上端 0.75。它的 seen/aligned 是 0.965，区间 0.935 到 0.99；unseen/aligned 是 0.85，区间 0.8 到 0.9；unseen/shifted 是 0.705，区间 0.64 到 0.77。unseen/shifted 的下端高于 B0 的上端 0.62，但和这一轮 B1 的 0.56 到 0.69 重叠。混入之后，规范顺序和随机打乱仍然抬高的不是同一格：分开的是 seen/shifted，unseen/shifted 没有在两档之间分开。

加一致性（`experiments/e1/pilot_mix15/b3/metrics.json`）的 seen/aligned 是 0.975，区间 0.95 到 0.995；seen/shifted 是 0.875，区间 0.83 到 0.915，和 B0 分开，和这一轮 B2 的 seen/shifted 重叠。unseen/aligned 是 0.79，区间 0.73 到 0.845。unseen/shifted 是 0.64，区间 0.57 到 0.705，和 B0 重叠，也和 B2 重叠。验证集平均准确率之差是 0.030000000000000027，B2 是 0.95，B3 是 0.9199999999999999，阈值仍是 0.05（`lambda_comparison`）。差小于阈值，所以没有改跑更大的一致性系数。

按负担挑选（`experiments/e1/pilot_mix15/ours/metrics.json`）选中负担均值是 2.2286666666666664 和 2.203333333333333，候选仍是 1.6714583333333333（`train.mine_stats`）。seen/aligned 是 0.895，区间 0.855 到 0.935。seen/shifted 是 0.84，区间 0.785 到 0.89，点估计低于 B2 的 0.905，区间和 B2 重叠。unseen/aligned 是 0.785，区间 0.73 到 0.84。unseen/shifted 是 0.7，区间 0.64 到 0.76，点估计 0.7 不高于 B2 的 0.705，区间和 B2 的 0.64 到 0.77 重叠，也和 B3 的 0.57 到 0.705 重叠。文件里仍然没有两格相减的差值区间。这一轮按负担挑选没有显示出高于随机打乱的一格。

普通数学题相对 B0 的 0.688（区间 0.65 到 0.728）都低了超过 1 个百分点。B1 是 0.544，区间 0.502 到 0.588。B2 是 0.526，区间 0.48595 到 0.5700499999999997。B3 是 0.508，区间 0.468 到 0.55。ours 是 0.458，区间 0.414 到 0.502。四格 GSM8K 的区间上端都低于 0.65。混入把部分条件从第一轮的更低点上拉回了一些，例如 B1 从 0.51 到 0.544，ours 则从 0.514 到 0.458。没有一个回到 B0 的区间里。

## 6. 能说的和不能说的

能说的上限是这一句：在深度 2、种子 0、这一套合成题上，只练规范顺序和随机打乱抬高的不是同一格。第一轮里，B1 和 B0 分开的是 seen/aligned 与 unseen/aligned；B2 比 B1 再分开的是 seen/shifted。第二轮混入大约 15% 的 GSM8K 训练题之后，seen/shifted 上 B2 的区间仍然高于 B1，unseen/shifted 上两档的区间仍然重叠。同图上的呈现变化，和换一张没见过的依赖图，在这两轮里不是同一个格子在动。

下面三句现在没有对应的结果。第一，不能说按负担挑选已经稳定超过随机打乱。第一轮 unseen/shifted 的点估计更高，区间重叠；第二轮点估计不更高，区间也重叠。两份文件都没有差值区间。第二，不能说模型已经学会可迁移的推理。unseen/shifted 在有的条件里和 B0 分开了，在 B1 和 B3 里仍然和 B0 重叠，而且这一轮没有交互项。第三，不能说普通数学能力没有损失。两轮训练条件的 GSM8K 都低于 B0 的 0.688，间隔都超过 1 个百分点，区间也不重叠。

四条限制都压在这句上限上。区间重叠：B1 和 B2 在 unseen/shifted 上两轮都重叠，ours 和 B2 在这一格上也重叠，深度 2 上「随机或逆序高于证明顺序」的配对差区间也跨过 0。只有 1 个种子：E0 的 `config.seed` 是 0，打开的每一份 E1 metrics 的 `seed` 都是 0。只有深度 2：E0 里 4、6、8 的 `gate_pass` 是 false，E1 的 `depth` 都是 2。GSM8K 下降超过 1 个百分点：对照是没有重跑的 B0，0.688 对两轮里 0.458 到 0.544 这一段点估计。另外，B0 的评估量化是 bf16-eval，训练条件是 nf4，所以相对 B0 的升降还夹着量化设定；B1 和 B2 的对比是在同一轮、同为 nf4 的文件之间做的。

## 附：和提示不一致的地方

提示里的路径 `build/e0/round1/metrics.json` 打得开，深度、门禁和区间以这份为准。E0 的区间不是 `lo` / `hi` 字段，而是每个呈现下长度为 3 的数组。

提示说深度 2 上随机顺序和逆序的准确率高于证明顺序。点估计如此：随机 0.735、逆序 0.69、规范顺序 0.65。配对差 `diff_vs_aligned.random` 的上端是 0.005，`diff_vs_aligned.reversed` 的上端是 0.05，都大于 0。稿中保留了点估计的方向，并写明区间没有和 0 分开。

提示要求写明 aligned 的回读次数是 0、reversed 是满额。回读均值不在 E0 的 metrics 里，在每一份 E1 四格里：aligned 为 0.0，seen 的 reversed 为 1.0，unseen 的 reversed 为 3.0。文件没有「满额」字段。满额是拿这两个均值对照生成代码：深度 2 的 seen 图族有 2 条核心规则，unseen 图族有 3 条，倒序的逆序对上限分别是 1 和 3。

提示举的例子是文件里没有 SS、SF、交互项。E1 两轮确实没有这三个数。E0 有 SF，没有 SS，也没有交互项。旧试点文件有 SS、SF 和 Delta_T，只用于说明 0.76 不是证据，没有拿去和 E0、E1 做差。

提示说 C2 两轮都没有过判据。metrics 里没有「C2 是否通过」的字段，也没有 Ours 减 B2 的区间。能读到的是：第一轮 unseen/shifted 点估计 ours 为 0.77、B2 为 0.705，区间重叠；第二轮点估计 ours 为 0.7、B2 为 0.705，区间重叠。负结果按这个重叠来写。GSM8K 相对 0.688 的下降超过 1 个百分点，这和提示一致，两轮都如此。

混入比例在文件里是请求 0.15、实际 0.14965986394557823，稿中同时写了这两个字段。没有再加大混入、没有开全量：打开的 `experiments/e1` metrics 只有 pilot 和 pilot_mix15 这两轮。
