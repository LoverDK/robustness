# 分析论文框架：同图改口，还是换图也能跟上

本稿只排逻辑骨架，不写引言正文。它按 `tech-paper-template` 的表格、四项自洽检查和方法节大纲来写，并换成 2026-10-12 已定的分析论文定位。`docs/03_paper_skeleton.md` 里把负担挑选写成方法贡献的那一格，本稿不再沿用。`docs/03_paper_skeleton.md` 与 `docs/04_c1_diagnosis.md` 本轮不改。

数字只来自本轮打开的 metrics。点估计照抄 JSON 字符串。证据表里的区间也照抄 JSON。后文若把区间说成三位小数，会写明「由端点四舍五入」。E0、wording、transfer 的区间是长度为 3 的数组，顺序是点估计、下端、上端，没有 `lo` / `hi` 字段名。E1 的区间用 `acc`、`lo`、`hi`。

本轮打开的文件里都没有名为 `SS` 的键。下面不报告 SS。

## 1. 论文类型与那一句研究问题

- 类型：New Problem/Setting Paper。
- 不是 benchmark 论文：测量用的是求解器核过的潜在任务，基准本身不是贡献。
- 不是「方法战胜随机重排」的技术论文：叙述重心是要估计的那一问。负担挑选两轮都没有和随机重排分开，只作为负结果进入结果，不作为 Key Idea。
- 定位依据（`paper-types.md` 的顺序）：主贡献不是一套基准；主贡献是一个尚未被现成准确率回答的分割。Goal 承载叙述，Key Idea 只说明这个分割为什么估得了。

研究问题，审稿人可以整句引用：

**顺序不变性训练改善的是同图稳定，还是也能迁到没见过的依赖结构。**

同图稳定看改口率 SF：依赖图还是训练见过的那一类，只换呈现，规范呈现答对的题里有多少在目标呈现上改口。换图看差距 \(\Delta_T\)：同一种写法下，见过的图族比没见过的图族高多少。交互看换一种写法之后，这个差距有没有改掉。三件事分开报。普通数学用 GSM8K，单独报，不并进这三件。

这一问的训练半边，现有试跑还没有 SF、\(\Delta_T\) 和交互。骨架允许写的，是不训练考试里这三项已经怎么动，以及试跑里四格准确率和 GSM8K 怎么动。训练之后能不能迁移，不在本稿里预写答案。

## 2. Thinking template

| Stage | Your content |
|---|---|
| Research background | 部署里的风险是：模型只在见过的题型换一种合法写法后变稳，却被当成已经会做新的依赖结构。Chen 等人（ICML 2024）表明前提顺序会改准确率；He 等人（EMNLP 2025）与 MEND（ACL 2025 Findings）用重排做监督微调；DyVal、iGSM、GSM-∞ 用图控制难度。这些书目沿 `docs/03_paper_skeleton.md` 与 `docs/HANDOFF_LEADER.md` 第 3 节，本轮没有重核原文，核实放在第 7 节。 |
| Limitation 1 | 只按每种呈现各报一个准确率，回答不了失败来自同图改口，还是来自换了一族依赖图。`docs/03_paper_skeleton.md` 把这个缺口记在 He 等人与 MEND 上；交接文档写明「现有工作确实没做这种区分」仍要核实。核实完成之前，这条限制是测量问题，不是已核对的文献判决。 |
| Limitation 2 | 用图控制难度，可以把复杂度与表面填充分开，但若规范呈现和偏移呈现不与留出图族正交，图上的下降仍然混着写法。`docs/03_paper_skeleton.md` 把 DyVal、iGSM、GSM-∞ 放在这条旁边，书目同样待核。 |
| Limitation 3 | 把四格准确率、负担挑选和 GSM8K 收成一句「鲁棒性变强且推理能力上升」，会盖住文件里方向并不相同的几项：哪一格离开不训练对照、ours 与随机重排的区间是否重叠、普通数学是降还是升。 |
| Key Idea / Our Goal | 要估计的只有这一问：顺序不变性训练改善的是同图稳定（同图改口率 SF），还是也能迁到没见过的依赖结构（换图差距 \(\Delta_T\) 及其与改口的交互）。 |
| Challenge 1 | 规范呈现就不会、同图换呈现才改口、换了依赖图才失败，可以落在不同的题上。一个打乱测试集的准确率把三件事加在一起，目标句无法估计。 |
| Challenge 2 | 两轮训练试跑的 metrics 没有 SF、\(\Delta_T\) 和交互。用四格准确率代替这三项，回答的就不是这一问。 |
| Challenge 3 | 同一轮里不同训练条件抬高的格子不一样；ours 与均匀随机重排的区间重叠；GSM8K 相对不训练下降。把它们收成一次方法胜利，负结果和能力下降都会消失。 |
| Methodology topic sentence | 用潜在任务配对，把同图改口、换图差距和二者的交互分开估计，并按已有实验逐项报告。 |
| Module A (addresses Challenge 1) | 配对估计。潜在任务与呈现分开。SF 是「规范呈现答对的前提下，目标呈现答错」的比例；\(\Delta_T\) 是同一呈现下 seen 准确率减去 unseen 准确率；交互是同一次重采样上 \(\Delta_T(p)-\Delta_T(\mathrm{aligned})\)。定义字符串以 `build/e0/transfer/metrics.json` 的 `SF_definition`、`delta_T_definition`、`interaction_definition` 为准。重采样单位是潜在任务。 |
| Module B (addresses Challenge 2) | 分跑清单。某个估计量只在该跑的 metrics 含有对应键时报告。训练试跑没有 SF、\(\Delta_T\)、交互这三项时，四格准确率只做较弱分割：该格区间是否离开不训练对照。缺的键写入第 7 节，不换算成这三项。 |
| Module C (addresses Challenge 3) | 分开报告。每种训练条件各自写哪一格离开 B0；C2 只写 ours 与 B2 的区间关系，重叠则作为负结果；GSM8K 单独对照 B0 的 `gsm8k.acc`。三项不并成一句总判断。 |
| Contribution 1 | 把同图改口和换图差距拆开定义，并在含有这些键的跑上估计。（Section 3 定义，Section 5 给不训练换图考试的数） |
| Contribution 2 | 用已有实验按估计量报告方向，不预写成功。不训练的换图考试给出 SF、\(\Delta_T\) 和交互：规范顺序与倒序的 \(\Delta_T\) 区间不跨 0，同义的 \(\Delta_T\) 以及倒序、同义的交互区间都跨 0。该跑 `config.training` 为 false，不写成训练后的迁移。（Section 5） |
| Contribution 3 | 用两轮试跑报告较弱的四格分割、C2 没有超过随机重排、以及 GSM8K 相对不训练的 0.688 下降。训练后的 SF、\(\Delta_T\) 和交互没有字段，标为待补，不用四格准确率冒充。（Section 5） |

贡献只有这三条。负担挑选不是第四条贡献。效应方向都写在第 4 节的文件字段上；上表贡献 2、3 里的方向是这些字段的读法，不是事先假定的成功。

## 3. 四项自洽检查

- Check 1 Limitations → Key Idea：pass。三条限制分别是「一个准确率把改口和换图加在一起」「图基准若不正交于呈现，下降仍然混着写法」「总判断会盖住分项方向」。目标句就是要把这三件事拆开估计。它不去承诺一个限制里没有的涨点。
- Check 2 Key Idea → Challenges：pass。三个挑战都是实现这句估计时会碰到的障碍：失败来源混在一个准确率里；已有训练文件缺少三个键；分项方向被收成方法胜利。它们不是为了给模块找名字后补的。
- Check 3 Challenges → Methodology：pass。模块 A 对挑战 1，模块 B 对挑战 2，模块 C 对挑战 3。
- Check 4 Methodology → Contributions：pass。贡献 1 由模块 A 的定义交付。贡献 2 由模块 A 在 `build/e0/transfer/metrics.json` 上的字段交付，并写明该跑没有训练。贡献 3 由模块 B、C 交付：四格只做区间分割，C2 写重叠，GSM8K 单独写，缺的三个键不计入已交付结果。没有理论收敛之类没有模块的声称，也没有「ours 战胜 B2」这种没有差值区间的声称。

四项都是 pass。骨架不标 needs user attention。

### 严重程度

- 0 CRITICAL，1 MAJOR，2 MINOR。
- MAJOR：训练半边的 SF、\(\Delta_T\)、交互没有文件。C1 问的「训练之后改善的是哪一件」因此还不能用这三个估计量回答。这个缺口写在第 7 节，贡献 3 不把它说成已经回答，所以逻辑链不断。
- MINOR 1：第 2 节背景里的论文书目本轮未重核。
- MINOR 2：同一轮里 ours 的 `train.consumed_supervised_tokens` 高于 B1、B2、B3。第一轮 B1/B2/B3 为 51458，ours 为 51628。混入轮 B1/B2/B3 为 68754，ours 为 69164。步数和 `consumed_sequences` 在轮内相同。正文若写等预算，要带上这个 token 差。
- 先处理的三件事：训练后的三项保持待补；「已有论文没做这个分割」在核实前不写成事实；等预算句带上 ours 的 token 差。

### 下一支写作技能

逻辑链通过之后，技能顺序上的下一支是 `intro-drafter`。本轮按任务停在骨架，不写引言。以后写引言时，第 6 节的禁止句和第 7 节的待补句不能进正文声称。

## 4. 证据地图

读法先固定，避免把较弱的分割写成三个估计量。

- 「区间跨 0」：该数组的下端小于 0 且上端大于 0。这是端点的读法，文件里没有一个叫「是否跨 0」的键。
- 「离开 B0」：训练条件该格的 `lo` 高于 B0 同格的 `hi`，或 `hi` 低于 B0 同格的 `lo`。本轮见到的离开都是前一种。做不到这一点就写「重叠」。
- 离开 B0 只说明这一格的准确率区间和不训练对照分开。它不是 SF，不是 \(\Delta_T\)，也不是交互。
- 相对 B0 的比较夹着量化：B0 的 `quant` 是 `bf16-eval`，训练条件的 `quant` 是 `nf4`。同一轮里 B1、B2、B3、ours 彼此都是 `nf4`。
- 打开的每一份 E1 metrics 的 `seed` 都是 0。每种条件一个种子。

### 4.1 与提示检查点不一致的地方

`build/e0/transfer/metrics.json` 与本轮提示对照后，准确率、SF、`n_latent`、`n_yes`、`n_no`、`config.training`、`config.depth`、图族、以及下面点明以外的区间端点，都与提示里的字符串一致。不一致的是三处浮点尾数，以文件为准：

| 字段 | 文件中的字符串 | 提示中的写法 |
|---|---|---|
| `delta_T.reversed[0]` | `0.14500000000000002` | `0.145` |
| `delta_T.synonym[0]` | `0.07500000000000007` | `0.075` |
| `interaction.reversed[1]` | `-0.14500000000000002` | `-0.145` |

四舍五入到三位小数后，这三处与提示相同（0.145、0.075、-0.145）。正文点估计用文件字符串。

### 4.2 不训练，换图：`build/e0/transfer/metrics.json`

这一跑有 SF、\(\Delta_T\)、交互。没有 `SS`。`config.training` 为 false，`config.adapter` 为 null，`config.depth` 为 2，`config.seed` 为 2，`config.model` 为 `Qwen/Qwen2.5-1.5B-Instruct`。`config.seen_families` 为 chain、conj_fact，`config.unseen_families` 为 tree、diamond，`config.n_per_family` 为 100。`config.conditions` 为 aligned、reversed、synonym。`n_latent.seen` / `n_latent.unseen` 为 200 / 200。`n_yes` 与 `n_no` 在 seen、unseen 上都是 100 / 100。`paired_unit` 为 `latent_task`，`n_boot` 为 2000。

定义字段：

- `SF_definition`：`P(target wrong | aligned right), latent-task bootstrap within the pool`
- `delta_T_definition`：`acc(seen, p) - acc(unseen, p); latent tasks are resampled independently in each pool`
- `interaction_definition`：`delta_T(p) - delta_T(aligned) on the same bootstrap draw`

准确率 `acc`，数组为点估计、下端、上端：

| 池 | aligned | reversed | synonym |
|---|---|---|---|
| seen | 0.71，0.649875，0.775 | 0.655，0.59，0.72 | 0.685，0.62，0.75 |
| unseen | 0.555，0.485，0.625 | 0.51，0.44，0.58 | 0.61，0.54，0.675 |

SF：

| 池 | reversed | synonym |
|---|---|---|
| seen | 0.352112676056338，0.2758357462490129，0.42657575757575755 | 0.3028169014084507，0.2318840579710145，0.379316637301787 |
| unseen | 0.5225225225225225，0.4285409035409035，0.6153846153846154 | 0.3333333333333333，0.25，0.4220448129851798 |

\(\Delta_T\)（`delta_T`，seen 减 unseen）：

| 呈现 | 点估计 | 下端 | 上端 | 区间是否跨 0 |
|---|---|---|---|---|
| aligned | 0.15499999999999992 | 0.06487499999999996 | 0.25 | 不跨 |
| reversed | 0.14500000000000002 | 0.04999999999999993 | 0.235 | 不跨 |
| synonym | 0.07500000000000007 | -0.020000000000000018 | 0.16999999999999993 | 跨 |

交互（`interaction`）：

| 呈现 | 点估计 | 下端 | 上端 | 区间是否跨 0 |
|---|---|---|---|---|
| reversed | -0.009999999999999898 | -0.14500000000000002 | 0.12 | 跨 |
| synonym | -0.07999999999999985 | -0.21499999999999997 | 0.039999999999999925 | 跨 |

这一跑允许写的只有这句：未训练模型上，见过的图族比没见过的图族更准，这是规范顺序和倒序（这两格的 \(\Delta_T\) 下端大于 0）。同义呈现的 \(\Delta_T\) 区间跨 0。倒序和同义的交互区间都跨 0，所以换一种写法并没有把这个差距明显改掉。`config.training` 为 false。它不回答训练之后能不能迁移。

三位小数（由上表端点四舍五入，只为对照）：aligned 的 \(\Delta_T\) 为 0.155（0.065–0.250），reversed 为 0.145（0.050–0.235），synonym 为 0.075（-0.020–0.170）；交互 reversed 为 -0.010（-0.145–0.120），synonym 为 -0.080（-0.215–0.040）。点估计仍以上表全长字符串为准。

### 4.3 不训练，四种写法：`build/e0/wording/metrics.json`

有准确率和 SF。没有 \(\Delta_T\)、没有交互、没有 `SS`、没有 unseen。`config.training` 为 false，`config.depth` 为 2，`config.seed` 为 1，`config.families` 只有 chain、conj_fact，`n_latent` 为 200，`n_yes` / `n_no` 为 100 / 100。`SF_definition` 写明 ambiguous 不计入 SF。`ambiguous.entailment_broken` 为 true。

准确率：aligned 0.665（0.6，0.73），reversed 0.635（0.57，0.7），synonym 0.695（0.63，0.755），ambiguous 0.585（0.52，0.655）。SF reversed 为 0.3308270676691729（0.2519056510516777，0.41008133791474277），synonym 为 0.2631578947368421（0.18880105401844532，0.3440206896551723）。两个 SF 区间的端点重叠。文件没有「两种 SF 之差」的键。

这一跑只说明同一批见过的图族上，倒序和同义改写都有改口，歧义句另报准确率。它没有换图差距。`config.seed` 是 1，transfer 的 `config.seed` 是 2。两份文件都没有共同任务标识，不能写成同一批题。

### 4.4 不训练，深度门禁：`build/e0/round1/metrics.json`

有各深度的准确率、`diff_vs_aligned` 和 SF。没有 \(\Delta_T\)、没有交互、没有 `SS`、没有 unseen。`config.families` 只有 chain、conj_fact。`config.depths` 为 2、4、6、8，`config.n_per_level` 为 200，`config.seed` 为 0，`config.model` 为 `Qwen/Qwen2.5-1.5B-Instruct`。文件没有 `training` 键。

`gate.any_level_pass` 为 true，因为只有 `levels.2.gate_pass` 为 true。深度 4、6、8 的 `gate_pass` 都是 false。

深度 2：`levels.2.acc.aligned` 为 0.65（0.58，0.715），`random` 为 0.735（0.675，0.795），`reversed` 为 0.69（0.625，0.755）。`levels.2.SF.random` 为 0.2692307692307692（0.1937921980495124，0.34558823529411764）。`levels.2.diff_vs_aligned.random` 为 -0.085（-0.175，0.005），`reversed` 为 -0.04（-0.14，0.05）。两个配对差的区间都跨 0。点估计是 random 高于 reversed、reversed 高于 aligned；区间没有和 0 分开。SF 的区间下端大于 0，所以「条件改口率」和「准确率之差」不是同一个数。

深度 4、6、8 的 `acc.aligned` 点估计为 0.56、0.51、0.495。深度 4 的区间下端是 0.49。这三档没有进入后面的训练。训练文件的 `depth` 都是 2。

这一跑没有留出图族，不能用来回答换图。

### 4.5 训练试跑缺哪些估计量

下面 9 份 E1 metrics 都没有 `SS`、`SF`、`delta_T`、`Delta_T`、`interaction` 这些键。因此训练后的改口率、换图差距和交互全部待补。四格准确率不充当它们。

每格 `n` 为 200，`n_yes` / `n_no` 为 100 / 100。`cells.*.presentation` 在 aligned 格是 `aligned`，在 shifted 格是 `reversed`。`depth` 为 2，`seed` 为 0，`model` 为 `Qwen/Qwen2.5-1.5B-Instruct`。GSM8K 的 `gsm8k.n` 为 500。不训练对照只有第一轮的 B0，混入轮没有 B0 文件。

`back_reads_mean`：aligned 格为 0.0；seen 的 reversed 为 1.0；unseen 的 reversed 为 3.0。文件没有名为「满额」的键。

### 4.6 第一轮：`experiments/e1/pilot/`

B0 不训练，`quant` 为 `bf16-eval`，`train.steps` 为 0。B1、B2、B3、ours 的 `quant` 为 `nf4`，`train.steps` 为 63，`train.consumed_sequences` 为 1000。监督 token 见第 3 节 MINOR 2。B3 的 `lambda_comparison.lambda` 为 0.1，`lambda_0_5_ran` 为 false，`b2_minus_b3` 为 0.0050000000000000044，`threshold_mean_acc` 为 0.05。ours 的 `train.mine_stats` 里，两轮 `selected_burden_mean` 为 2.2286666666666664 与 2.1401666666666666，`candidate_burden_mean` 都是 1.6714583333333333。挑选偏向更高负担。`train.n_latents_selection_changed` 为 470。

四格与 GSM8K。区间是 `lo`、`hi`。最后一列只比较该条件与 B0 的区间，不比较成 SF 或 \(\Delta_T\)。

| 条件 | 文件 | seen/aligned | seen/shifted | unseen/aligned | unseen/shifted | GSM8K |
|---|---|---|---|---|---|---|
| B0 | `experiments/e1/pilot/b0/metrics.json` | 0.645（0.58–0.71） | 0.685（0.62–0.75） | 0.64（0.57–0.705） | 0.55（0.48–0.62） | 0.688（0.65–0.728） |
| B1 | `experiments/e1/pilot/b1/metrics.json` | 0.97（0.945–0.99），离开 | 0.73（0.665–0.79），重叠 | 0.88（0.835–0.925），离开 | 0.645（0.58–0.71），重叠 | 0.51（0.468–0.554） |
| B2 | `experiments/e1/pilot/b2/metrics.json` | 0.985（0.965–1.0），离开 | 0.895（0.855–0.935），离开 | 0.79（0.735–0.845），离开 | 0.705（0.645–0.765），离开 | 0.544（0.496–0.586） |
| B3 | `experiments/e1/pilot/b3/metrics.json` | 0.975（0.95–0.995），离开 | 0.9（0.855–0.94），离开 | 0.81（0.755–0.865），离开 | 0.68（0.615–0.745），重叠 | 0.544（0.5–0.588） |
| ours | `experiments/e1/pilot/ours/metrics.json` | 0.885（0.84–0.93），离开 | 0.91（0.87–0.95），离开 | 0.745（0.685–0.805），重叠 | 0.77（0.715–0.8251249999999992），离开 | 0.514（0.468–0.56） |

同一轮、同为 nf4 的分格：seen/shifted 上，B1 的上端 0.79 低于 B2 的下端 0.855，两格区间分开。unseen/shifted 上，B1（0.58–0.71）与 B2（0.645–0.765）重叠。

C2：ours 的 unseen/shifted 点估计 0.77，B2 为 0.705，B3 为 0.68。ours 的下端 0.715 低于 B2 的上端 0.765，也低于 B3 的上端 0.745。seen/shifted 上 ours（0.87–0.95）与 B2（0.855–0.935）重叠。文件没有 ours 减 B2 的差值，也没有这个差值的区间。

GSM8K：四个训练条件的 `gsm8k.hi` 最高是 B3 的 0.588，低于 B0 的 `gsm8k.lo` 0.65。点估计 0.51、0.544、0.544、0.514 都低于 B0 的 0.688。

### 4.7 混入轮：`experiments/e1/pilot_mix15/`

没有 B0。B1、B2、B3、ours 的 `seed` 为 0，`depth` 为 2，`quant` 为 `nf4`，`train.steps` 为 74，`train.consumed_sequences` 为 1176。`gsm_mix.fraction_requested` 为 0.15，`gsm_mix.gsm_fraction` 为 0.14965986394557823，`n_synthetic_sequences` 为 1000，`n_gsm_sequences` 为 176，`n_gsm_questions` 为 88，`n_total_sequences` 为 1176，`held_out_test_n` 为 500，`selected_text_overlap_with_held_out` 为 0。B3 的 `lambda_comparison.b2_minus_b3` 为 0.030000000000000027，`b2_val_mean_acc` 为 0.95，`b3_val_mean_acc` 为 0.9199999999999999，`threshold_mean_acc` 为 0.05，`lambda_0_5_ran` 为 false。ours 的 `selected_burden_mean` 为 2.2286666666666664 与 2.203333333333333，`candidate_burden_mean` 仍是 1.6714583333333333。

对照仍用第一轮 B0。混入轮没有重跑 B0。

| 条件 | 文件 | seen/aligned | seen/shifted | unseen/aligned | unseen/shifted | GSM8K |
|---|---|---|---|---|---|---|
| B1 | `experiments/e1/pilot_mix15/b1/metrics.json` | 0.965（0.935–0.99），离开 | 0.775（0.715–0.83），重叠 | 0.8（0.74–0.855），离开 | 0.625（0.56–0.69），重叠 | 0.544（0.502–0.588） |
| B2 | `experiments/e1/pilot_mix15/b2/metrics.json` | 0.965（0.935–0.99），离开 | 0.905（0.865–0.94），离开 | 0.85（0.8–0.9），离开 | 0.705（0.64–0.77），离开 | 0.526（0.48595–0.5700499999999997） |
| B3 | `experiments/e1/pilot_mix15/b3/metrics.json` | 0.975（0.95–0.995），离开 | 0.875（0.83–0.915），离开 | 0.79（0.73–0.845），离开 | 0.64（0.57–0.705），重叠 | 0.508（0.468–0.55） |
| ours | `experiments/e1/pilot_mix15/ours/metrics.json` | 0.895（0.855–0.935），离开 | 0.84（0.785–0.89），离开 | 0.785（0.73–0.84），离开 | 0.7（0.64–0.76），离开 | 0.458（0.414–0.502） |

同一轮的分格：seen/shifted 上，B2 的下端 0.865 高于 B1 的上端 0.83。unseen/shifted 上，B2（0.64–0.77）与 B1（0.56–0.69）重叠。

C2：ours 的 unseen/shifted 点估计 0.7，不高于 B2 的 0.705。区间 0.64–0.76 与 B2 的 0.64–0.77 重叠，也与 B3 的 0.57–0.705 重叠。seen/shifted 上 ours（0.785–0.89）与 B2（0.865–0.94）重叠。文件仍然没有差值区间。

GSM8K：四个 `gsm8k.hi` 都低于 0.65。点估计 0.544、0.526、0.508、0.458 都低于 0.688。

### 4.8 两轮合在一起能读到的分割

只练规范顺序（B1）和随机打乱（B2）抬高的不是同一格。第一轮里，B1 相对 B0 分开的是两个 aligned 格，两个 shifted 格重叠；B2 四个格都离开 B0，并且 seen/shifted 的区间高于 B1。混入轮里，seen/shifted 上 B2 的区间仍然高于 B1，unseen/shifted 上两档仍然重叠。这是四格准确率的分割，不是 SF，也不是 \(\Delta_T\)。

ours 两轮都与 B2 重叠。第一轮 unseen/shifted 的点估计更高，第二轮点估计不更高。C2 是负结果。

四个训练条件、两轮，GSM8K 的区间上端都低于 B0 的下端。不能写成推理能力上升。

## 5. 将来论文的节结构

下面是节骨架，不是正文。每一节只允许使用第 4 节里有字段的句子。

### 1 引言

允许：场景、第 1 节那一句研究问题、为什么一个准确率不够、三条贡献的范围。

不允许：鲁棒性变强且推理能力上升；训练之后已经迁到新图；C2 是方法贡献；把 `build/e0/transfer` 写成训练后的结果；把再训一版 LoRA 写成已定计划。本框架文件本身不写引言段落。

### 2 相关工作

允许：点名 Chen、He、MEND、DyVal、iGSM、GSM-∞，并当场说明书目尚未在本轮重核。

不允许：在第 7 节的核实完成之前，把「他们没有做同图 / 换图分割」写成已核对事实。

### 3 估计量

开头用第 2 节的主题句。三小节：

**3.1 潜在任务与呈现。** 潜在任务是依赖图、查询和求解器标签；呈现是 aligned、reversed，以及 wording / transfer 里才有的 synonym。展开：E1 的 shifted 就是 `presentation` = `reversed`；seen 图族是 chain 与 conj_fact；unseen 图族是 tree 与 diamond，只出现在 transfer 和 E1 的 unseen 格；E0 与 wording 没有 unseen。yes/no 在这些格上是 100/100。不要把旧框架的适用域函数和八个假设写回来。

**3.2 三个估计量。** 抄 transfer 文件里的三句定义。SF 只在规范呈现答对的任务上计数。\(\Delta_T\) 是同一呈现下两个池的准确率之差，两个池独立重采样。交互是同一次重采样上的 \(\Delta_T\) 之差。展开时写明：有键才报告；本轮打开的 metrics 都没有 `SS`；E1 没有这三项。

**3.3 什么不能互相代替。** 四格准确率、SF、\(\Delta_T\)、交互、GSM8K 是五件东西。展开时各给一条第 4 节里的例子：E0 深度 2 的配对差区间跨 0，而 SF 的区间下端大于 0；transfer 的交互区间跨 0，而 aligned 与 reversed 的 \(\Delta_T\) 不跨 0。

### 4 已跑过的训练条件

允许：把 B0、B1、B2、B3、ours 写成已经跑完的因子；写步数、序列数、混入比例、负担均值、以及 ours 多出来的监督 token；写 C2 的负结果将在下一节用区间重叠报告。

不允许：把 ours 写成贡献或写成超过 B2；把「再用 LoRA 练一版来调控鲁棒性」写成计划；把监督 token 写成各条件完全相同；把未打开的全量、消融、第二种任务族写成已有结果。

### 5 已有结果

允许，并且要分开写：

1. transfer 的 SF、\(\Delta_T\)、交互，加上第 4.2 节那句限制（不训练；同义的 \(\Delta_T\) 和两种交互都跨 0）。
2. wording 与 E0 只作为同图改口的旁证，并写明它们没有 \(\Delta_T\)。
3. 两轮试跑的四格：哪一格离开 B0，哪一格重叠；B1 与 B2 在 seen/shifted 上分开、在 unseen/shifted 上重叠。
4. C2：两轮 ours 与 B2 重叠，没有差值区间。
5. GSM8K：相对 B0 的 0.688 下降，区间上端低于 0.65。

不允许：用四格准确率填写 SF、\(\Delta_T\) 或交互的表格；把点估计的高低写成区间已经分开；把不训练的 \(\Delta_T\) 写成训练后的迁移；把 GSM8K 的下降写成持平或上升。

### 6 讨论

允许：重述第 6 节禁止清单和第 7 节待补清单，并写明一个种子、只有深度 2、B0 与训练条件的量化不同。

不允许：在讨论里把待补句子改写成结论，或把负结果改写成「方向正确、只差种子」。

## 6. 禁止的总判断清单

下列句子没有文件支持，正文不能出现。

1. 鲁棒性变强，并且推理能力上升。
2. 顺序不变性训练已经把能力迁到没见过的依赖结构上。transfer 的 `config.training` 为 false；训练试跑没有 \(\Delta_T\)。
3. 四格准确率就是改口率、换图差距或交互。
4. 按负担挑选稳定超过随机重排。两轮 unseen/shifted 的区间都与 B2 重叠，文件没有差值区间。第二轮点估计 0.7 不高于 B2 的 0.705。
5. GSM8K 上升，或普通数学没有损失。对照是 B0 的 `gsm8k.acc` 0.688，训练条件的区间上端都低于 0.65。
6. 再用 LoRA 练一版来调控鲁棒性。这不是已定计划，也没有对应的新实验文件。
7. 这篇是 benchmark 论文，或是一篇以战胜随机重排为贡献的技术论文。
8. 倒序或同义的交互已经把换图差距改掉。两格 `interaction` 的区间都跨 0。
9. 同义呈现的换图差距已经和 0 分开。`delta_T.synonym` 的区间跨 0。
10. 多种子、全量训练、深度 4/6/8 的训练、更大的 GSM8K 混入、或 \(\lambda=0.5\) 已经完成。打开的 E1 都是 `seed` 0、`depth` 2；两份 `lambda_0_5_ran` 都是 false。
11. 不训练的换图考试回答了「训练之后能不能迁移」。
12. 深度 2 上随机顺序或倒序的准确率已经显著高于规范顺序。`diff_vs_aligned.random` 与 `diff_vs_aligned.reversed` 的区间都跨 0。SF 是另一个量，不能用来改写这句话。
13. 相对 B0 的升降只来自训练。B0 是 `bf16-eval`，训练条件是 `nf4`。
14. wording 与 transfer 是同一批题。种子分别是 1 和 2，文件没有共同任务标识。
15. 各训练条件的监督 token 数完全相同。ours 高于同轮的 B1、B2、B3，数字见第 3 节。

## 7. 待补清单

没有文件支持的句子放在这里，不写进贡献，也不写进将来正文的声称。

1. 训练之后的 SF、\(\Delta_T\) 和交互。`experiments/e1/pilot/` 与 `experiments/e1/pilot_mix15/` 的 metrics 没有这些键。四格准确率不填这个空。
2. 训练之后的 SS。本轮打开的全部 metrics 都没有 `SS` 键，包括三份不训练文件。
3. ours 减 B2（以及减 B3）的差值及其区间。两份 ours metrics 都没有这个字段。
4. 第二个和第三个随机种子。设计写过 3 个种子；打开的 E1 文件 `seed` 都是 0。
5. 深度 4、6、8 上的训练。E0 这三档 `gate_pass` 为 false，E1 的 `depth` 都是 2。
6. 全量 E1、消融、以及算术图那一条边界实验。本轮打开的训练结果只有 pilot 与 pilot_mix15。
7. 更大的 GSM8K 混入。混入轮的 `gsm_mix.fraction_requested` 是 0.15，没有另一档的 metrics。
8. \(\lambda=0.5\) 的一致性项。两份 B3 的 `lambda_0_5_ran` 都是 false。
9. GSM-Symbolic、ProofWriter 分布外测试、自建的 GSM8K 前提重排。这些名字出现在交接文档的设计里，本轮打开的 metrics 没有对应结果。
10. 书目核实。Chen 等人（ICML 2024）、He 等人（EMNLP 2025）、MEND（ACL 2025 Findings）、DyVal、iGSM、GSM-∞ 的题名、作者、年份、会议，以及「他们没有估计同图改口与换图差距的分割」，本轮没有对照原文。交接文档第 3.2 节写明这条仍需核实。
11. wording 与 transfer 是否共用潜在任务。两份 `config.seed` 不同，没有共同标识字段。
12. 从已有生成文件重算训练后的 SF、\(\Delta_T\) 和交互。本轮没有这样一份重算结果。本稿不把重算写成已定计划，更不把再训练写成计划。
13. 「再用 LoRA 调控鲁棒性」的任何实验安排。没有文件，也不进入计划句。
