# E1 已有答卷的 CPU 重算

本轮只读已有 `logic.jsonl`，按种子 0 重建测试池后重新计分。没有加载模型，没有训练，没有新的生成。

B0 的 `quant` 是 bf16-eval。B1、B2、B3、Ours 的 `quant` 是 nf4。`pilot_mix15` 没有 B0。

区间写成 [point, low, high]。准确率用 `stats.mean_ci`，SF 用 `stats.conditional_flip_ci`，Δ_T 与交互按 `transfer.gap_and_interaction` 的写法：两个池独立重采样，交互用同一次重采样。重采样次数为 2000，种子为 0，alpha 为 0.05。

SF = P(reversed 答错 | aligned 答对)。Δ_T(p) = acc(seen, p) - acc(unseen, p)。交互 = Δ_T(reversed) - Δ_T(aligned)。潜在任务按图哈希排序后配对。shifted 的 presentation 是 reversed。

字段路径：

- 准确率：`rounds.<轮次>.conditions.<条件>.acc.<seen|unseen>.<aligned|reversed>`
- SF：`rounds.<轮次>.conditions.<条件>.SF.<seen|unseen>`
- Δ_T：`rounds.<轮次>.conditions.<条件>.delta_T.<aligned|reversed>`
- 交互：`rounds.<轮次>.conditions.<条件>.interaction.reversed`
- 四格核对：`cell_check.rows`
- 池负担：`burden.pools.<seen|unseen>.presentations.<aligned|reversed>.<特征>`
- 图族负担：`burden.families.<图族>.presentations.<aligned|reversed>.<特征>`
- 改口分组均值：`burden.by_condition.<轮次>.<条件>.<图族>.reversed_burden_mean.<特征>`

## 四格核对

四个格子的准确率、n、n_yes、n_no 与原 metrics 的 cells 一致。准确率最大绝对差为 0.0，容许 1e-12。

每个格子的 n 为 200，n_yes 为 100，n_no 为 100。

## 准确率、SF、Δ_T、交互

### pilot

#### b0

quant 为 `bf16-eval`。
logic.jsonl 有 896 行、800 个 uid，重复 uid 有 96 个；其中生成文本相同的有 74 个，文本不同的有 22 个，抽出的答案不同的有 5 个。计分保留每个 uid 的最后一行。

seen aligned 准确率 [0.645, 0.575, 0.715]
seen reversed 准确率 [0.685, 0.62, 0.75]
SF seen [0.2868217054263566, 0.21008403361344538, 0.3643427276310603]
seen 的 n 为 200，aligned 做对 129，其中改口 37，未改口 92。
unseen aligned 准确率 [0.64, 0.57, 0.705]
unseen reversed 准确率 [0.55, 0.475, 0.62]
SF unseen [0.4765625, 0.393435617206109, 0.5639249333009944]
unseen 的 n 为 200，aligned 做对 128，其中改口 61，未改口 67。
Δ_T(aligned) [0.0050000000000000044, -0.09499999999999997, 0.09999999999999998]
Δ_T(reversed) [0.135, 0.040000000000000036, 0.235]
交互 [0.13, -0.009999999999999898, 0.265]

#### b1

quant 为 `nf4`。

seen aligned 准确率 [0.97, 0.945, 0.99]
seen reversed 准确率 [0.73, 0.665, 0.79]
SF seen [0.26804123711340205, 0.20833333333333334, 0.33160687850269643]
seen 的 n 为 200，aligned 做对 194，其中改口 52，未改口 142。
unseen aligned 准确率 [0.88, 0.83, 0.925]
unseen reversed 准确率 [0.645, 0.575, 0.71]
SF unseen [0.36363636363636365, 0.2921116804543771, 0.4375]
unseen 的 n 为 200，aligned 做对 176，其中改口 64，未改口 112。
Δ_T(aligned) [0.08999999999999997, 0.039999999999999925, 0.14]
Δ_T(reversed) [0.08499999999999996, -0.010000000000000009, 0.17500000000000004]
交互 [-0.0050000000000000044, -0.10999999999999999, 0.10000000000000009]

#### b2

quant 为 `nf4`。

seen aligned 准确率 [0.985, 0.965, 1.0]
seen reversed 准确率 [0.895, 0.85, 0.935]
SF seen [0.10152284263959391, 0.06060606060606061, 0.1457286432160804]
seen 的 n 为 200，aligned 做对 197，其中改口 20，未改口 177。
unseen aligned 准确率 [0.79, 0.735, 0.845]
unseen reversed 准确率 [0.705, 0.64, 0.77]
SF unseen [0.21518987341772153, 0.15, 0.2802547770700637]
unseen 的 n 为 200，aligned 做对 158，其中改口 34，未改口 124。
Δ_T(aligned) [0.19499999999999995, 0.135, 0.255]
Δ_T(reversed) [0.19000000000000006, 0.10999999999999999, 0.265]
交互 [-0.004999999999999893, -0.08999999999999997, 0.07499999999999996]

#### b3

quant 为 `nf4`。

seen aligned 准确率 [0.975, 0.95, 0.995]
seen reversed 准确率 [0.9, 0.86, 0.94]
SF seen [0.09743589743589744, 0.05699481865284974, 0.13919328561508457]
seen 的 n 为 200，aligned 做对 195，其中改口 19，未改口 176。
unseen aligned 准确率 [0.81, 0.755, 0.865]
unseen reversed 准确率 [0.68, 0.615, 0.745]
SF unseen [0.25308641975308643, 0.18867005635873563, 0.32278972403490835]
unseen 的 n 为 200，aligned 做对 162，其中改口 41，未改口 121。
Δ_T(aligned) [0.16499999999999992, 0.10999999999999999, 0.22499999999999998]
Δ_T(reversed) [0.21999999999999997, 0.14487499999999992, 0.29500000000000004]
交互 [0.05500000000000005, -0.029999999999999916, 0.13500000000000012]

#### ours

quant 为 `nf4`。

seen aligned 准确率 [0.885, 0.84, 0.925]
seen reversed 准确率 [0.91, 0.87, 0.945]
SF seen [0.04519774011299435, 0.017045454545454544, 0.07650273224043716]
seen 的 n 为 200，aligned 做对 177，其中改口 8，未改口 169。
unseen aligned 准确率 [0.745, 0.685, 0.805]
unseen reversed 准确率 [0.77, 0.71, 0.825]
SF unseen [0.15436241610738255, 0.09866808565531476, 0.21251008064516122]
unseen 的 n 为 200，aligned 做对 149，其中改口 23，未改口 126。
Δ_T(aligned) [0.14, 0.06499999999999995, 0.21500000000000008]
Δ_T(reversed) [0.14, 0.06500000000000006, 0.21000000000000008]
交互 [0.0, -0.08500000000000008, 0.08500000000000008]

### pilot_mix15

这一轮没有 B0。

#### b1

quant 为 `nf4`。

seen aligned 准确率 [0.965, 0.935, 0.985]
seen reversed 准确率 [0.775, 0.715, 0.83]
SF seen [0.22279792746113988, 0.16666666666666666, 0.28350515463917525]
seen 的 n 为 200，aligned 做对 193，其中改口 43，未改口 150。
unseen aligned 准确率 [0.8, 0.74, 0.855]
unseen reversed 准确率 [0.625, 0.56, 0.69]
SF unseen [0.325, 0.25308641975308643, 0.4024409418622488]
unseen 的 n 为 200，aligned 做对 160，其中改口 52，未改口 108。
Δ_T(aligned) [0.16499999999999992, 0.10499999999999998, 0.22999999999999998]
Δ_T(reversed) [0.15000000000000002, 0.05999999999999994, 0.2350000000000001]
交互 [-0.014999999999999902, -0.11999999999999988, 0.08000000000000018]

#### b2

quant 为 `nf4`。

seen aligned 准确率 [0.965, 0.935, 0.99]
seen reversed 准确率 [0.905, 0.865, 0.945]
SF seen [0.07772020725388601, 0.041666666666666664, 0.11641084521795177]
seen 的 n 为 200，aligned 做对 193，其中改口 15，未改口 178。
unseen aligned 准确率 [0.85, 0.799875, 0.895]
unseen reversed 准确率 [0.705, 0.64, 0.765]
SF unseen [0.2235294117647059, 0.16666666666666666, 0.28651685393258425]
unseen 的 n 为 200，aligned 做对 170，其中改口 38，未改口 132。
Δ_T(aligned) [0.11499999999999999, 0.05999999999999994, 0.16999999999999993]
Δ_T(reversed) [0.20000000000000007, 0.125, 0.275]
交互 [0.08500000000000008, 0.010000000000000009, 0.16000000000000003]

#### b3

quant 为 `nf4`。

seen aligned 准确率 [0.975, 0.95, 0.995]
seen reversed 准确率 [0.875, 0.825, 0.92]
SF seen [0.12307692307692308, 0.08121827411167512, 0.1717171717171717]
seen 的 n 为 200，aligned 做对 195，其中改口 24，未改口 171。
unseen aligned 准确率 [0.79, 0.73, 0.845]
unseen reversed 准确率 [0.64, 0.5748749999999999, 0.705]
SF unseen [0.25316455696202533, 0.1875, 0.3231707317073171]
unseen 的 n 为 200，aligned 做对 158，其中改口 40，未改口 118。
Δ_T(aligned) [0.18499999999999994, 0.125, 0.24512499999999932]
Δ_T(reversed) [0.235, 0.15000000000000002, 0.31500000000000006]
交互 [0.050000000000000044, -0.03500000000000003, 0.13000000000000012]

#### ours

quant 为 `nf4`。

seen aligned 准确率 [0.895, 0.85, 0.935]
seen reversed 准确率 [0.84, 0.785, 0.89]
SF seen [0.1005586592178771, 0.05714285714285714, 0.14606741573033707]
seen 的 n 为 200，aligned 做对 179，其中改口 18，未改口 161。
unseen aligned 准确率 [0.785, 0.725, 0.84]
unseen reversed 准确率 [0.7, 0.635, 0.765]
SF unseen [0.20382165605095542, 0.14093959731543623, 0.2709677419354839]
unseen 的 n 为 200，aligned 做对 157，其中改口 32，未改口 125。
Δ_T(aligned) [0.10999999999999999, 0.040000000000000036, 0.17500000000000004]
Δ_T(reversed) [0.14, 0.05500000000000005, 0.2201249999999993]
交互 [0.030000000000000027, -0.05012500000000004, 0.11499999999999999]

## 负担

负担由 `burden.burden` 在每个潜在任务的 aligned 与 reversed 上计算。seen 与 unseen 分开。改口分组只在同一图族内比较，图族是 chain、conj_fact、tree、diamond。分组限定为 aligned 做对的题，比较的是 reversed 的 back_reads、d_rel、f_active。

### seen 与 unseen

seen 有 200 个潜在任务，图族为 chain、conj_fact，n_core 取值为 2。
seen 的 aligned back_reads 是常数 0.0（n 为 200）。
seen 的 aligned d_rel 均值是 2.02（n 为 200），在该池内不是常数。
seen 的 aligned f_active 均值是 0.76（n 为 200），在该池内不是常数。
seen 的 reversed back_reads 是常数 1.0（n 为 200）。
seen 的 reversed d_rel 均值是 2.02（n 为 200），在该池内不是常数。
seen 的 reversed f_active 均值是 1.26（n 为 200），在该池内不是常数。

unseen 有 200 个潜在任务，图族为 diamond、tree，n_core 取值为 3。
unseen 的 aligned back_reads 是常数 0.0（n 为 200）。
unseen 的 aligned d_rel 均值是 1.6875（n 为 200），在该池内不是常数。
unseen 的 aligned f_active 均值是 0.84（n 为 200），在该池内不是常数。
unseen 的 reversed back_reads 是常数 3.0（n 为 200）。
unseen 的 reversed d_rel 均值是 1.6875（n 为 200），在该池内不是常数。
unseen 的 reversed f_active 均值是 1.34（n 为 200），在该池内不是常数。

### 图族内的常数

chain 属于 seen，n 为 100，n_core 取值为 2。
chain 的 aligned back_reads 在该图族内是常数 0.0。
chain 的 aligned d_rel 均值是 2.11，在该图族内不是常数。
chain 的 aligned f_active 均值是 0.74，在该图族内不是常数。
chain 的 reversed back_reads 在该图族内是常数 1.0。
chain 的 reversed d_rel 均值是 2.11，在该图族内不是常数。
chain 的 reversed f_active 均值是 1.24，在该图族内不是常数。

conj_fact 属于 seen，n 为 100，n_core 取值为 2。
conj_fact 的 aligned back_reads 在该图族内是常数 0.0。
conj_fact 的 aligned d_rel 均值是 1.93，在该图族内不是常数。
conj_fact 的 aligned f_active 均值是 0.78，在该图族内不是常数。
conj_fact 的 reversed back_reads 在该图族内是常数 1.0。
conj_fact 的 reversed d_rel 均值是 1.93，在该图族内不是常数。
conj_fact 的 reversed f_active 均值是 1.28，在该图族内不是常数。

tree 属于 unseen，n 为 100，n_core 取值为 3。
tree 的 aligned back_reads 在该图族内是常数 0.0。
tree 的 aligned d_rel 均值是 1.665，在该图族内不是常数。
tree 的 aligned f_active 均值是 0.82，在该图族内不是常数。
tree 的 reversed back_reads 在该图族内是常数 3.0。
tree 的 reversed d_rel 均值是 1.665，在该图族内不是常数。
tree 的 reversed f_active 均值是 1.32，在该图族内不是常数。

diamond 属于 unseen，n 为 100，n_core 取值为 3。
diamond 的 aligned back_reads 在该图族内是常数 0.0。
diamond 的 aligned d_rel 均值是 1.71，在该图族内不是常数。
diamond 的 aligned f_active 均值是 0.86，在该图族内不是常数。
diamond 的 reversed back_reads 在该图族内是常数 3.0。
diamond 的 reversed d_rel 均值是 1.71，在该图族内不是常数。
diamond 的 reversed f_active 均值是 1.36，在该图族内不是常数。

### aligned 做对时，reversed 负担在改口与未改口两组的均值

chain 的 reversed back_reads 在该图族内是常数 1.0，不能区分改口。
conj_fact 的 reversed back_reads 在该图族内是常数 1.0，不能区分改口。
tree 的 reversed back_reads 在该图族内是常数 3.0，不能区分改口。
diamond 的 reversed back_reads 在该图族内是常数 3.0，不能区分改口。

pilot b0 chain 的 reversed d_rel：改口组 n 为 15，均值 2.066666666666667；未改口组 n 为 57，均值 2.087719298245614。aligned 做对的题有 72。
pilot b0 chain 的 reversed f_active：改口组 n 为 15，均值 1.2666666666666666；未改口组 n 为 57，均值 1.2105263157894737。aligned 做对的题有 72。
pilot b0 conj_fact 的 reversed d_rel：改口组 n 为 22，均值 1.9090909090909092；未改口组 n 为 35，均值 1.9142857142857144。aligned 做对的题有 57。
pilot b0 conj_fact 的 reversed f_active：改口组 n 为 22，均值 1.1818181818181819；未改口组 n 为 35，均值 1.2285714285714286。aligned 做对的题有 57。
pilot b0 tree 的 reversed d_rel：改口组 n 为 28，均值 1.5；未改口组 n 为 28，均值 1.75。aligned 做对的题有 56。
pilot b0 tree 的 reversed f_active：改口组 n 为 28，均值 1.2857142857142858；未改口组 n 为 28，均值 1.2857142857142858。aligned 做对的题有 56。
pilot b0 diamond 的 reversed d_rel：改口组 n 为 33，均值 1.6666666666666667；未改口组 n 为 39，均值 1.7820512820512822。aligned 做对的题有 72。
pilot b0 diamond 的 reversed f_active：改口组 n 为 33，均值 1.303030303030303；未改口组 n 为 39，均值 1.3333333333333333。aligned 做对的题有 72。
pilot b1 chain 的 reversed d_rel：改口组 n 为 22，均值 2.1363636363636362；未改口组 n 为 75，均值 2.0933333333333333。aligned 做对的题有 97。
pilot b1 chain 的 reversed f_active：改口组 n 为 22，均值 1.0；未改口组 n 为 75，均值 1.32。aligned 做对的题有 97。
pilot b1 conj_fact 的 reversed d_rel：改口组 n 为 30，均值 2.0；未改口组 n 为 67，均值 1.8955223880597014。aligned 做对的题有 97。
pilot b1 conj_fact 的 reversed f_active：改口组 n 为 30，均值 1.0333333333333334；未改口组 n 为 67，均值 1.373134328358209。aligned 做对的题有 97。
pilot b1 tree 的 reversed d_rel：改口组 n 为 28，均值 1.7857142857142858；未改口组 n 为 61，均值 1.6065573770491803。aligned 做对的题有 89。
pilot b1 tree 的 reversed f_active：改口组 n 为 28，均值 1.0；未改口组 n 为 61，均值 1.3934426229508197。aligned 做对的题有 89。
pilot b1 diamond 的 reversed d_rel：改口组 n 为 36，均值 1.75；未改口组 n 为 51，均值 1.7058823529411764。aligned 做对的题有 87。
pilot b1 diamond 的 reversed f_active：改口组 n 为 36，均值 1.0555555555555556；未改口组 n 为 51，均值 1.5098039215686274。aligned 做对的题有 87。
pilot b2 chain 的 reversed d_rel：改口组 n 为 11，均值 2.1818181818181817；未改口组 n 为 87，均值 2.103448275862069。aligned 做对的题有 98。
pilot b2 chain 的 reversed f_active：改口组 n 为 11，均值 1.0；未改口组 n 为 87，均值 1.264367816091954。aligned 做对的题有 98。
pilot b2 conj_fact 的 reversed d_rel：改口组 n 为 9，均值 1.4444444444444444；未改口组 n 为 90，均值 1.9777777777777779。aligned 做对的题有 99。
pilot b2 conj_fact 的 reversed f_active：改口组 n 为 9，均值 1.1111111111111112；未改口组 n 为 90，均值 1.3。aligned 做对的题有 99。
pilot b2 tree 的 reversed d_rel：改口组 n 为 10，均值 1.4；未改口组 n 为 73，均值 1.6643835616438356。aligned 做对的题有 83。
pilot b2 tree 的 reversed f_active：改口组 n 为 10，均值 1.6；未改口组 n 为 73，均值 1.2328767123287672。aligned 做对的题有 83。
pilot b2 diamond 的 reversed d_rel：改口组 n 为 24，均值 1.6458333333333333；未改口组 n 为 51，均值 1.7549019607843137。aligned 做对的题有 75。
pilot b2 diamond 的 reversed f_active：改口组 n 为 24，均值 1.4583333333333333；未改口组 n 为 51，均值 1.2352941176470589。aligned 做对的题有 75。
pilot b3 chain 的 reversed d_rel：改口组 n 为 11，均值 2.090909090909091；未改口组 n 为 87，均值 2.1264367816091956。aligned 做对的题有 98。
pilot b3 chain 的 reversed f_active：改口组 n 为 11，均值 1.0；未改口组 n 为 87，均值 1.264367816091954。aligned 做对的题有 98。
pilot b3 conj_fact 的 reversed d_rel：改口组 n 为 8，均值 1.5；未改口组 n 为 89，均值 1.9550561797752808。aligned 做对的题有 97。
pilot b3 conj_fact 的 reversed f_active：改口组 n 为 8，均值 1.125；未改口组 n 为 89，均值 1.303370786516854。aligned 做对的题有 97。
pilot b3 tree 的 reversed d_rel：改口组 n 为 13，均值 1.6538461538461537；未改口组 n 为 69，均值 1.6159420289855073。aligned 做对的题有 82。
pilot b3 tree 的 reversed f_active：改口组 n 为 13，均值 1.4615384615384615；未改口组 n 为 69，均值 1.2463768115942029。aligned 做对的题有 82。
pilot b3 diamond 的 reversed d_rel：改口组 n 为 28，均值 1.7321428571428572；未改口组 n 为 52，均值 1.7596153846153846。aligned 做对的题有 80。
pilot b3 diamond 的 reversed f_active：改口组 n 为 28，均值 1.3928571428571428；未改口组 n 为 52，均值 1.2307692307692308。aligned 做对的题有 80。
pilot ours chain 的 reversed d_rel：改口组 n 为 4，均值 1.75；未改口组 n 为 93，均值 2.118279569892473。aligned 做对的题有 97。
pilot ours chain 的 reversed f_active：改口组 n 为 4，均值 1.0；未改口组 n 为 93，均值 1.2580645161290323。aligned 做对的题有 97。
pilot ours conj_fact 的 reversed d_rel：改口组 n 为 4，均值 1.75；未改口组 n 为 76，均值 1.9210526315789473。aligned 做对的题有 80。
pilot ours conj_fact 的 reversed f_active：改口组 n 为 4，均值 1.0；未改口组 n 为 76，均值 1.368421052631579。aligned 做对的题有 80。
pilot ours tree 的 reversed d_rel：改口组 n 为 8，均值 1.6875；未改口组 n 为 70，均值 1.6714285714285715。aligned 做对的题有 78。
pilot ours tree 的 reversed f_active：改口组 n 为 8，均值 1.25；未改口组 n 为 70，均值 1.4285714285714286。aligned 做对的题有 78。
pilot ours diamond 的 reversed d_rel：改口组 n 为 15，均值 1.6；未改口组 n 为 56，均值 1.6339285714285714。aligned 做对的题有 71。
pilot ours diamond 的 reversed f_active：改口组 n 为 15，均值 1.5333333333333334；未改口组 n 为 56，均值 1.5。aligned 做对的题有 71。
pilot_mix15 b1 chain 的 reversed d_rel：改口组 n 为 18，均值 2.0555555555555554；未改口组 n 为 80，均值 2.125。aligned 做对的题有 98。
pilot_mix15 b1 chain 的 reversed f_active：改口组 n 为 18，均值 1.0；未改口组 n 为 80，均值 1.2875。aligned 做对的题有 98。
pilot_mix15 b1 conj_fact 的 reversed d_rel：改口组 n 为 25，均值 2.04；未改口组 n 为 70，均值 1.9142857142857144。aligned 做对的题有 95。
pilot_mix15 b1 conj_fact 的 reversed f_active：改口组 n 为 25，均值 1.0；未改口组 n 为 70，均值 1.4。aligned 做对的题有 95。
pilot_mix15 b1 tree 的 reversed d_rel：改口组 n 为 25，均值 1.7；未改口组 n 为 61，均值 1.6311475409836065。aligned 做对的题有 86。
pilot_mix15 b1 tree 的 reversed f_active：改口组 n 为 25，均值 1.16；未改口组 n 为 61，均值 1.3442622950819672。aligned 做对的题有 86。
pilot_mix15 b1 diamond 的 reversed d_rel：改口组 n 为 27，均值 1.7222222222222223；未改口组 n 为 47，均值 1.7234042553191489。aligned 做对的题有 74。
pilot_mix15 b1 diamond 的 reversed f_active：改口组 n 为 27，均值 1.1851851851851851；未改口组 n 为 47，均值 1.3829787234042554。aligned 做对的题有 74。
pilot_mix15 b2 chain 的 reversed d_rel：改口组 n 为 7，均值 2.2857142857142856；未改口组 n 为 88，均值 2.102272727272727。aligned 做对的题有 95。
pilot_mix15 b2 chain 的 reversed f_active：改口组 n 为 7，均值 1.0；未改口组 n 为 88，均值 1.25。aligned 做对的题有 95。
pilot_mix15 b2 conj_fact 的 reversed d_rel：改口组 n 为 8，均值 1.625；未改口组 n 为 90，均值 1.9555555555555555。aligned 做对的题有 98。
pilot_mix15 b2 conj_fact 的 reversed f_active：改口组 n 为 8，均值 1.125；未改口组 n 为 90，均值 1.2777777777777777。aligned 做对的题有 98。
pilot_mix15 b2 tree 的 reversed d_rel：改口组 n 为 17，均值 1.5588235294117647；未改口组 n 为 67，均值 1.6567164179104477。aligned 做对的题有 84。
pilot_mix15 b2 tree 的 reversed f_active：改口组 n 为 17，均值 1.4705882352941178；未改口组 n 为 67，均值 1.1940298507462686。aligned 做对的题有 84。
pilot_mix15 b2 diamond 的 reversed d_rel：改口组 n 为 21，均值 1.7142857142857142；未改口组 n 为 65，均值 1.7538461538461538。aligned 做对的题有 86。
pilot_mix15 b2 diamond 的 reversed f_active：改口组 n 为 21，均值 1.380952380952381；未改口组 n 为 65，均值 1.2461538461538462。aligned 做对的题有 86。
pilot_mix15 b3 chain 的 reversed d_rel：改口组 n 为 8，均值 2.25；未改口组 n 为 90，均值 2.088888888888889。aligned 做对的题有 98。
pilot_mix15 b3 chain 的 reversed f_active：改口组 n 为 8，均值 1.0；未改口组 n 为 90，均值 1.2555555555555555。aligned 做对的题有 98。
pilot_mix15 b3 conj_fact 的 reversed d_rel：改口组 n 为 16，均值 1.8125；未改口组 n 为 81，均值 1.9506172839506173。aligned 做对的题有 97。
pilot_mix15 b3 conj_fact 的 reversed f_active：改口组 n 为 16，均值 1.0；未改口组 n 为 81，均值 1.3209876543209877。aligned 做对的题有 97。
pilot_mix15 b3 tree 的 reversed d_rel：改口组 n 为 15，均值 1.7666666666666666；未改口组 n 为 67，均值 1.6343283582089552。aligned 做对的题有 82。
pilot_mix15 b3 tree 的 reversed f_active：改口组 n 为 15，均值 1.1333333333333333；未改口组 n 为 67，均值 1.2388059701492538。aligned 做对的题有 82。
pilot_mix15 b3 diamond 的 reversed d_rel：改口组 n 为 25，均值 1.68；未改口组 n 为 51，均值 1.7254901960784315。aligned 做对的题有 76。
pilot_mix15 b3 diamond 的 reversed f_active：改口组 n 为 25，均值 1.2；未改口组 n 为 51，均值 1.2745098039215685。aligned 做对的题有 76。
pilot_mix15 ours chain 的 reversed d_rel：改口组 n 为 13，均值 1.6153846153846154；未改口组 n 为 79，均值 2.1392405063291138。aligned 做对的题有 92。
pilot_mix15 ours chain 的 reversed f_active：改口组 n 为 13，均值 1.0；未改口组 n 为 79，均值 1.3037974683544304。aligned 做对的题有 92。
pilot_mix15 ours conj_fact 的 reversed d_rel：改口组 n 为 5，均值 1.0；未改口组 n 为 82，均值 1.975609756097561。aligned 做对的题有 87。
pilot_mix15 ours conj_fact 的 reversed f_active：改口组 n 为 5，均值 1.0；未改口组 n 为 82，均值 1.329268292682927。aligned 做对的题有 87。
pilot_mix15 ours tree 的 reversed d_rel：改口组 n 为 19，均值 1.6578947368421053；未改口组 n 为 66，均值 1.6818181818181819。aligned 做对的题有 85。
pilot_mix15 ours tree 的 reversed f_active：改口组 n 为 19，均值 1.5789473684210527；未改口组 n 为 66，均值 1.303030303030303。aligned 做对的题有 85。
pilot_mix15 ours diamond 的 reversed d_rel：改口组 n 为 13，均值 1.7692307692307692；未改口组 n 为 59，均值 1.694915254237288。aligned 做对的题有 72。
pilot_mix15 ours diamond 的 reversed f_active：改口组 n 为 13，均值 1.8461538461538463；未改口组 n 为 59，均值 1.3389830508474576。aligned 做对的题有 72。

## 区间关系

这里只比较 unseen reversed 准确率的区间是否重叠。这是单次试跑、按潜在任务重采样的区间。

pilot 的 Ours 区间为 [0.77, 0.71, 0.825]，B2 区间为 [0.705, 0.64, 0.77]，二者重叠。
pilot 的 Ours 区间为 [0.77, 0.71, 0.825]，B3 区间为 [0.68, 0.615, 0.745]，二者重叠。
pilot_mix15 的 Ours 区间为 [0.7, 0.635, 0.765]，B2 区间为 [0.705, 0.64, 0.765]，二者重叠。
pilot_mix15 的 Ours 区间为 [0.7, 0.635, 0.765]，B3 区间为 [0.64, 0.5748749999999999, 0.705]，二者重叠。

产出文件是 `experiments/e1/rescore/metrics.json` 与 `experiments/e1/rescore/report.md`。
