# MVP 真实模型 Pilot 记录

## 运行条件

- 模型：`Qwen/Qwen2.5-1.5B-Instruct`
- 数据：`build/mvp_data/dataset.jsonl`，200 条，4 个图族，每格 50 条
- 解码：贪心解码，`max_new_tokens=4`
- GPU：RTX 5060 Laptop，约 8 GB 显存
- 环境：项目内 `.venv`，Python 3.13，`torch==2.11.0+cu128`
- 结果：`build/mvp_predictions_full/`
- 指标：`build/mvp_results_full/metrics.json`

## 结果

| cell | accuracy |
|---|---:|
| seen / canonical | 0.76 |
| seen / shifted | 0.76 |
| unseen / canonical | 0.76 |
| unseen / shifted | 0.76 |

- `SS = 1.00`
- `SF = 0.00`
- `Delta_T = 0.00`

## 解释边界

每个 cell 的标签分布都是 38 个 `yes`、12 个 `no`。模型输出几乎全部为 `yes`，所以 0.76 等于标签多数类基线。`SS=1.00` 只说明 canonical/shifted 输出没有翻转，不能说明模型已经具备结构鲁棒性；`Delta_T=0.00` 也不能说明未见结构没有影响。

## 下一轮修正

1. 让每个图族和每个 cell 的 `yes/no` 接近 1:1；
2. 在生成后随机打乱评估顺序，避免按图族排列造成顺序效应；
3. 增加更深链、更宽分支、双链合流和负例组合，使模型出现可分析的错误；
4. 至少重复两个随机种子，再决定是否把结果升级为正文中的正式实验表。
