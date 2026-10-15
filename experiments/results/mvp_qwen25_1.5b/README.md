# Qwen2.5-1.5B-Instruct MVP 结果

这是一次探索性 pilot，不是确认性实验。

- 数据：200 条，四个 cell 各 50 条；
- 模型：`Qwen/Qwen2.5-1.5B-Instruct`；
- 推理：贪心解码，`max_new_tokens=4`；
- 设备：RTX 5060 Laptop，CUDA；
- 标签：每个 cell 为 38 个 `yes`、12 个 `no`。

由于模型几乎总输出 `yes`，0.76 等于多数类基线。`SS=1.00` 和
`Delta_T=0.00` 不能解释为结构鲁棒性或结构迁移已经得到证明。

详细指标见 `metrics.json`，运行信息见 `run_metadata.json`。原始生成文本和
完整 build 输出保留在本地 `build/mvp_predictions_full/`，不纳入版本库。
