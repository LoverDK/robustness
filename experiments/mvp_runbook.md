# MVP 实验运行手册

这份手册对应当前新增的 scripts/structshift_mvp.py。它先把符号数据、切分和指标链路跑通，再接入真实模型。生成器和分析器只依赖 Python 标准库。

## 1. 生成 Stage 0 数据

在项目根目录运行：

~~~
python scripts/structshift_mvp.py generate --output-dir build/mvp_data --per-family 100 --seed 20261008
~~~

输出：

- dataset.jsonl：每条 canonical/shifted 样本及其 solver 标签和元数据；
- predictions_template.csv：真实模型预测的填充模板；
- manifest.json：图族、随机种子、样本数和数据隔离说明。

当前 MVP 的已见图族是 chain、fork_join，未见图族是 chain_then_fork、double_chain_join。未见图族不会出现在 stability/discovery 的已见池中。

## 2. 接入模型

真实模型只需要写回 predictions.csv，字段固定为：

~~~
model,sample_id,prediction
Qwen3-1.7B,aa6a2d3cbd5129bc-canonical,yes
Qwen3-1.7B,aa6a2d3cbd5129bc-shifted,yes
~~~

prediction 可以是 yes/no，也可以是包含最终答案的短文本；分析器会取最后一个 yes/no/true/false/1/0 标记。无法解析的预测会被排除并计入未评分数。

## 3. 计算指标

~~~
python scripts/structshift_mvp.py analyze --dataset build/mvp_data/dataset.jsonl --predictions build/mvp_data/predictions.csv --output-dir build/mvp_results
~~~

输出 metrics.json，包括：

- 四格准确率；
- SS、SF 及稳定性 bootstrap 区间；
- Delta_T = R_U - R_S；
- 每格有效样本数。

这些指标只能在真实模型预测接入后用于模型实验。没有真实模型时，生成器的答案保持和切分检查只能算 Stage 0 证据。

## 3.1 本机接入一个小模型

本项目已经在 D 盘建立 `.venv`。Windows 下如果要使用 RTX 5060，应先安装 CUDA wheel，再安装其余依赖：

~~~powershell
$env:PIP_CACHE_DIR="D:\study\科研拔尖项目\.cache\pip"
.\.venv\Scripts\python.exe -m pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers accelerate
~~~

如果只执行普通的 `pip install -r`，pip 可能选择 CPU 版 torch；安装后应检查 `torch.cuda.is_available()`。旧版文档中的 Python 3.11/3.12 是保守建议，本机 Python 3.13 已验证可以运行当前推理脚本。

通用依赖安装命令仍然是：

~~~
.\.venv\Scripts\python.exe -m pip install -r experiments/requirements-hf.txt
~~~

使用默认 1.5B 指令模型跑一个小 pilot：

~~~
python scripts/run_hf_predictions.py --dataset build/mvp_data/dataset.jsonl --output-dir build/mvp_predictions --limit 50
~~~

然后把模型输出交给分析器：

~~~
python scripts/structshift_mvp.py analyze --dataset build/mvp_data/dataset.jsonl --predictions build/mvp_predictions/predictions.csv --output-dir build/mvp_results
~~~

如果显存不足，先减小 limit；不要把无法加载模型误判成数据实验失败。

## 4. 当前实验顺序

1. per-family=25：快速检查数据和模板；
2. 接入一个 1B--3B 指令模型，跑 stability/transfer pilot；
3. 检查 clean accuracy 是否足够高、SS 和 Delta_T 是否有可测差异；
4. 冻结主确认模型为三个依赖特征的向量模型；
5. 再扩大确认集和加入 LoRA。

## 5. 记录要求

每次模型运行必须另存：

- 模型名称和版本；
- 解码参数；
- 运行时间和显存；
- 预测原文；
- 解析后的标签；
- 失败和无法解析的样本；
- dataset 的 manifest 哈希。

不要覆盖原始 predictions.csv。不同模型或不同解码条件使用不同文件名。
