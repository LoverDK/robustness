# Leader 交接文档

> 用途：Opus 额度有限，leader 角色随时可能交给 Grok 4.7（或其他模型）。接手的 leader 只读这一份文件，就应能继续指挥。
> 维护规则：每完成一个节点（worker 汇报、leader 做出决策、方向调整），**先更新本文件，再做下一步**。
> 最后更新：2026-10-15 14:45（UTC+8），由新会话 Grok leader 写入。用户要求当前这一轮（倒序答对的 yes 题）做完后先停，自己检查方向。不要再派下一轮。不新训。

---

## 0. 接手须知（先读这一节）

1. **角色分工**
   - leader：决定方向、设计实验、审核 worker 的产出、更新本文件、向用户汇报。
   - worker：写代码、跑实验、写文档和论文。leader 不亲自做大段实操。
   - 2026-10-14 用户指定此后 worker 继承当前模型（`inherit`）。不要再改派 `grok-4.7-high-fast`。
2. **如何联系 worker**
   - 用 cursor 命名空间里的 `Task` 工具：`subagent_type: generalPurpose`，`model: inherit`，长任务设 `run_in_background: true`。
   - 新启动的 worker 看不到对话历史，prompt 必须自带全部上下文。可以直接让它先读本文件，再给出本轮的具体任务。
   - 续派任务：`resume: <agent id>`。resume 时不能换模型。
   - 打断正在运行的 worker：`resume` 加 `interrupt: true`。
   - 用户另开的 Cursor 会话，Task 工具无法联系（实测报 "Subagent not found"）。
3. **环境的坑（每次都要写进给 worker 的 prompt）**
   - 系统是 Windows + PowerShell。沙箱不可用，所有 Shell 命令都要带 `required_permissions: ["all"]`。
   - 运行 Python：`.\.venv\Scripts\python.exe`；设 `$env:PYTHONPATH="src"`，`HF_HOME` 指向项目内 `.cache/huggingface`。
   - Grep 工具搜不到工作区外的文件，改用 `Select-String`。
   - GPU 是 RTX 5060 Laptop，8GB 显存。长任务放后台运行，日志写到 `build/logs/`。
   - 文档和幻灯片只用 WPS，不用 Microsoft Word / PowerPoint。文字用 `kwps.Application`，演示用 `kwpp.Application`。程序在 `D:\wps\WPS Office\12.1.0.28505\office6\`。不要结束用户 12:25 已经打开的 wps 进程。
4. **铁律**
   - 严禁编造任何实验数字。
   - 不 push，不 force，不改写 git 历史。
   - 旧文件要等新版完成后才删除（见第 6 节）。
   - 每条引用都必须核实（标题、作者、年份、会议）。
   - 写论文时，没有真实结果支撑的段落标为"待补"，并单独列出。

---

## 1. 研究目标与已定决策

- **用户目标：** 研究"结构鲁棒性对大模型推理能力的提升"。用户对旧框架不满意，已同意从头重写。
- **论文类型：** 2026-10-12 19:20 起改为 **以 C1 为主的分析论文**。用户已同意，不再训新模型，不开全量，不加大 GSM8K 混入。两轮试跑里 C2 没有超过随机重排，作为负结果写入。原先的技术论文定位保留在下面，供对照。不写 benchmark 论文。理由：
  1. 用户的目标是"提升"，这是方法问题；
  2. 合成依赖图上的可控推理评测已被以下工作占据：Premise Order Matters（Chen et al., ICML 2024）、DyVal（ICLR 2024）、GSM-∞（ICML 2025）、iGSM / Physics of LMs 2.1（ICLR 2025）；
  3. 8GB 显卡撑不起 benchmark 论文需要的大规模评测。
- **写作流程**（用户给定，leader 在第 3 步后加了实验关卡）：
  1. deep-research
  2. idea-evaluator
  3. tech-paper-template
  4. **实验关卡（E0 → E1 → E2 → E3）**
  5. intro-drafter
  6. paper-writer
  7. figure-designer / drawio-reconstruction
  8. paper-polish
  9. pre-submission-reviewer

  技能文件路径：`C:\Users\Qiutian\.cursor\skills\<name>\SKILL.md`。worker 每一步都要先 Read 对应的 SKILL.md。
- **输出语言：** 英文主稿 `paper/main_en.tex`，中文对照稿 `paper/main_zh.tex`，两份都要能编译出 PDF（TinyTeX 可用）。

## 2. 对旧项目的评估（为什么重写）

- 旧项目几乎全是实验前的设计文档，`deliverables/` 下有 v1–v6 六个 zip，框架反复改了很多轮。
- 唯一一次真实 pilot：Qwen2.5-1.5B，200 条数据，四格准确率全是 0.76，等于多数类基线（模型几乎全答 yes），而且 `max_new_tokens=4` 不允许思维链。结论：没有任何信号。
- 统计装置过重（适用域函数 V(τ)、8 个假设、等价界限），实证却是零。
- 旧文献综述漏掉了上面 4 篇最接近的工作。
- **值得保留的思想：**
  - 潜在任务 z=(G,q,y) 与文本呈现 r 分离；
  - 同图稳定性 SS/SF 与新图迁移 Δ_T 分开测量；
  - 2×2 设计：seen/unseen 图族 × canonical/shifted 呈现，外加交互项；
  - 稳定性池与迁移池在符号层隔离；
  - 按 latent task 做配对 bootstrap。

## 3. 当前方法设计（v2，2026-10-12 13:05 定稿）

### 3.1 v1 → v2 为什么改

v1 方法 BIT 由三部分组成：等价呈现增强、负担引导挖掘、跨呈现一致性项。第一个 worker 检索后发现：

- 重排增强 SFT 已有人做：He et al.（EMNLP 2025）、MEND（ACL Findings 2025）。He et al. 报告"均匀随机重排在各档位都最好"。
- 跨变体一致性训练已有：BCT、F2C。
- 最坏排列 / 排列组训练已有：Pearl、PA-GRPO。
- 负担引导挖掘：未检索到先例。
- R-GSM 没有公开发布。

**以上都只是元数据级别的结论，正在由 worker 重新核实。**

### 3.2 v2 的贡献

- **C1（诊断）：** 用 2×2 分解检验现有的顺序不变性训练（重排 SFT、一致性训练）改善的是同图稳定性，还是也能迁移到未见依赖结构。（需要核实：现有工作确实没做这种区分。）
- **C2（方法）：** Burden-guided Mining（BGM）。按依赖负担分数和当前模型的翻转错误挑选训练呈现。负担特征：
  - D_rel：相关前提的文本距离；
  - 回读次数：证明顺序相对文本顺序的逆序数；
  - F_active：活动依赖前沿。

  它直接挑战 He et al. 的"均匀随机最好"结论，结果是正是负都如实报告。
- **退路：** C2 未达到第 4 节判据。用户已于 2026-10-12 同意转为以 C1 为主的分析论文。不再开全量，不再加大混入，不再训新模型。

## 4. 实验设计

- **模型：** 主模型 Qwen2.5-1.5B-Instruct（已缓存）；资源允许时加 Qwen2.5-3B（未下载）。训练用 QLoRA。
- **任务族：**
  - A：ProofWriter 风格的命题规则推理，求解器验证，深度 2–8；
  - B（边界实验）：iGSM / GSM-∞ 风格的算术计算图。
- **真实基准：**
  - GSM-Symbolic（如能获取）；
  - ProofWriter 深度分布外测试集；
  - 自制的 GSM8K 前提重排版（替代未公开的 R-GSM，论文里如实说明）；
  - GSM8K clean：检查能力是否退化。

### E0：健全性关卡（无训练）

- 开思维链，max_new_tokens ≥ 512，答案用正则抽取；yes/no 标签 1:1；评估顺序随机打乱。
- 每个难度档位比较 proof-aligned、random、reversed 三种呈现，外加干扰前提版本；每格 ≥ 200 条；配对 bootstrap 给出 95% 置信区间。
- **达标条件：** 至少一个难度档位 canonical 准确率在 0.55–0.90 之间，且显著高于多数类基线；同时 SF ≥ 10%，置信区间不含 0。
- 最多调整 3 轮（调难度，或换 3B 模型）。
- **跑完后 worker 必须先停下汇报**，leader 审核后才能进入 E1。

### E1：主实验

- **条件：**
  - B0：不训练；
  - B1：canonical SFT；
  - B2：均匀随机重排 SFT（对应 He et al.）；
  - B3：随机重排 + 跨变体一致性项（BCT 风格）；
  - Ours：BGM（可叠加一致性项）。
- **严格等预算：** 训练 token 数、步数、LoRA rank=16、target modules 一致；3 个随机种子。
- **评估：** 2×2 四格准确率、SS/SF、Δ_T、交互项；真实基准；GSM8K。
- **判据：** Ours 在 unseen/shifted 格上比 B2 和 B3 中较强的一个高 ≥ 2 个百分点，且置信区间不跨 0；GSM8K 下降 ≤ 1 个百分点。

### E2 / E3

- E2：消融，分别去掉负担分数、错误引导、一致性项。
- E3：在任务族 B 上检验边界。
- 算力不够时的优先级：E0 > E1 > E2 > E3。

## 5. 进度日志（按时间倒序追加）

| 时间 | 事件 | 结果 / 产出 |
|---|---|---|
| 10-15 14:45 | 用户要求当前任务做完后暂停，自己检查方向 | `yes_order_correct` 仍在跑，worker `efc21dfb-8bda-4329-8866-eda06fc7cde3`。它结束并核对后就停。不要派下一轮，不要新训，不要改论文，不要把数字写进正文。方向改动等用户开口。 |
| 10-15 14:36 | 用户问实验是否还在推进 | 13:10 的阶段结论之后没有在跑的实验。失败侧不再切。下一轮换人群：金标 yes、规范抽出 yes、倒序抽出 yes。解析规则与 `yes_order_justified.py` 的 `parse_generation` 相同。Leader 抽查：ours 见过图 n=70，倒序 justified 70、cannot_query 0，规范 justified 70，两边都正当 70；ours 没见过图 n=41，倒序 justified 37、early 4、cannot 0，规范 justified 39、early 2，两边都正当 37；b2 见过图 n=83，倒序 justified 81、never 2、cannot 0，规范 justified 81、never 2，两边都正当 80；b2 没见过图 n=71，倒序 justified 60、early 9、never 2、cannot 0，规范 justified 57、early 11、never 3，两边都正当 53；b1 见过图 n=48，倒序与规范都是 justified 48、cannot 0，两边都正当 48；b1 没见过图 n=36，倒序 justified 35、early 1、cannot 0，规范 justified 34、early 2，两边都正当 34。写入 `experiments/e1/reversed/yes_order_correct.json`，不覆盖已有倒序文件。worker `efc21dfb-8bda-4329-8866-eda06fc7cde3`。不新训，不占 GPU，不改论文。不能写成推理上升。 |
| 10-15 13:10 | Leader 重算倒序步骤是否正当，并写下阶段结论 | `yes_order_justified.json` 十八格与重算一致，四类之和等于 n。八个训练条件在见过图上，倒序全部是未推出查询，且每一题都写了查询 cannot be proved；规范侧几乎都是前提齐了才推出查询。没见过图上只有两处例外：pilot ours 8 题里 4 题过早使用尚未出现的前提，pilot b1 58 题里 1 题正当推出后仍写无法证明。b0 的 cannot_query 为 0。阶段结论写入 `experiments/e1/stage_c1.md`。Since 句停刀。不是推理上升，C2 仍为负结果。不新训，不改论文。 |
| 10-15 12:58 | 用户要求把实验收到阶段性成果 | `yes_order_prefix.json` 十八格重算一致，`skipped_query` 全是 0。见过图链长 2，倒序前缀停在 0 或 1；没见过图链长 3，多数停在 2。集合意义上写全查询，不等于这一步的前提已经齐：pilot ours 没见过图有 4 题把查询写进 Since 句，用到的前提还没出现，最后仍写查询 cannot be proved。收口只做这一刀，写入 `experiments/e1/reversed/yes_order_justified.json`。人群与倒序改口相同。已知集从事实开始；Since 行里该实体的最后一个 “is” 是结论，前面的是前提；前提都已在已知集里才把结论加入。查询至少有一次这样加入则为 justified，否则查询当过结论则为 early，否则有 Since 步则为 never，否则 no_since。另计倒序正文是否有 “{实体} is {查询} cannot be proved”。四类之和必须等于 n。门限，不同就停：ours 见过图 n=7，倒序 never 7、cannot 7，规范 justified 7；ours 没见过图 n=8，倒序 early 4、never 4、cannot 8，规范 justified 7、early 1；b2 见过图 n=15，倒序 never 15、cannot 15，规范 justified 15；b2 没见过图 n=12，倒序 never 12、cannot 12，规范 justified 10、early 1、never 1；b1 见过图 n=48，倒序 never 48、cannot 48，规范 justified 47、early 1；b1 没见过图 n=58，倒序 never 57、justified 1、cannot 58，规范 justified 53、early 5。不新训，不占 GPU，不改论文。worker `357f6209-b9ce-4a79-97a1-276108a23189`。做完就停 Since 句，由 leader 写成阶段结论。 |
| 10-15 12:46 | Leader 重算倒序改口的 Since 链 | `yes_order_trace.json` 九个条件、见过图和没见过图共 18 格与重算一致，七类之和等于 n。门限仍是 pilot ours 见过图 n=7、含查询 0、含第 0 步 6、第一条第 0 步 4、同前提干扰 3。见过图的核心链长都是 2，查询就是第 2 条结论；倒序答 no 的题前缀停在 0 或 1，没有写到查询。ours 见过图前缀 0 有 1 题、前缀 1 有 6 题。没见过图核心链长是 3，查询是最后一条；ours 没见过图前缀 0 有 1、前缀 2 有 3、前缀 3 有 4，其中 4 题链里有查询但仍抽出 no。这和换词不同：换词在见过图上从同前提干扰进入并且写不到第 0 步。不是推理上升，也不是 C2 胜利。下一轮把九个条件的核心链前缀、以及同前提干扰开头之后有没有第 0 步，写入 `experiments/e1/reversed/yes_order_prefix.json`。worker `8d4f4d4d-2927-41f9-8c26-cba8aff1400f`。不新训，不占 GPU，不改论文。 |
| 10-15 12:36 | 用户要实验意义、进度和剩余项，并继续做实验 | 换词 Since 句对照停在 10-14 21:30，不再切。C1 的倒序只有改口率。下一轮用已有 `logic.jsonl` 的 shifted 行：金标 yes、规范抽出 yes、倒序抽出 no，看倒序 Since 链有没有查询、有没有第 0 步，第一条来自哪条规则。见过图和没见过图都做。不新训，不占 GPU。Leader 抽查 pilot ours 见过图：7 题，含查询 0，含第 0 步 6，第一条是第 0 步 4、同前提干扰 3。写到 `experiments/e1/reversed/yes_order_trace.json`，不覆盖同义文件。worker `a1bfb831-5bdb-46cf-8fce-4e9a65b05fda`。不能写成推理上升。 |
| 10-14 21:30 | Leader 核对共同题上的换词答错差 | `experiments/e1/synonym/yes_decoy_paired.json` 见过图门限与先前重算一致：ours 减 B2 为 48 题、39 对 27，差值 0.250（0.083–0.396）；混入轮 48 题、43 对 29，差值 0.292（0.146–0.438）。Leader 重算没见过图：26 题、17 对 6，差值 0.423（0.154–0.654）；混入轮 36 题、18 对 5，差值 0.361（0.167–0.556）。四个区间都不含 0。见过图只有 ours 答错的 15 题，ours 的 Since 链都不含查询、15 题都写到干扰结论，B2 的 15 题都含查询。没见过图只有 ours 答错的 13 题，ours 含查询 2、写到干扰结论 13，B2 的 13 题都含查询。这是换词答错更多，不是准确率更高。不能写成推理上升，也不能把 C2 写成方法胜利。这组 Since 句对照到此停止，不再切同一批句子。 |
| 10-14 21:25 | Leader 核对干扰规则与换词答错的交叉表 | `experiments/e1/synonym/yes_decoy_availability.json` 四格加空抽取等于 n，空抽取只有没见过图 B0 的 1 题。见过图 ours 为 39、3、9、26，B2 为 36、1、25、36，与先前重算一致。见过图上，有这条规则时答错的比例：ours 39/48，B2 36/61；没有这条规则时：ours 3/29，B2 1/37。没见过图上答错两侧人数接近，例如 ours 18 与 10，B2 14 与 11。有规则仍可答对。不能写成推理上升。共同题上 Leader 重算：见过图、两边规范都做对、且有这条规则，ours 与 B2 共 48 题，换词答错 39 对 27；混入轮也是 48 题，43 对 29。下一轮对这组共同题做配对差值的 bootstrap。不覆盖已有文件。worker `d88c93ee-5320-41f9-ac0f-85591daa2a5a`。 |
| 10-14 21:20 | Leader 核对没见过图答对的链，并看干扰规则是否每题都有 | `experiments/e1/synonym/yes_correct_since_unseen.json` 的五个 first 类之和等于 n。ours 门限 21、21、15 与先前重算一致。训练条件里，没见过图上换词答对的题几乎都把查询写进 Since 链；含第 0 步的只是一部分。B0 的 50 题里含查询的是 2。见过图金标 yes 里，同前提干扰规则不是每题都有：100 题里 62 题有。Leader 在规范做对的题上重算：ours 换词答错 42 题里 39 题有这条规则，答对 35 题里只有 9 题有；B2 是答错 37 题里 36 题有，答对 61 题里 25 题有。答错集中在有这条规则的题上，但有这条规则时仍有人答对。不能写成推理上升。下一轮对九个条件、见过图和没见过图，把有无这条规则与换词答对答错交叉计数。不覆盖已有文件。worker `d290d1eb-f87a-42d5-80d3-f5c80ab549b7`。 |
| 10-14 21:15 | Leader 核对见过图上换词答对的链 | `experiments/e1/synonym/yes_correct_since_seen.json` 的四个 first 类之和等于 n。ours 门限 35、35、26、16、9 与先前重算一致，35+42=77。Leader 重算 B2 为 61、60、41，与文件一致。训练条件里，换词答对的题几乎都在 Since 链里写到查询；含第 0 步的只是其中一部分。B0 的 57 题里，Since 链含查询的是 0。答错是链没有写到查询，答对是写到了。不能写成推理上升。抽查 ours 没见过图、换词也答对的 21 题，Since 链都含查询，含第 0 步 15。下一轮把没见过图上换词答对的题按九个条件计数。不覆盖已有文件。worker `9e90af28-badf-4b06-8dad-c55a63eda45e`。 |
| 10-14 21:05 | Leader 核对旁支上的规范链 | `experiments/e1/synonym/yes_flip_unseen_aligned_on_later.json` 的人数没有超过题数，n 与 `yes_flip_unseen_later_core.json` 一致。ours 门限 13、12、13、13 与先前重算一致。Leader 重算 B2 为 15、15、14、15、14，与文件一致。训练条件里，换词停在后步旁支的题，规范顺序几乎都写了第 0 步、查询和那条相同旁支。B2 的查询是 14/15，B3 是 21/22，其余满员。不能写成推理上升。抽查按负担挑选、见过图、换词也答对的 35 题：Since 原子都含查询，含第 0 步 26，第一条是第 0 步 16、同前提干扰 9。下一轮把见过图上换词答对的题按九个条件计数。不覆盖已有文件。worker `a3c6c686-fc79-479b-b004-75faa5211033`。 |
| 10-14 21:00 | Leader 核对没见过图的后步规则 | `experiments/e1/synonym/yes_flip_unseen_later_core.json` 的人数都没有超过题数。ours 门限 13、13、3、0 与先前重算一致。Leader 重算混入轮 ours 为 17、17、1、0。九个训练条件里，换词侧第一条后步规则的前提都在已知事实里。后续写到查询的只有 B1、B2、混入轮 B2 各 1 题。同一批 ours 的 13 题上，规范顺序的 Since 原子含查询 13、含第 0 步 12、含那条相同后步结论 13。换词侧停在这条能由事实直接触发的旁支上，规范侧把旁支和查询都写了。不能写成推理上升。下一轮对其余条件做同样的规范侧对照。不覆盖已有文件。worker `ca8d0217-29e1-4a56-8130-c60bb163687c`。 |
| 10-14 20:50 | Leader 核对没见过图的第一条 Since | `experiments/e1/synonym/yes_flip_first_since_unseen.json` 七类之和等于各条件题数。ours 与 b2 的门限与先前重算一致。Leader 重算混入轮 ours：n=30，换词侧第一条后步核心 17、同前提干扰 11、其他干扰 2。ours 换词侧后步核心打头的 13 题，对应规则的前提都在已知事实里；后续有第 0 步的 3 题，有查询的 0 题。见过图上换词链停在同前提干扰；没见过图上换词链多从另一条能直接用事实触发的核心规则进入。同前提打头后，换词侧后续第 0 步除 ours 为 2 外，其余训练条件是 0。不能写成推理上升。下一轮只计没见过图、换词侧第一条为后步核心的题：前提是否都在事实里，后续有没有第 0 步和查询。不覆盖已有文件。worker `a157932d-85da-45c2-90c9-fe0cee9528b7`。 |
| 10-14 20:45 | Leader 核对第一条 Since，并抽查没见过图 | `experiments/e1/synonym/yes_flip_first_since.json` 的 ours 与 b2 门限与先前重算一致。Leader 再重算 ours 与混入轮 ours：规范侧第一条是同前提干扰时，后续都有第 0 步和查询（23/23 与 26/26）。换词侧后续第 0 步是 0。混入轮 ours 的第一条：规范侧第 0 步 20、同前提干扰 26、其他干扰 5；换词侧同前提干扰 40、其他干扰 10、其他 1。见过图上，训练后的换词链停在干扰规则，规范链会继续。没见过图抽查不同：ours 换词改口 28 题，第一条是后步核心 13、同前提干扰 9、其他干扰 4、第 0 步 2；同前提打头的 9 题里有 2 题后续仍有第 0 步。B2 没见过图 25 题，换词侧第一条后步核心 15、同前提干扰 8。不能写成推理上升。下一轮对九个条件的没见过图用同一套第一条分类。不覆盖已有文件。worker `05196649-532f-44ef-a2d4-e513b37855c0`。 |
| 10-14 20:40 | Leader 核对规范顺序的 Since 链 | `experiments/e1/synonym/yes_flip_aligned_trace.json` 的门限通过。Leader 重算 pilot ours：规范顺序 42 题都推出第 0 步和查询，其中 35 题也推出同前提干扰。第一条 Since 的来源是第 0 步 16、同前提干扰 23、其他干扰 3。换词侧第一条是同前提干扰 38、其他干扰 4，没有第 0 步。B2 重算为 37 题推出第 0 步、35 题推出查询，两条缺查询的都在 conj_fact，抽出的答案仍是 yes。第一条 Since：规范侧第 0 步 13、同前提干扰 22；换词侧同前提干扰 36。训练后的换词改口题，差在规范链走到了查询、换词链停在干扰规则，不是规范链从不写干扰规则。B0 两侧都没有 Since 句。不能写成推理上升。下一轮把九个条件的第一条 Since 来源分开计数。不覆盖已有文件。worker `27fecfa4-be54-496f-a6af-8d2bfc22a4ca`。 |
| 10-14 20:30 | Leader 核对干扰规则来源 | `experiments/e1/synonym/yes_flip_rule_source.json` 的来源人数之和等于 other_derived，四桶之和等于题数，chain 上 other_derived 全部是同前提干扰。Leader 重算 b2 的纯同前提干扰为 28、混入轮 ours 为 39，与文件一致。frozenset 与元组的差异题数是 0。抽查题进入 same_premise_decoy。other_derived 里的同前提干扰：B1 35/47，B2 28/36，B3 40/56，ours 38/42，混入轮 B1 41/57，B2 32/44，B3 20/25，ours 39/51。后步核心都是 0。ours 的 42 题里，第 0 步结论作为整词出现 0 次。ours 相对 B2 多错的 18 题：左边同前提干扰 14；右边 18 题都推出查询，其中 14 题也推出第 0 步结论，15 题也推出同前提干扰。混入轮是 20 题，右边 20 题推出查询、10 题推出第 0 步、15 题推出同前提干扰。做对的一边不是避开干扰规则，而是同时走完核心链。B0 的 13 题没有 Since 句，但 Leader 重算有 12 题正文含 “is always”，是编号列举，不是没读到换词句。混入轮 B2 另有 1 题把换词句抄进 Since，正则抓到的是 always 而不是结论。不能写成推理上升。下一轮对同一批换词改口题，改看该模型规范顺序的生成里有没有推出第 0 步、查询和同前提干扰。不覆盖已有文件。worker `ab19defb-6c88-4d7b-9c13-28a987213c28`。 |
| 10-14 20:05 | Leader 核对换词改口的 Since 句 | `experiments/e1/synonym/yes_flip_trace.json` 四桶之和等于各条件题数，空抽取为 0。见过图、金标 yes、规范做对、换词抽出 no：b0 是 13 题且全部没有 Since 句；b1 48 题里 47 题是其他原子；b2 37 题里 36 题；b3 57 题里 56 题，另有 1 题推出第 0 步结论；ours 42 题全部是其他原子；混入轮 B1 57、B2 44、B3 25、ours 51，也都是其他原子。ours 相对 B2 多错的 18 题，ours 都没推出查询，B2 这 18 题都推出了查询。混入轮是 20 对 20，同样如此。抽查哈希 `099cd1bf…` 的第 0 步是 vivid 推出 silver，查询是 snowy；生成推出的是 windy，而 windy 来自干扰规则 vivid 推出 windy。Leader 重算 ours 的 42 题：38 题的 Since 原子是与第 0 步前提相同的干扰结论，3 题是其他干扰结论，1 题混有两类，没有后步核心结论，也没有规则外原子。不能写成推理上升。下一轮对九个条件用前提集合相等来区分同前提干扰、其他干扰、后步核心和规则外原子。不覆盖 `yes_flip_trace.json`。worker `4daf2f1d-5364-44ec-b30c-104c800c3e39`。 |
| 10-14 19:45 | 用户问实验为何没有继续 | 17:42 之后去改定理并编译 PDF，实验循环停了。九个条件的同义生成都已在盘上，不新训，不占 GPU。下一轮只读已有生成：见过图、答案为 yes、规范做对、换词后抽出 no 的题，按 Since 句统计有没有推出第 0 步结论、有没有推出查询。另看 ours 相对 B2 多错的那些题。写到 `experiments/e1/synonym/yes_flip_trace.json`，不覆盖已有同义文件。worker `6c9a7d0e-f508-412b-98f7-d6ae6f2181ae`。不能写成推理上升。 |
| 10-14 18:00 | 用户要求编译 PDF | `paper/chinese/paper_zh.pdf` 11 页、`paper/english/paper_en.pdf` 12 页，各编两遍，交叉引用已对齐。附录里原有的长路径仍超出页边。 |
| 10-14 17:42 | 用户要求定理不要写成大段文字 | `paper/chinese/paper_zh.tex` 与 `paper/english/paper_en.tex` 的五条引理和主定理改为等式。符号与 Lean 里的有限反例一致：顺序与同义保持 $y$，删前提把 $\mathrm{no}$ 变成 $\mathrm{yes}$，准确率之差不推出改口率为 $0$，两个长度为 $2$ 的世界 $\Delta_T$ 相同而改口次数为 $0$ 与 $1$，$I_{S\times T}=-J$，倒序逆序对为 $m(m-1)/2$。正支仍是闭包求解器上 $SS(f)=1$，反支仍是见过写法上为 $\mathrm{yes}$、没见过规范写法上为 $\mathrm{no}$ 的函数 $g$。没有把正支写成实验结果。另存的检查稿能编译；原来的 `paper_zh.pdf` 与 `paper_en.pdf` 当时被占用，没有覆盖。 |
| 10-14 17:32 | Leader 核对配对差值，并比较同一批 yes 题 | `experiments/e1/synonym/label_paired.json` 36 格：点估计等于两种改口数之差除以规范做对数，含 0 标记与端点一致。见过图 yes 的门限与先前手算一致：B1 为 0，B2 为 22/98，B3 为 45/96，ours 为 35/77，混入轮 ours 为 35/80。同一批见过图、两边规范都做对的 yes 题：B2 与 B1 共 95 题，换词改口 B2 有 35、B1 有 47，其中 31 题相同，B2 独有 4，B1 独有 16。B2 从 B1 那里收回的 16 题里，倒序也做对了 13 题。混入轮 B2 与 B1 共 94 题，换词改口 40 对 57，相同 36，B2 独有 4，B1 独有 21。ours 与 B1 共 75 题，换词改口 40 对 35，ours 独有 8，B1 独有 3。混入轮 ours 与 B2 共 80 题，换词改口 51 对 32，ours 独有 20，B2 独有 1。随机打乱的换词减倒序变大，是因为倒序改口降得更多；绝对的换词改口比只练规范顺序少。按负担挑选的换词改口没有同样减少。不能写成推理上升，也不能把 C2 写成方法胜利。下一轮把这组重叠计数写入 `experiments/e1/synonym/yes_item_overlap.json`。worker `65e9319d-eab3-4ed6-b944-8ef8dcae40fd`。不新训，不占 GPU。
| 10-14 | Leader 核对倒序的 yes/no，并计算换词减倒序 | `experiments/e1/synonym/label_sf_reversed.json` 的 36 格点估计等于改口数除以规范做对数，预测计数之和等于改口数，重叠标记与端点一致，同义区间与 `label_sf.json` 一致。见过图倒序：B0 yes 22/70 与 no 15/59 重叠；B1 yes 48/96 与 no 4/98 不重叠；B2 yes 15/98 与 no 5/99 重叠；B3 yes 12/96 与 no 7/99 重叠；ours yes 7/77 与 no 1/100 不重叠；混入轮 ours yes 16/80 与 no 2/99 不重叠。同一批规范做对的题上，换词改口减倒序改口，见过图的 yes：B0 −0.129（−0.271–0.000），B1 0.000（−0.135–0.135），B2 0.224（0.102–0.337），B3 0.469（0.354–0.573），ours 0.455（0.337–0.584），混入轮 B1 0.168（0.053–0.284），混入轮 B2 0.354（0.253–0.455），混入轮 B3 0.040（−0.061–0.141），混入轮 ours 0.438（0.287–0.562）。多出来的换词改口在 yes 上。见过图的 no 大多含 0。不能写成推理上升。下一轮把这组配对差值写入 `experiments/e1/synonym/label_paired.json`，不覆盖已有同义文件。worker `c2d2358f-6ef8-4d18-97a8-454eeb2c5e4d`。不新训，不占 GPU。
| 10-14 | Leader 核对同义改口的 yes/no | `experiments/e1/synonym/label_sf.json` 的 36 格：点估计等于改口数除以规范做对数，预测计数之和等于改口数，区间重叠标记与端点一致。B0、B1、B2、B3、ours、混入轮 ours 的见过图同义 SF 按图哈希重算后与文件一致。见过图上九个条件的 yes 与 no 区间都不重叠。不训练是 no 更高：13/70 对 25/59。训练之后是 yes 更高。ours 见过图 no 为 0/100，混入轮 ours 为 0/99。没见过图上 B1、B2、混入轮 B2、混入轮 B3 的 yes 与 no 区间仍重叠。抽取成功时，yes 改口都是答成 no。另用已有倒序答案看了见过图：B1 倒序也是 yes 48/96、no 4/98，和同义次数相同。B2 倒序 yes 15/98，同义是 37/98。ours 倒序 yes 7/77，同义是 42/77。只练规范顺序已经把改口集中到 yes；打乱和按负担挑选再把换词上的 yes 改口抬高。不能写成推理上升。下一轮把九个条件的倒序改口率也按 yes/no 写入 `experiments/e1/synonym/label_sf_reversed.json`，不覆盖 `label_sf.json`。worker `17be40ae-7643-49b7-a9c2-0c25528a511f`。不新训，不占 GPU。
| 10-14 | Leader 核对换词负担对比 | `experiments/e1/synonym/synonym_burden_contrast.json` 有 72 个差值，点估计等于两组均值之差，含 0 标记与端点一致。30 个不含 0：28 个为负，2 个为正。正的两个都是不训练的 f_active，在 chain 与 diamond。规范顺序的 back_reads 四个图族都是 0。B0 chain 的 f_active 差值重算为 0.767，与文件一致。混入轮 ours 见过图 179 题做对、51 题换词改口。按答案标签拆开后，大的负差值不再成立：混入轮 ours 的 conj_fact 上 26 次换词改口全是 yes，yes 内部 f_active 差值为 0。标签内仍不含 0 的格子符号不一致，而且不少改口只有 1 到 8 题。不能写成负担越高越容易在换词后改口。抽查改口率：不训练时 no 更高，见过图 yes 0.186（0.099–0.286）、no 0.424（0.290–0.556）。随机打乱和按负担挑选之后反过来，见过图的 no 几乎不改口，ours 与混入轮 ours 的 no 改口是 0。抽取失败为 0 时，yes 改口就是答成了 no。下一轮把九个条件的 yes/no 同义改口率写成 `experiments/e1/synonym/label_sf.json`。worker `20009975-92df-40f9-93f3-6452b3938a9a`。不新训，不占 GPU。
| 10-14 | Leader 核对混入轮 ours 的同义补考 | `build/e1/synonym/mix15_ours/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/mix15_ours_metrics.json` 一致。见过图 0.285（0.220–0.349）与倒序 0.101（0.057–0.146）不重叠；配对差值 0.184（0.117–0.257），不含 0。conj_fact 不重叠（换词 26/87，倒序 5/87）；chain 重叠（25/92 对 13/92）。这和第一轮 ours 由 chain 撑起不重叠不同。没见过图 0.255（0.188–0.323）与倒序 0.204（0.141–0.271）重叠；配对差值 0.051（−0.038–0.147），含 0。见过图四格：都改口 11，只倒序 7，只换词 40，都不改口 121。规范准确率 0.895 / 0.785，抽取失败 0。同一批规范做对的题上，混入轮 ours 减 B1：见过 0.102（0.028–0.171，n=176），没见过 0.208（0.100–0.323，n=130），都不含 0。减同轮 B2 的见过图 0.057（−0.012–0.126，n=174），含 0。减第一轮 ours 的见过图 −0.012（−0.071–0.041，n=169），含 0。换词上的见过减没见过是 0.055（−0.040–0.145），含 0；与规范写法差距的交互 −0.055（−0.145–0.035），含 0。C2 仍是负结果：混入后这个差值没有大于随机打乱。加载是 bfloat16 合并 adapter。已保存的 B0、B1、B2、B3、ours 和混入轮 B1、B2、B3、ours 都有同义生成。下一轮不占 GPU：在每个图族内，用规范顺序的 d_rel、f_active 比较换词改口与未改口。back_reads 在规范顺序上是 0，不测。worker `0cd619c7-beac-404c-965b-998d4f547d3d`。写到 `experiments/e1/synonym/synonym_burden_contrast.json`，不覆盖 `experiments/e1/rescore/burden_contrast.json`。不新训。
| 10-14 | Leader 核对第一轮 ours 的同义补考 | `build/e1/synonym/ours/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/ours_metrics.json` 一致。见过图 0.237（0.178–0.302）与倒序 0.045（0.017–0.077）不重叠；配对差值 0.192（0.130–0.254），不含 0。倒序改口只有 8 题。chain 不重叠（换词 28/97，倒序 4/97）；conj_fact 重叠（14/80 对 4/80）。没见过图 0.215（0.151–0.279）与倒序 0.154（0.099–0.213）重叠；配对差值 0.060（−0.027–0.141），含 0。见过图四格：都改口 4，只倒序 4，只换词 38，都不改口 131。规范准确率 0.885 / 0.745，抽取失败 0。同一批规范做对的题上，ours 减 B1 见过图 0.185（0.110–0.260，n=173），不含 0；没见过图含 0。ours 减 B2：见过图 0.108（0.040–0.182，n=176），没见过图 0.149（0.033–0.273，n=121），都不含 0。这是换词减倒序的差值，不是准确率胜利。B2 见过图规范准确率仍是 0.985，ours 是 0.885。C2 维持负结果。加载是 bfloat16 合并 adapter。下一轮用已有混入轮 ours 做同样补考，看这个差值在混入 GSM8K 后是否还在。worker `ce571a8a-a336-4650-9624-c9a9b8d3af0b`。写到 `build/e1/synonym/mix15_ours/` 与 `experiments/e1/synonym/mix15_ours_metrics.json`。不新训。
| 10-14 | Leader 核对混入轮 B3 的同义补考 | `build/e1/synonym/mix15_b3/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/mix15_b3_metrics.json` 一致。见过图 0.169（0.119–0.223）与倒序 0.123（0.081–0.172）重叠；配对差值 0.046（−0.016–0.108），含 0。没见过图 0.184（0.126–0.244）与倒序 0.253（0.188–0.323）重叠；配对差值 −0.070（−0.152–0.013），含 0。见过图四格：都改口 8，只倒序 16，只换词 25，都不改口 146。规范准确率 0.975 / 0.790，抽取失败 0。同一批规范做对的题上，混入轮 B3 减 B1 的见过图差值 −0.032（−0.111–0.048，n=189），含 0。混入轮 B3 减 B2 的见过图差值 −0.126（−0.199–−0.052，n=191），不含 0。混入轮 B3 减第一轮 B3 的见过图差值 −0.157（−0.236–−0.084，n=191），不含 0。第一轮 B3 见过图上换词多于倒序，混入 GSM8K 之后这个差值含 0，并且小于同轮只打乱。不能写成推理上升。下一轮用已有第一轮 ours：按负担挑选顺序，不是随机打乱。看换词多于倒序是否还出现。worker `eb66465f-394c-46d3-b838-8289dd02fadb`。写到 `build/e1/synonym/ours/` 与 `experiments/e1/synonym/ours_metrics.json`。不新训，不把 C2 改写成方法胜利。
| 10-14 | Leader 核对第一轮 B3 的同义补考 | `build/e1/synonym/b3/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/b3_metrics.json` 一致。见过图 0.297（0.234–0.362）与倒序 0.097（0.057–0.139）不重叠；配对差值 0.200（0.128–0.272），不含 0。没见过图 0.259（0.194–0.325）与倒序 0.253（0.189–0.323）重叠；配对差值 0.006（−0.086–0.099），含 0。见过图四格：都改口 10，只倒序 9，只换词 48，都不改口 128。规范准确率 0.975 / 0.810，抽取失败 0。B3 与 B2 用同一套 `b2-random` 呈现。同一批规范做对的题上，B3 减 B1：见过 0.206（0.127–0.291，n=189），没见过 0.149（0.007–0.291，n=148），都不含 0。B3 减 B2 的见过图差值 0.098（0.031–0.171，n=193），不含 0；没见过图 −0.013（−0.114–0.087），含 0。一致性损失没有拿掉换词多于倒序的部分，见过图上这部分比只打乱更大。加载是 bfloat16 合并 adapter。不能写成推理上升。下一轮用已有混入轮 B3 做同样补考，看这个更大的差值是否重现。worker `4569b78c-5709-4f5a-8bf5-8d073d13d12d`。写到 `build/e1/synonym/mix15_b3/` 与 `experiments/e1/synonym/mix15_b3_metrics.json`。不新训。
| 10-14 | Leader 核对混入轮 B1 的同义补考 | `build/e1/synonym/mix15_b1/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/mix15_b1_metrics.json` 一致。见过图 0.306（0.245–0.372）与倒序 0.223（0.167–0.284）重叠；没见过图 0.250（0.185–0.316）与倒序 0.325（0.253–0.402）重叠。规范准确率 0.965 / 0.800，抽取失败 0。见过图四格：两种都改口 30，只倒序 13，只换词 29，都不改口 121。配对差值（换词改口减倒序改口）在见过图为 0.083（0.021–0.150），不含 0；没见过图 −0.075（−0.163–0.013），含 0。同一批规范做对的题上，打乱减规范的这个差值：第一轮见过 0.094（0.026–0.172，n=192）、没见过 0.163（0.048–0.286，n=147）；混入轮见过 0.075（0.005–0.150，n=187）、没见过 0.110（0.007–0.221，n=145）。这四格都不含 0。与不训练比，区间含 0。B1 与 B2 都是 bfloat16 合并 adapter，这条对比不靠 B0 的加载差异。不能写成推理上升。下一轮用已有第一轮 B3：它和 B2 抽同一套随机呈现，另外加一致性损失。看多出来的换词改口是否还在。worker `ad230e65-0595-42e2-9068-7e8505e264a8`。写到 `build/e1/synonym/b3/` 与 `experiments/e1/synonym/b3_metrics.json`。不新训。
| 10-14 | Leader 核对混入轮 B2 的同义补考 | `build/e1/synonym/mix15_b2/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/mix15_b2_metrics.json` 一致。见过图 47/193 = 0.244（0.182–0.306），倒序 0.078（0.042–0.116），不重叠。没见过图 42/170 = 0.247（0.185–0.317），倒序 0.224（0.167–0.287），重叠。不重叠主要在 conj_fact（28/98 对 8/98）；chain 点估计同向但区间重叠。规范准确率 0.965 / 0.850，抽取失败 0。在规范做对的题里，见过图只换词改口 39、只倒序改口 7、两种都改口 8。第一轮 B2 见过图是 33、13、7。第一轮 B1 见过图是 26、26、26。不训练是 27、26。打乱训练多出来的是倒序仍对、换词改错的题。加载仍是 bfloat16 合并 adapter。下一轮用已有混入轮 B1 做同样补考：它的见过图规范准确率也是 0.965，用来看分离是不是打乱训练特有。worker `a73b25d5-80ed-4109-836e-646a9e2add43`。写到 `build/e1/synonym/mix15_b1/` 与 `experiments/e1/synonym/mix15_b1_metrics.json`。不新训。
| 10-14 | Leader 核对第一轮 B1 的同义补考 | `build/e1/synonym/b1/generations.jsonl` 400 行、400 个 uid。同义 SF 与 `experiments/e1/synonym/b1_metrics.json` 一致。见过图 52/194 = 0.268（0.209–0.333），与倒序 0.268（0.208–0.332）区间重叠，点估计相同。没见过图 49/176 = 0.278（0.215–0.348），与倒序 0.364（0.292–0.438）重叠。四个图族都重叠。规范准确率 0.97 / 0.88，抽取失败 0。只练规范顺序没有把换词改口抬到倒序之上。这和第一轮 B2 见过图上换词高于倒序、区间不重叠不同。下一轮用已有混入轮 B2（`build/e1/pilot_mix15/b2/adapter`）做同样补考，看这个分离是否在另一份打乱适配器上重现。worker `295ce361-b729-4451-8b1e-b9cba1fc78ac`。写到 `build/e1/synonym/mix15_b2/` 与 `experiments/e1/synonym/mix15_b2_metrics.json`，不覆盖已有同义文件。不新训。 |。
| 10-14 15:28 | Leader 核对 B0/B2 同义补考，并派 B1 | 生成各 400 行、400 个 uid。按图哈希重算，四个同义 SF 与 `experiments/e1/synonym/metrics.json` 一致，规范准确率仍是 B0 0.645/0.640、B2 0.985/0.790。B0 见过与没见过、B2 没见过：同义 SF 与倒序 SF 区间重叠。B2 见过图：同义 0.203（0.150–0.263），倒序 0.102（0.061–0.146），不重叠。chain 与 conj_fact 的点估计同向，分开看区间仍重叠（倒序改口 11 与 9）。B2 抽取失败为 0。加载是 `load_for_eval` 的 bfloat16 合并 adapter，训练记录里的 nf4 不是这次的加载方式。不能写成推理上升。下一轮只给已有 pilot B1 做同义补考，worker `1309b962-fc9a-45c0-b935-784f7d2bf096`，写到 `build/e1/synonym/b1/` 与 `experiments/e1/synonym/b1_metrics.json`，不覆盖 B0/B2。不新训。 |
| 10-14 15:12 | Leader 核对图族内负担对比 | `experiments/e1/rescore/burden_contrast.json` 有 72 个差值。点估计等于两组均值之差，含 0 的标记与区间端点一致。与已有负担均值比了 360 项，差异为 0。48 个区间含 0。24 个不含 0，其中 13 个改口人数少于 15。不含 0 的方向不一致：第一轮 B1 在 chain、conj_fact 上，改口组的 f_active 更低，差值约 −0.32 与 −0.34。不能写成负担越高越容易改口。同义补考 `17d2ab51-86c9-4911-a039-bc253ebb46a4` 仍在，不另开 GPU。 |
| 10-14 14:49 | 用户指定 worker 继承当前模型 | 14:48 时还没有同义生成进程，也没有 `burden_contrast.json`。已打断 Grok 任务。改由继承当前模型重派：负担对比 `2fc09bf9-761d-4959-8d85-332f72f691fb`，同义补考 `17d2ab51-86c9-4911-a039-bc253ebb46a4`。不新训模型。此后 worker 都继承当前模型。 |
| 10-14 14:47 | 用户指出 Grok 4.7 可用 | 14:45 的停派是因为当时交给 leader 的可用模型列表里只有 inherit 和 composer-2.5-fast。用户确认后按原约定派 grok-4.7-high-fast，派发成功。CPU：`bd7a736f-3058-4425-9413-673b67bb3de5` 在每个图族内比较改口与未改口的 d_rel、f_active。GPU：`94df02f0-a053-4ac8-a844-353d7555b509` 用已有 B0 与 B2 补考同义改写。都不新训模型。 |
| 10-14 14:45 | 用户要求一直推进实验，直到叫停 | 当时没有 `structrobust` 进程。下一轮仍不新训模型：先在每个图族内，对规范写法做对的题，用 bootstrap 比较改口与未改口的 `d_rel`、`f_active` 之差，`back_reads` 已是常数不再测；随后用已有 adapter 给 B0 与 B2 补考同义改写，看改口下降是否只发生在倒序。交接约定的 worker 模型 `grok-4.7-high-fast` 不在当前可用列表里，可用的是 `inherit` 与 `composer-2.5-fast`。未派发。 |
| 10-14 13:02 | Leader 独立复算重评分 | `experiments/e1/rescore/metrics.json` 的 36 格准确率与原 metrics 最大差 0.0。另写脚本重算 pilot 的 B0、B2 和混入轮 B2 的 SF，与文件一致。第一轮 B2 相对 B0：见过图 SF 从 0.287（0.210–0.364）到 0.102（0.061–0.146），没见过图 SF 从 0.477（0.393–0.564）到 0.215（0.150–0.280），两组区间都不重叠。规范写法的 Δ_T 从 0.005（−0.095–0.100）到 0.195（0.135–0.255）。交互在第一轮 B2 为 −0.005（−0.090–0.075），区间跨 0。`tests/test_rescore.py` 5 项通过。没有新训练。 |
| 10-14 12:44 | 用户说可以继续做实验 | 不训新模型，不开全量，不加大 GSM8K 混入，不占 GPU。worker `48c759ed-3b91-40ac-84b2-b7bb5dbfda24` 用已有 `build/e1/pilot/*/logic.jsonl` 与 `pilot_mix15/*/logic.jsonl`，按种子 0 重建测试池，配回同一图哈希，补算改口率 SF、换图差距 Δ_T 和交互。先核对四格准确率与原 metrics 一致，不一致就停。倒序负担按图族分开看。结果写到 `experiments/e1/rescore/`，不覆盖原 metrics。 |
| 10-14 12:40 | Leader 复编 Lean 并抽查论文 | `lake build` 退出码 0，Lean 4.35.0。`proof/StructRobust` 的证明文件没有 `sorry`、没有自定义 `axiom`。引理 2 的两个 100 条总体、引理 4 的互为相反数、引理 5 的 \(m(m-1)/2\) 与 1≠3，都与源文件一致。中英文稿把定理放在二乘二之后、证明放在附录，并写明不是已训练模型的实测。有未使用 `simp` 参数的警告，不构成失败。 |
| 10-14 11:50 | 用户要求用 Lean 4 证明，证出后写入中英文论文 | 本机没有 `lean` 和 `elan`。不编译 Lean 编译器源码。worker `b4d91e4d-ad9d-4721-90b3-5874b744edcf` 用 elan 安装稳定版 Lean 4，在 `proof/StructRobust` 建不依赖 mathlib 的项目。五条引理和主定理都要 `lake build` 通过，源文件里不得有 `sorry` 或自造 `axiom`。只有证完的条目才能进 `paper/chinese/paper_zh.tex` 与 `paper/english/paper_en.tex`：定理在正文，证明在附录。不得写成 Qwen 已被证明是闭包求解器，也不得写成实验已经提升了推理。不训练，不 push。 |
| 10-13 21:12 | 用户要求记下命题，今天停止 | 五条引理和主定理写入 `docs/06_propositions.md`，只陈述，不含证明。明天再考虑是否证明、是否写入 `paper/chinese/paper_zh.tex`。不新开训练，不把正支写成已有实验的结论。 |
| 10-13 21:10 | 用户询问有没有直指题目的主定理 | 主定理分成两支。闭包求解器上，换序、同义不改答案，见过的算法能用于同一演算的新图，结构鲁棒与推理提升是同一件事。任意字符串映射上，可以把见过图的所有写法都做对，同时把没见过的图做差；只在见过图上重排训练，数学上推不出迁移。先前五条命题降为引理。尚未写入 `paper_zh.tex`。现有试跑没有把 Qwen 证成闭包求解器，也不能把主定理写成已经提升了推理。 |
| 10-13 21:06 | 用户要求用证明说明每种改变 | 已读 `paper/chinese/paper_zh.tex` 第 3、4 节。原因的证明写成算子：顺序、同义、歧义、换图各改 `(G,q,y,π,ℓ)` 的哪一个坐标，并证明准确率之差不等于条件改口率。不证明训练提升了推理。`V(τ)` 仍是适用域定义，不是这条原因的证明。证明还没写进 tex。 |
| 10-13 21:02 | Leader 说明论文的解释方法 | 原因不写成「鲁棒性提升了推理」。解释是三条可分开核对的机制：文本顺序和证明顺序不一致（回读、相关句距离、未闭合前沿）解释等价重排上的改口；句子换词而顺序不变解释同义改写上的改口；图族本身没见过解释规范顺序下的换图差距。顺序训练若起作用，应压低见过图上的改口，不必然缩小换图差距，也不必然抬高 GSM8K。C2 仍是挑选策略的负结果。用已有生成核对「负担是否预测改口」尚未开算，要等用户点头。 |
| 10-13 21:00 | Leader 抽查分析论文框架 | `docs/05_analysis_framework.md` 通过。换图、四种写法、两轮试跑的点估计和「离开 / 重叠」与 metrics 一致。三处浮点尾数按文件保留。训练后的 SF、Δ_T、交互仍标待补。C2 为负结果。没有改其他文件，没有新开训练，没有写引言。 |
| 10-13 20:56 | Leader 判断理论与实验是否吻合 | 第 4 节的全量阶梯已经停掉。和已定的分析论文部分吻合：不训练的 wording 与 transfer 给出了 SF、Δ_T、交互；两轮试跑给出了四格和 GSM8K，C2 为负结果。训练后的 SF、Δ_T、交互没有写进 metrics。`build/e1/pilot` 与 `pilot_mix15` 的 `logic.jsonl` 仍在，uid 以图哈希配对，标签可按原种子重建，所以这三项是未计算，不是实验没做。倒序在 seen 上 `back_reads_mean` 为 1.0、unseen 上为 3.0，和「呈现与图族正交」并不一致。理论只需补一段识别说明，不补定理证明，不恢复 V(τ)。CPU 重算要等用户点头，不新开训练。 |
| 10-13 20:47 | 派 worker 整理分析论文框架 | 不训练。worker `a1c4b47a-150d-40ca-a597-db97f45d2180` 只新建 `docs/05_analysis_framework.md`。C2 保持负结果。训练试跑没有 SF、Δ_T、交互，框架里标成待补，不许用四格准确率冒充这三项。 | 进程已退出，无 `structrobust` 进程。`build/e0/transfer/metrics.json` 与 `rows.jsonl`、`generations.jsonl`（1200 行）的时间都是 20:45:13。`config.training` 为 false。seen/unseen 各 200 个潜在任务，yes/no 各 100。准确率 seen：aligned 0.71（0.650–0.775），reversed 0.655（0.59–0.72），synonym 0.685（0.62–0.75）；unseen：aligned 0.555（0.485–0.625），reversed 0.51（0.44–0.58），synonym 0.61（0.54–0.675）。SF seen：reversed 0.352（0.276–0.427），synonym 0.303（0.232–0.379）；unseen：reversed 0.523（0.429–0.615），synonym 0.333（0.25–0.422）。Δ_T：aligned 0.155（0.065–0.25），reversed 0.145（0.050–0.235），synonym 0.075（−0.020–0.170）。交互：reversed −0.010（−0.145–0.12），synonym −0.080（−0.215–0.040）。倒序和同义的交互区间都跨 0。这是不训练的考试，不能写成训练后的鲁棒性或推理能力上升。不新开训练。下一步只派 worker 把已有实验整理成分析论文框架。 |
| 10-13 20:40 | 派 CPU 等候，不占 GPU | 换图进程继续自己跑。新 worker `4ba2871e-d5a7-494c-bdca-58bc9c3f66a7` 只等 PID 24344 退出并读 `build/e0/transfer/metrics.json`，不新开 `structrobust`，不训练，不改代码，不整理论文。 |
| 10-13 20:39 | 新会话 leader 按 Step 0 接手 | HEAD `989e72a`。10 月 12 日决定仍有效：以 C1 为主的分析论文，不再训新模型，不开全量，不加大 GSM8K 混入；C2 两轮未超过随机重排，作负结果。全文不写成「鲁棒性变强且推理能力上升」。不把「再用 LoRA 调控鲁棒性」写成已定计划。wording 已完成（`build/e0/wording/metrics.json`）。换图仍在跑：`structrobust.transfer`，PID 2920、子进程 24344，20:29:28 启动，输出 `build/e0/transfer`。日志停在 512/1200（392s）；20:39 时 `generations.jsonl` 为 736/1200 行，无 `metrics.json`、无 `rows.jsonl`。不另开 GPU，不 resume `c23c9eb4-1647-46c9-94ab-95a74376d18c`。PPT 不负责。下一步：等该进程自己结束并落盘后，leader 核对数字，再派 worker 只整理已有实验的论文框架，不新开训练。 |
| 10-13 20:36 | 用户认为叙述偏了，上下文将满 | 10 月 12 日的决定仍然有效：以 C1 为主的分析论文，不再训新模型，C2 作负结果。C1 问的是顺序训练改善的是同图稳定还是也能迁到新图。10 月 13 日用户另外要求的 wording 考试已完成，换图考试 `build/e0/transfer` 可能仍在跑，worker `c23c9eb4-1647-46c9-94ab-95a74376d18c` 属于本会话，新会话无法 resume，也不能收到它的完成通知。新 leader 先看 `build/logs/e0_transfer.log` 和 `build/e0/transfer/metrics.json`，有进程就不要再开 GPU。不要把「再训一版来调控鲁棒性」写成已定计划。换图数出来后只整理论文框架，不新开训练。 |
| 10-13 20:33 | 用户要求第三步完成后整理论文框架 | 全文不写成「鲁棒性变强且推理能力上升」。预期是几笔分开的效应：改口率、换图差距、二者的交互、不同预习各动哪一笔，以及普通数学下降。框架等 `build/e0/transfer` 的数出来再整理。 |
| 10-13 20:22 | 用户要求按计划做下一步：换图 | 不训练。同一模型上考 seen（chain、conj_fact）和 unseen（tree、diamond），深度 2，每种图 100 题。写法为规范顺序、倒序、同义改写。报告组内 SF、\(\Delta_T\)（seen 准确率减 unseen 准确率）和交互。输出 `build/e0/transfer`。不覆盖 wording、round1、E1。 |
| 10-13 20:08 | 四种写法考试完成 | `build/e0/wording/metrics.json`，退出码 0。200 题，不训练。准确率：规范 0.665（0.60–0.73），倒序 0.635（0.57–0.70），同义 0.695（0.63–0.755），歧义 0.585（0.52–0.655）。SF 倒序 0.331（0.252–0.410），同义 0.263（0.189–0.344），两区间重叠。歧义不计入 SF。 |
| 10-13 19:45 | 用户确认现在测改写，不训练 | 同一批深度 2、200 题。四种写法：规范顺序、倒序、只把第一步改成同义句、只把第一步改成歧义句。同义句保持原规则，报告 SF；歧义句拆掉蕴含，只报准确率，不计入 SF。输出 `build/e0/wording`。不改 E0 round1 和 E1。 |
| 10-13 19:40 | 用户建议增加「改写某一步的句子」 | 同义改写保持原规则的前提和结论，只改句子；歧义改写会拆掉蕴含，不能算进 SS/SF。尚未开跑。下一步若做，只评估、不训练：同一批深度 2 的题上比较倒序、同义改写、歧义改写。待用户确认是否现在测。 |
| 10-13 18:00 | 用户审核实验是否跑偏 | 模型量 SS、SF、Δ_T、交互写在全量 E1 的 `results.csv` 里，试跑清单只要求四格准确率和 GSM8K。全量已取消。骨架要的结论仍是：顺序训练改善的是同图翻转还是换图迁移，以及负担挑选会不会改变这个答案。现有结果只能用四格准确率做较弱的分割，交不出那四个估计量。 |
| 10-13 17:40 | Leader 核对 C1 诊断稿 | `docs/04_c1_diagnosis.md` 的四格和 GSM8K 与 metrics 一致。E0 深度 2：aligned 0.65、random 0.735、reversed 0.69；配对差 random 为 -0.085（-0.175–0.005），reversed 为 -0.04（-0.14–0.05），区间跨 0。第一轮 B2 的 seen/shifted 区间高于 B1，unseen/shifted 仍重叠。混入轮同样。Ours 两轮都与 B2 重叠。没有新训练。表格不改。 |
| 10-13 17:27 | 按昨天的停点开始写 C1 诊断 | 不训新模型，不改表格，不改 PPT。派 worker 只写 `docs/04_c1_diagnosis.md`，数字必须来自 `build/e0/round1/metrics.json`、`experiments/e1/pilot/*/metrics.json` 和 `experiments/e1/pilot_mix15/*/metrics.json`。C2 写成负结果。 |
| 10-12 21:06 | 用户要求停在已完成的阶段，明天继续 | 实验不再开。表格副本已反映：混入结果、不再做全量、11 月写分析论文。第一节末句已改为「不再做全量多种子训练。」`.doc` 保存时间 21:04:22。明天第一件事：用已有的 E0 与两轮试跑写 C1 诊断，不训新模型。 |
| 10-12 21:02 | Leader 核对第五节计划改写 | `form_after_mix15.txt` 末尾的改后第五节与要求一致：不再做全量训练；11 月写分析论文、不再训练新模型；12 月整理代码和中英文稿。消融已去掉。`.doc` 保存时间 21:00:49。第一节末句仍写「全量多种子训练还没做」，与第五节矛盾，只改这一句。 |
| 10-12 20:55 | 核对混入提交，并恢复被清空的两份文档 | `43820b5`、`989e72a` 在 git log 里，作者 Codex，数字与已核对的 metrics 一致。`docs/01_literature_review.md` 与 `docs/03_paper_skeleton.md` 在 20:49 被清空成 0 字节，已从当前提交恢复（16894 与 8550 字节）。没有新的训练。第五节计划仍由 `7b184d91-1f53-404f-bc92-7ac0da41df78` 在改。 |
| 10-12 20:51 | 用户询问当前在做什么 | 没有实验进程。混入重跑 19:14 已结束。表格数字已在 19:18 写入 `form_after_mix15.txt`，与 metrics 一致。第五节第 4–6 点仍写「训练全量 / 消融」，与用户已选的分析论文不一致。只改这几句，不训新模型。 |
| 10-12 19:20 | 用户选择转为分析论文 | 不再训新模型，不开 E1 全量，不加大 GSM8K 混入。C2 两轮试跑作为负结果保留。下一步是把 E0 和两轮试跑写成 C1 诊断，不派新的 GPU 任务。表格的混入数字仍由 `26488f66-a6f2-4eae-863d-2e6961b25d67` 在改；改完后还要把第五节里「11 月中下旬做训练全量」改成写分析论文。 |
| 10-12 19:16 | 15% 混入四组完成，不开全量 | 日志 19:14:40 写 `MIX15 COMPLETE`，进程已退出。`experiments/e1/pilot_mix15/`：没见过且倒序，B1 0.625，B2 0.705（0.640–0.770），B3 0.640，Ours 0.700（0.640–0.760）。Ours 点估计低于 B2。GSM8K：B1 0.544，B2 0.526，B3 0.508，Ours 0.458，相对 B0 的 0.688 低 0.144–0.230。训练题与已考 500 题的题面重叠为 0。不加大混入，不开全量。正在把这组数字写入表格，并询问是否转分析论文。 |
| 10-12 18:42 | 15% 混入仍在跑，Ours 第二次挑选中 | 进程 25124 仍在，日志停在 18:18:58 的 `ours step 36/74`，与第一次挑选约 27 分钟不写日志的情况一致。B2 已完成：unseen/shifted 0.705，GSM8K 0.526。B3 已完成：unseen/shifted 0.640，GSM8K 0.508；验证集差距 0.030，未跑 λ=0.5。B0 不重跑。表格先不改。 |
| 10-12 17:35 | 15% 混入重跑进行中，B1 已完成 | 进程从 17:19 起只有这一份 `structrobust.e1 mix15`（父进程 37376，子进程 25124）。日志 `build/logs/e1_mix15.log`。B1 来自 `experiments/e1/pilot_mix15/b1/metrics.json`：seen/aligned 0.965，seen/shifted 0.775，unseen/aligned 0.800，unseen/shifted 0.625，GSM8K 0.544。相对上一轮 B1，GSM8K 从 0.510 升到 0.544，仍比 B0 的 0.688 低超过 1 个百分点。B2 训练已存盘，17:34 正在评估。B3、Ours 未开始。不重跑 B0，不加大混入比例。表格先不改。 |
| 10-12 17:15 | Leader 核对表格里的试跑句子 | 对照 `midterm/_inspect/form_after_pilot.txt` 与五份 `experiments/e1/pilot/*/metrics.json`。没见过且倒序：B0 0.550、B1 0.645、B2 0.705（0.645–0.765）、B3 0.680、Ours 0.770（0.715–0.825）；GSM8K 从 0.688 到 Ours 的 0.514。区间重叠和「只有 1 个种子、全量未做」都写上了。第一节 1631 字，第五节 443 字。文件仍是 `midterm/` 下 `_已填.doc`。 |
| 10-12 17:10 | 根据 E1 试跑报告做决定，不开全量 | 提交 `bf21363`、`16160e9`、`2ad9044` 属实。λ=0.5 不补跑：验证集只差 0.005，未到 0.05。全量先不开：GSM8K 从 0.688 掉到 0.51–0.54，超过 1 个百分点的判据。下一步按第 9.1 节重跑四种训练，各混入 15% 的 GSM8K 训练集，且不得使用已评测的 500 道测试题；合成题预算不变。深度 2 的槽位 random 只有 1 种文本，训练重排改为打乱全部 core 与 other 规则，评估的 shifted 仍是 reversed。 |
| 10-12 17:06 | E1 深度 2 试跑五组全部完成 | 进程退出码 0。Ours 来自 `experiments/e1/pilot/ours/metrics.json`：seen/aligned 0.885，seen/shifted 0.910，unseen/aligned 0.745，unseen/shifted 0.770（0.715–0.825），GSM8K 0.514。对照 B2 的 unseen/shifted 0.705（0.645–0.765）。点估计高 6.5 个百分点，但各自区间重叠，且只有 1 个种子，不能写成稳定超过随机重排。训练中途有 470/500 个任务换了呈现。GSM8K 仍低于 B0 的 0.688。正在改表格里「试跑未完成」的句子。 |
| 10-12 16:39 | E1 试跑 B3（随机重排加一致性）完成，Ours 已开始 | 数字来自 `experiments/e1/pilot/b3/metrics.json`。seen/aligned 0.975，seen/shifted 0.900，unseen/aligned 0.810，unseen/shifted 0.680，GSM8K 0.544。相对 B2，unseen/shifted 从 0.705 降到 0.680。验证集差距 0.005，未达到 0.05，所以没有再跑 λ=0.5。训练约 3.4 分钟。表格仍不改。 |
| 10-12 16:30 | E1 试跑 B2（均匀随机重排）完成，B3 已开始 | 数字来自 `experiments/e1/pilot/b2/metrics.json`。seen/aligned 0.985，seen/shifted 0.895，unseen/aligned 0.790，unseen/shifted 0.705，GSM8K 0.544。相对 B1：逆序格升高（0.73→0.895，0.645→0.705），unseen/aligned 从 0.88 降到 0.790，GSM8K 从 0.51 升到 0.544，仍低于 B0 的 0.688。训练约 3 分钟。表格仍不改。 |
| 10-12 16:21 | E1 试跑 B1（只练规范顺序）完成，B2 已开始 | 数字来自 `experiments/e1/pilot/b1/metrics.json`。相对 B0：seen/aligned 0.645→0.97，seen/shifted 0.685→0.73，unseen/aligned 0.640→0.88，unseen/shifted 0.550→0.645。GSM8K 0.688→0.510。训练约 27 分钟，评估约 6 分钟，loss 1.24→0.034。表格仍不改，试跑未完成。 |
| 10-12 15:48 | E1 试跑 B0（不训练）完成，B1 已开始训练 | 数字来自 `experiments/e1/pilot/b0/metrics.json`，评估约 27 分钟。seen/aligned 0.645，seen/shifted（逆序）0.685，unseen/aligned 0.640，unseen/shifted 0.550，每格 200，yes/no 各 100。GSM8K 500 题 0.688。B1 在 15:48 开始，loss 从 1.24 降到第 6 步的 0.16。表格先不改，试跑未完成。 |
| 10-12 15:18 | E1 试跑在训练前停下 | `experiments/e1/pilot/STOPPED.json`：GSM8K 报 OfflineModeIsEnabled，镜像同样失败。五种条件都没有指标。原因是前面测试把 `HF_HUB_OFFLINE` 留在了同一个 PowerShell 会话。已让 E1 agent 关掉离线、重试下载；下载仍失败就先跑合成题，不要再因 GSM8K 停掉整个试跑。GPU 当时空着。 |
| 10-12 15:08 | Leader 核对表格 agent 改过的 E0 句子 | 对照 `midterm/_inspect/form_after_e0.txt` 与已复算数字，一致。第一节至第五节已把「正在运行 / 正式数字未出」改成 10 月 12 日完成：深度 2 为 0.65（0.58–0.715）、SF 0.269（0.194–0.346）；深度 4/6/8 为 0.56、0.51、0.495。旧试点 0.76 未改，没有写成训练已完成。文件仍是 `midterm/` 下 `_已填.doc`。 |
| 10-12 14:52 | 用户要求进入下一步；每步完成后改表格里已过时的「未完成」，并用白话说明提升 | E0 已完成，表格仍写健全性检查正在运行。派表格 agent 只改这几句，数字只用已复算的 round1。同时派 E1 试跑：仅深度 2、1 个种子、训练 500 个潜在任务，五种等预算条件。不改 PPT。表格 agent `26488f66-a6f2-4eae-863d-2e6961b25d67`，E1 agent `8200e19d-7a0e-46b4-82fc-52393339f72d`。派发后立刻纠正了错误路径，正确路径是 `D:\study\科研拔尖项目`。 |
| 10-12 14:43 | Leader 复算 E0 round1（进程已退出） | 14:29 落盘 `build/e0/round1/metrics.json`。自己重跑 `analyze`，点估计与文件一致。每档 200 个潜在任务，标签总盘 yes/no = 1600/1600。深度 2 过关：aligned 0.65（0.58–0.715），对 random 的 SF 0.269（0.194–0.346），抽取失败 3.9%，答 yes 比例 0.52，aligned 答对 130。深度 4/6/8 未过：aligned 分别为 0.56、0.51、0.495；深度 4 置信下限 0.49；深度 6、8 抽取失败 14.5%、18.9%。回读次数 aligned 恒为 0、reversed 为满额（深度 2/4/6/8 为 1/6/15/28），标签没有写反。深度 2 的准确率却是 random 0.735 > reversed 0.69 > aligned 0.65。训练未开始。 |
| 10-12 14:42 | 用户：PPT 不再由 leader 负责，只核对表格；用户看到的表是空的 | 用 `kwps.Application` 读了两份 `.doc`，没有 Quit 用户已打开的 WPS。根目录原件内容栏是空的，这是未改的原件。已填副本在 `midterm/` 下的 `_已填.doc`（30208 字节，14:34）：项目名称、邱添 / [学号已从公开副本删除]、滕佳烨，以及第一至第五节都有正文。项目编号、报告时间、项目组成员、老师/专家/学院签字栏仍空。此后不改 PPT。 |
| 10-12 14:31 | 用户要求改用 WPS，不用 Microsoft Office | 本机文字 ProgID `kwps.Application`，演示 `kwpp.Application`，程序目录 `D:\wps\WPS Office\12.1.0.28505\office6\`。已打断答辩 agent。不要结束用户 12:25 起就在跑的 wps 进程。 |
| 10-12 14:04 | 用户要求表格直接在 Word 里填，不要 .doc 转 .docx | E0 round1 仍在跑，不另启 GPU。日志最后一档 1536/3200（1869s，261 tok/s）；`generations.jsonl` 14:04 约 1696/3200 行，尚无 `metrics.json` / `rows.jsonl`。答辩 agent 用 Word 转 docx 卡住（shell 715008 失败），已打断，改为复制原 `.doc` 到 `midterm/` 后用 Word 直接填写。 |
| 10-12 14:10 | 用户回复答辩四项信息，派独立答辩 agent | 答辩在 11 月上旬，具体日未定；**表格 2026-10-31 前要交**。自述 3 分钟、问答 5 分钟；PPT 页数无硬性要求，按 3 分钟压缩，不按第 10.3 节 12–15 页的默认。模板实际文件是根目录 `统院ppt模板-蓝.pptx`（用户写的是「交统院ppt模板-蓝」）。表格已知：邱添，学号 [学号已从公开副本删除]，导师滕佳烨，项目名称「结构鲁棒性对大模型推理能力提升的探究」，起止 2026-5 至今，经费留空不编。E0 round1 仍占 GPU，答辩 agent 只用 CPU，与实验并行。答辩 agent id：`2b2e7bf7-26c5-4123-9424-24f5ef3ae6f0`（grok-4.7-high-fast，后台）。产出应落在 `midterm/`。 |
| 10-12 13:52 | **Grok leader 按 Step 0 接手** | HEAD `27a6886`。docs/01–03 已落盘；`docs/02_idea_evaluation.md` 有未提交修改；本文件与根目录答辩表（`.doc`）未跟踪。E0 round1 **仍在跑**，不另启 GPU 任务：PID 17984，命令 `structrobust.e0 --depths 2 4 6 8 --n 200 --batch-size 32 --max-new-tokens 512 --seed 0 --out build/e0/round1`，13:31 启动。13:52 时 `build/e0/round1/generations.jsonl` 约 960/3200 行；`build/logs/e0_round1.log` 最后一档是 768/3200（1103s，219 tok/s）。尚无 round1 的 `metrics.json` / `rows.jsonl`，不能审核。较早的 smoke 分析曾因准确率为 None 崩溃；当前 `e0.py`（13:29）已加空值判断，round1 进程晚于该修改启动。Qwen2.5-3B 权重已下到本地缓存（`build/dl_3b.log` 显示 9/9）。`midterm/` 不存在，答辩 agent 未启动。下一步：问用户答辩四项信息，再派独立答辩 agent。 |
| 10-12 13:55 | **Opus leader 额度见底，交给新 leader** | worker-grok（`b1d95b50`）仍在后台做 E0，它的完成通知会发到旧会话，新 leader 收不到，要靠文件判断进度（见 Step 0）。中期答辩 agent 未启动 |
| 10-12 13:50 | 用户告知即将中期答辩：需要填写根目录下的中期答辩表格，并做一份 PPT | 新增第 10 节方案和派发 prompt。**下一任 leader 必须新开独立子 agent 处理，先问用户答辩日期等信息。** Opus leader 未查看表格文件，答辩 agent 尚未启动 |
| 10-12 13:45 | 用户要求交接文档写明后续每一步怎么做 | 新增第 7 节 Step 0–9 操作手册、第 9 节问题处理手册、第 11 节 worker prompt 模板。此时 worker-grok 正在做 E0 前的推理速度调优（批大小、dtype 计时） |
| 10-12 13:05 | 启动 worker-grok（agent id `b1d95b50-8949-45c6-8cd3-1c1e9bbd797e`，模型 grok-4.7-high-fast，后台运行） | 任务：写 docs/01–03 → 补 tests → 完善并运行 E0 → 停下汇报。**不写正文，不删旧文件。** |
| 10-12 13:04 | 用户要求 worker 改用 Grok 4.7 | 用 interrupt 停掉第一个 worker，拿到交接报告 |
| 10-12 13:04 | 第一个 worker（Opus，id `2724ec06-44a7-4982-a51d-806f070b95c3`）的交接 | 快照提交 `9b1dd35` 已完成，tag `pre-restart`；文献已检索但 docs/01 未写；`src/structrobust/` 下有半成品（logic.py、burden.py、prompts.py、stats.py、infer.py、e0.py）；.venv 已装 peft、bitsandbytes、matplotlib、datasets、scipy、pytest、pandas；E0 未跑 |
| 10-12 12:55 | 启动第一个 worker（Opus） | 方法 v1（BIT） |
| 10-12 12:50 | 评估旧项目，定为技术论文 | 见第 1、2 节 |

## 6. 目录清理计划（用户要求，新版完成后执行）

- **保留：** `.git`、`.gitignore`、`.venv`、`.cache`、`git_first_steps_zh.md`（用户个人的 git 教程）、**中期答辩表格原件和 `midterm/` 目录**（见第 10.2 节；清理要等答辩材料定稿后再做）、新目录 `docs/`、`src/`、`tests/`、`configs/`、`experiments/`（新内容）、`figures/`（新内容）、`paper/`（新内容）、`README.md`。
- **删除：**
  - `deliverables/` 整个目录；
  - `research/`、`manuscript/`、`scripts/`（旧的）、旧 `figures/*`（fig1/2/3 与 drawio_manifest）；
  - `paper/chinese/`、`paper/english/`；
  - `main.tex`、`README_OVERLEAF.md`；
  - `experiments/` 下的 `mvp_*`、`toy_*`、`results/mvp_qwen25_1.5b/`、`requirements-hf.txt`（如果新版有自己的 requirements）；
  - `build/` 下的旧产物。
- **执行方式：**
  1. 删除前确认旧代码中有用的部分已迁移到 `src/`；
  2. 用 `git rm`；
  3. 单独做一次 commit："cleanup legacy framework"；
  4. 把删除清单写进本文件第 5 节。
- 旧版本可以通过 tag `pre-restart` 找回。

## 7. 操作手册：接下来一步步做什么

> 这一节是给接手 leader 的执行清单，按 Step 顺序做。每个 Step 写了做什么、怎么验收、出问题怎么办。遇到具体故障，查第 9 节"问题处理手册"。7.1–7.5 是早先写的审核要点，被下面各 Step 引用，保留不删。

### Step 0：每次接手（或新开会话）先做的事

1. 读完本文件，尤其是第 0、3、4、5 节。
2. 摸清实际状态，不要只相信日志，日志可能落后：
   - `git -C "D:\study\科研拔尖项目" log --oneline -10` 和 `git status`；
   - 列出 `docs/`、`experiments/`、`src/structrobust/`、`tests/` 下有什么文件、各自的修改时间；
   - 看 `build/logs/` 下最新的日志，确认有没有训练或推理还在跑。可以查 `Get-Process python`；如果正在跑，**不要另启 GPU 任务**，8GB 显存只够一个任务。
3. **重要：** 新会话的 leader 无法 resume 旧会话里启动的 worker（会报 "Subagent not found"），也收不到旧 worker 的完成通知。所以：
   - 如果旧 worker 的产出已经落盘（文件存在、日志显示完成），直接进入对应的审核 Step；
   - 如果做到一半，就新启一个 worker，prompt 里写："先读 docs/HANDOFF_LEADER.md，再检查 X、Y 文件的现状，从 Step N 继续"。
4. 在第 5 节进度日志里追加一行："leader 交接，当前状态 = …"。
5. **检查中期答辩事项（第 10 节）。** 如果答辩 agent 还没启动，这是接手后的第一优先级：先用 AskQuestion 问用户答辩日期等信息，再按 10.4 派发。它和实验线并行推进。

### Step 1：接收 worker-grok 的 E0 汇报

- worker-grok（`b1d95b50-…`）这一轮的任务：写 docs/01–03 → 补 tests → 完善并运行 E0 → 停下汇报。
- 汇报应该包含：三个 docs 的结论、idea-evaluator 的判定、E0 数字表、测试结果、commit hash、对 E1 的建议。缺哪项就 resume 它补哪项。
- 拿到汇报后，第一件事是把要点追加到第 5 节日志。

### Step 2：审核 E0 汇报（按 7.1 逐项做；不要只读 worker 的文字，要看文件）

1. **文献：** 用 WebSearch 抽查 He et al.（EMNLP 2025）、MEND、BCT、F2C、Pearl、PA-GRPO 中至少 3 篇。核对：
   - 标题、作者、会议是否属实；
   - 做的是不是 worker 说的内容；
   - 有没有哪篇其实已经区分了"同图稳定性与新图迁移"。**如果有，C1 就不成立**，要停下来问用户。
2. **idea 评估：** 打开 `docs/02_idea_evaluation.md` 看判定结果。
   - Strong Accept 或 Accept with Revisions：继续；
   - Reject and Pivot：停下，用 AskQuestion 问用户，选项是：按评估建议修改 / 转分析论文 / 换方向。
3. **E0 数字，自己复算一遍：** 找到 E0 的预测文件（build/ 下）和分析脚本（`src/structrobust/e0.py`）。由 leader 自己跑一次分析命令（只做 CPU 统计，几秒钟），核对和汇报是否一致。逐项检查：
   - 每格样本数 ≥ 200；yes/no 是否 1:1；
   - 模型输出的 yes 比例是否落在 30%–70% 之间，否则可能又是"全答 yes"；
   - 答案抽取失败率 < 5%；
   - 原题答对的样本数 ≥ 100（SF 的分母）；
   - proof-aligned 的准确率 ≥ random ≥ reversed，方向是否和 Chen et al. 一致；
   - 达标条件见第 4 节 E0。
4. **代码：** 运行 `.\.venv\Scripts\python.exe -m pytest tests -q`（需设 PYTHONPATH=src），必须全部通过。抽读测试文件，确认真的测了：
   - 求解器正确性；
   - 等价呈现不改变答案；
   - 稳定性池与迁移池无图哈希重叠。
5. 审核结论写进第 5 节日志，然后向用户简短汇报：先说结论，再说关键数字，最后说下一步。

### Step 3：按 E0 结果做决策（细则见 7.2）

- **达标：** 进入 Step 4。
- **接近达标**（有信号，但 SF 在 5%–10% 之间，或准确率不在 0.55–0.90 区间内）：resume worker 调整一轮。可调的有：
  - 加深度到 6–8；
  - 加 2–4 条干扰前提；
  - 把 reversed 作为主 shifted 呈现。
  
  每轮调整都要记录改了什么。
- **完全没信号：** 先排查 bug（见第 9 节"准确率等于多数类"和"三种呈现准确率一样"两条），再换 Qwen2.5-3B，3B 要用 4-bit 加载才能塞进 8GB。
- **3 轮仍不达标：** 用 AskQuestion 问用户，选项见 7.2。

### Step 4：E1 试跑（pilot，1 个种子，约 1/4 预算）

派给 worker（resume `b1d95b50` 或新启一个 grok worker），prompt 用第 11 节的模板，并写明以下默认配置。worker 可以调整，但要说明理由：

- **训练数据：** 从稳定性池的 seen 图族中取 2000 个 latent task z，作为 SFT 训练集（pilot 用 500 个）。另留 200 个 z 作为验证集。**测试集不许碰。**
- **监督目标：** 求解器生成的推理过程（按证明顺序逐步推导），最后一行 `Answer: yes/no`。
- **等预算的定义：** 所有条件用同一批 z；每个 z 恰好 2 条训练序列；总步数一致。
  - B1：canonical 呈现 ×2；
  - B2：2 个均匀随机重排；
  - B3：2 个随机重排，加一致性项；
  - Ours：2 个由 BGM 挖出的呈现。
- **QLoRA 默认值：**
  - 4-bit NF4，计算精度 bf16；
  - r=16，alpha=32，dropout=0.05；
  - target modules：q, k, v, o, gate, up, down 投影层；
  - lr=2e-4，cosine 衰减，warmup 3%；
  - 等效 batch 16（单卡 batch 2–4，配梯度累积）；
  - 最大序列长度 1024，1 个 epoch；
  - 开启 gradient checkpointing。
- **BGM 挖掘（默认用便宜版本）：**
  1. 每个 z 生成 K=8 个等价呈现；
  2. 用当前模型对每个呈现算"金标准推理过程的平均 NLL"，teacher forcing，只需一次前向，比生成快得多；
  3. 挖掘分数 = 标准化 NLL + 标准化负担分数 B（等权）；
  4. 每个 z 取分数最高的 2 个呈现；
  5. 训练到一半时用当前 checkpoint 重新挖掘一次。
  
  "用生成结果判断是否翻转"的版本作为消融，放到 E2。
- **B3 的一致性项：** 同一 z 两个呈现在 `Answer:` 后 yes/no 两个 token 概率上的对称 KL，λ ∈ {0.1, 0.5}，按验证集挑选。
- **评估：** 2×2 四格（每格 ≥ 200）+ GSM8K 子集（500 题，开思维链）。
- **pilot 要交回的东西：**
  - 每个条件的训练耗时、评估耗时、峰值显存；
  - 四格准确率；
  - 按实测耗时推算全量（2000 z × 5 条件 × 3 种子）总共需要多少小时。

**验收：** loss 正常下降；B1 / B2 在 seen/canonical 格上不比 B0 差；耗时推算合理。

### Step 5：E1 全量

- 如果推算总耗时超过 48 小时：先把种子减到 2 个，或训练集减到 1000 z，并告诉用户；也可以先只跑 B2、B3、Ours 三个关键条件（B0 不用训练，B1 最后补）。
- 训练任务要能断点续跑：每个"条件 × 种子"单独保存 adapter 和结果 json，跑完一个就 commit 一次结果文件。worker 中断后可以从没完成的组合继续。
- 结果统一放到 `experiments/e1/results.csv`，列为：condition、seed、cell、acc、n、SS、SF、Δ_T、交互项、GSM8K。

### Step 6：E1 判定（判据见第 4 节）

- leader 自己从 `results.csv` 复算：跨种子求均值，按 latent task 做配对 bootstrap，算 Ours − max(B2, B3) 在 unseen/shifted 格上的差值及置信区间。
- **通过：** 进入 Step 7。
- **不通过：** 整理 C1 的诊断结果，也就是 B2 / B3 各自改善了 SS 还是 Δ_T。然后用 AskQuestion 问用户，选项：
  1. 转为分析论文，以 C1 为主；
  2. 修改 BGM 再试一轮，比如换成生成式翻转挖掘，或者加课程学习；
  3. 换方向。
- 不论结论如何，都更新第 3、5 节。

### Step 7：E2 消融 + E3 边界

- E2：去掉负担分数（只用 NLL）、去掉错误引导（只用负担分数 B）、去掉一致性项，以及生成式挖掘版本。每个消融 2–3 个种子。
- E3：在任务族 B（算术计算图）上重复 B2 / B3 / Ours 三个条件。第一步先在任务族 B 上跑一遍 E0 式的健全性检查。
- 时间不够时，E3 可以缩成 1 个种子，在论文里如实写明。

### Step 8：写作（按顺序，每步一个 worker 任务，可以连续 resume 同一个 worker）

| 顺序 | 技能 | 输入 | 产出 | leader 验收要点 |
|---|---|---|---|---|
| 1 | intro-drafter | docs/03 骨架、E1 结论 | paper/main_en.tex 引言节 | 贡献和 C1/C2 一致；不夸大；引用都在 refs.bib 里且已核实 |
| 2 | paper-writer | 骨架、experiments/ 下的结果 | 方法、实验、相关工作、结论各节 | 每个数字都能在 experiments/ 中找到出处；没跑完的部分标"待补" |
| 3 | figure-designer（+ drawio-reconstruction 画 Fig.1） | 结果 csv | figures/*.pdf 和绘图脚本 | 图由脚本从 csv 生成，可复现 |
| 4 | paper-polish | 全文 | 润色后的 tex | 科学含义不变 |
| 5 | 中文稿 | 英文定稿 | paper/main_zh.tex | 用 xelatex 编译 |
| 6 | pre-submission-reviewer | PDF + tex | docs/08_review.md | leader 逐条看 CRITICAL / MAJOR，派 worker 修，修完再审一次 |

- 两份稿都要编译通过：英文用 pdflatex 或 xelatex，中文用 xelatex + ctex。PDF 放在 paper/ 下。

### Step 9：清理与收尾

1. 按第 6 节执行清理，`git rm` 后单独 commit。
2. 写 README.md：环境安装、数据生成、E0/E1 复现命令、各结果文件含义。
3. 跑一遍 pytest 和两份 tex 的编译，确认清理没有删掉依赖文件。
4. 向用户做最终汇报：论文 PDF 路径、核心结论和数字、仍标"待补"的部分、建议的下一步（例如借算力补 7B 实验）。

### 7.1 等 worker-grok 汇报 E0 时，leader 要审核什么

1. **docs/01_literature_review.md**
   - 抽查 3–5 条引用（WebSearch 核对标题、作者、会议）；
   - 确认 He et al.、MEND、BCT、F2C、Pearl、PA-GRPO 确实存在，而且做的就是 worker 说的内容；
   - 确认综述是否证实"现有工作没有区分同图稳定性与新图迁移"。这是 C1 能否成立的关键。
2. **docs/02_idea_evaluation.md**
   - 看判定结果。如果是 Reject and Pivot，先别继续，问用户。
3. **E0 数字**
   - 先看是否高于多数类基线，这是旧 pilot 栽过的坑：要看输出标签分布，不要只看准确率；
   - 答案抽取失败率要低于 5%，否则先修 prompt；
   - SF 的分母是"原题答对"的样本，这部分样本数至少要 100；
   - proof-aligned 应好于 random / reversed，这是对 Chen et al. 的复现；如果方向反了，先查数据生成有没有 bug。
4. **代码**
   - 抽查 tests 是否真的覆盖了"等价呈现不改变求解器答案"和"稳定性池与迁移池无重叠"（图哈希隔离）。

### 7.2 E0 的决策分支

- **达标：** 派 E1。先做一个小规模试跑（1 个种子、1/4 预算）估算耗时，再决定全量；1.5B 跑不完时，种子可以先 2 个，后补。
- **不达标但接近**（SF 在 5–10% 之间）：允许增加深度或干扰前提再试；仍不行就换 3B。
- **3 轮仍不达标：** 向用户汇报。可选方案：
  1. 换更大模型，通过 API 只做评估、不训练；
  2. 转向任务族 B；
  3. 重新考虑方向。

### 7.3 E1 → E3 的派发要点

- 给 worker 的 E1 prompt 必须写明：等预算的具体数值（token 数、步数、学习率、batch、rank=16、alpha、dropout、target modules）、3 个种子、评估集路径、判据。
- BGM 的实现细节（leader 建议，worker 可以提出修改）：
  - 每个 z 生成 K=8 个等价呈现；
  - 负担分数 B = D_rel、回读次数、F_active 标准化后的等权和；
  - 挖掘规则：从当前模型答错（或翻转）的呈现中，按 B 从高到低取 top-k；没有翻转的 z 退化为取 B 最高的呈现；
  - 每个 epoch 用最新 checkpoint 重新挖掘一次（离线，两轮即可）。
- B3 的一致性项：同一 z 的两个呈现，在最终答案 token 分布上的对称 KL，权重 λ 在 {0.1, 0.5} 中，用验证集选。
- E1 结果出来后：
  - 判据通过：进入 E2 / E3，并开始写作流程；
  - 判据不通过：把 C1 的诊断结果整理出来，问用户是否转为分析论文（选项：转分析论文 / 改方法再试一轮 / 换方向）。

### 7.4 写作阶段（E1 完成后）

1. 按顺序派发：intro-drafter → paper-writer → figure-designer（matplotlib 画真实结果，Figure 1 可以用 drawio-reconstruction）→ paper-polish → pre-submission-reviewer。
2. 写作可以单独派一个 worker，prompt 里给出：docs/03 骨架、experiments/ 下的结果文件路径、"严禁编造"铁律、两份 tex 的路径。
3. 审稿报告写到 docs/08_review.md。leader 读完 CRITICAL / MAJOR 级问题，再派 worker 修改。
4. 全部完成后执行第 6 节的清理，然后更新 README.md（复现步骤）。

### 7.5 向用户汇报的节奏

- 每个关卡（E0、E1、写作完成、清理完成）汇报一次：先说结论，再说数字，最后说下一步。
- 需要用户决策的情况（方向调整、转分析论文、需要外部算力），用 AskQuestion 给出选项。

## 8. 关键文件索引

| 路径 | 内容 |
|---|---|
| `docs/HANDOFF_LEADER.md` | 本文件 |
| `docs/01_literature_review.md` | 文献综述（worker-grok 在写） |
| `docs/02_idea_evaluation.md` | idea 评估（待写） |
| `docs/03_paper_skeleton.md` | 论文骨架（待写） |
| `src/structrobust/` | 新代码 |
| `experiments/e0/` | E0 小结果文件 |
| `build/` | 大文件和日志（已 gitignore） |
| `experiments/e1/results.csv` | E1 汇总结果（待产生） |
| 用户给的流程图片 | `C:\Users\Qiutian\.cursor\projects\d-study\assets\c__Users_Qiutian_AppData_Roaming_Cursor_User_workspaceStorage_5ad5cf98899bf733fa768161aa04f845_images_c2317d9c85d59e6cb56a2483db4ef531-56ac816c-d37c-4dd1-94ca-ef887065a959.jpg` |

## 9. 问题处理手册

原则：先找证据（日志、文件、复算），再改；同一个办法失败两次就换思路；凡是涉及方向变化的问题都问用户。

### 9.1 实验与代码

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 准确率约等于多数类比例 / 模型几乎全答一种 | 题太难；没开思维链；答案抽取错；标签不平衡 | 看 20 条原始输出；确认 max_new_tokens ≥ 512，标签 1:1；降低深度；改提示（要求最后一行写 `Answer: yes/no`） |
| 三种呈现的准确率几乎一样，SF≈0 | 题太简单（全对）；呈现变换没有真正打乱证明顺序；用到的前提太少 | 检查 reversed 呈现的回读次数是否大于 0；加深度、加干扰前提；只在准确率 0.55–0.90 的档位统计 |
| proof-aligned 反而最差 | 呈现生成有 bug（顺序标签弄反了）；求解器轨迹顺序有误 | 打印同一个 z 的三种呈现人工比对；跑测试；写一条单元测试固定住正确行为 |
| 答案抽取失败率 > 5% | 输出格式不稳定；被截断 | 看失败样本；放宽正则（大小写、True/False 同义词）；把 max_new_tokens 调到 768；截断的样本记为错误，并单独报告比例 |
| CUDA OOM | batch 太大；序列太长 | 减小 batch，加梯度累积；开 gradient checkpointing；推理时用 4-bit；限制 max_new_tokens；同一时间只跑一个 GPU 任务 |
| bitsandbytes 在 Windows 上报错 | 和 CUDA 版本不兼容 | 1.5B 改成 bf16 LoRA，不量化（权重约 3GB，8GB 显存够用）；3B 必须用 4-bit 时，试 `pip install -U bitsandbytes`；仍不行就把 3B 只用于推理评估 |
| loss 是 NaN 或发散 | lr 太大；fp16 溢出 | lr 降到 1e-4；用 bf16；LoRA 参数用 fp32；加梯度裁剪 1.0 |
| 训练后 GSM8K 掉分超过 1 个点 | 在合成数据上过拟合 | 混入 10%–20% 的通用指令数据或 GSM8K 训练集；lr 降一半；所有条件做同样处理以保持公平 |
| 微调后的模型比 B0 还差 | 格式没对齐；推理过程模板和评估提示不一致 | 训练和评估用同一个对话模板；检查 label mask 是否只对回答部分算 loss |
| 一个任务跑太久 | 生成太慢 | 先做小规模计时；批量生成；按长度分桶；降样本量时保证每格 ≥ 200 |
| HF 下载失败 | 网络问题 | `$env:HF_ENDPOINT="https://hf-mirror.com"`；Shell 加 full_network 权限 |
| 磁盘不够 | checkpoint 太多 | 只保存 LoRA adapter；删除中间 checkpoint；大文件都放在 build/ 下 |

### 9.2 写作与引用

| 现象 | 处理 |
|---|---|
| 某条引用核实不了 | 删除这条引用，相应论点改成保守表述；不能"凭印象"保留 |
| 结果不支持原定的叙事 | 改叙事，不改数据。负结果如实写，必要时问用户是否转分析论文 |
| LaTeX 中文编译失败 | 用 xelatex + ctex；缺宏包用 TinyTeX 的 `tlmgr install <pkg>` |
| 审稿报告里有 CRITICAL 问题 | 先修 CRITICAL，再修 MAJOR；每修一轮，重新跑一次 pre-submission-reviewer |

### 9.3 协作与流程

| 现象 | 处理 |
|---|---|
| worker 报告的数字和文件对不上，或者没有给出文件路径 | 要求它给出产生数字的文件和命令，由 leader 自己复算；复算不一致的数字一律不用 |
| worker 擅自改了研究方向，或者删了文件 | 用 `git diff` 或 `git log` 查看；必要时 `git checkout <commit> -- <path>` 恢复；重申铁律 |
| resume 报 "Subagent not found" | 那是别的会话启动的 agent，新启一个 worker，prompt 让它先读本文件 |
| resume 报 "still running" | 它还在跑，等通知；确实需要改它的任务时才用 interrupt |
| 很久没有动静 | 看 build/logs 最新日志的修改时间，以及 `Get-Process python`；进程已退出但没有汇报，就 resume 让它汇报 |
| Opus 额度快用完 | 立即把最新状态写进第 5 节，然后告诉用户：新开会话，选 Grok 4.7，让它读本文件并按 Step 0 接手 |
| 用户提出新要求 | 先记进第 5 节，再判断会影响哪些 Step，把影响写进对应 Step |

## 10. 中期答辩（紧急，由下一任 leader 负责）

### 10.1 要做什么

用户很快要参加中期答辩，需要两样东西：

1. **填好中期答辩表格。** 表格已经放在项目根目录 `D:\study\科研拔尖项目\` 下，具体文件名由子 agent 自己找。Opus leader 没有看过这份表。
2. **做一份答辩 PPT。**

### 10.2 leader 的硬性要求

- **必须新开一个独立的子 agent** 专门处理中期答辩：`subagent_type: generalPurpose`，`model: grok-4.7-high-fast`，`run_in_background: true`。不要交给正在做实验的 worker，也不要 leader 自己做。
- **开工前先用 AskQuestion 问用户四件事：**
  1. 答辩日期；
  2. 汇报时长和 PPT 页数要求；
  3. 有没有学校或学院的 PPT 模板；
  4. 表格里有没有已经确定的信息（姓名、学号、导师、项目名称、起止时间、经费等）。

  把答案写进 10.4 的 prompt 再派发。
- **优先级：** 答辩日期 ≤ 7 天时，中期答辩优先于 E1 及之后的所有工作；但实验 worker 可以继续并行跑 GPU 任务，答辩 agent 只用 CPU。
- **只用 WPS，不用 Microsoft Word / PowerPoint。** 表格用 `kwps.Application` 直接打开 `.doc` 副本填写并保存；不要调用 `Word.Application`。
- **2026-10-12 14:42 起，PPT 不再由 leader 负责。** 用户明确收回。不要再派 agent 改 `midterm/中期答辩.pptx`，除非用户重新要求。leader 只核对表格。
- **和清理计划的冲突：** 第 6 节的清理**不得删除**以下内容：
  - 中期答辩表格原件；
  - `midterm/` 目录；
  - 答辩 agent 引用到的任何旧文件。

  答辩 agent 需要的旧素材（旧 figures、旧文献综述）要先复制到 `midterm/assets/`。清理工作要等答辩材料定稿之后再做。

### 10.3 内容方案（leader 的设计，答辩 agent 照此执行）

**叙事主线：** "发现问题 → 调研与重新定位 → 新方案 → 初步结果 → 后续计划"。

旧框架的工作不要藏起来。它是真实的前期工作：
- 文献调研；
- 稳定性与迁移的形式化定义；
- 第一次 pilot，以及从中发现的问题（准确率等于多数类基线、没开思维链）。

这些恰好说明我们做了严谨的自我审查，并据此重新定位。

**中期报告应覆盖的内容（表格和 PPT 共用）：**

1. **研究背景与问题。** 大模型推理对等价的结构变化很敏感：前提顺序一变，准确率可以下降 30% 以上（Chen et al., ICML 2024）。高分不等于结构上稳健的推理能力。
2. **研究目标。** 把"同图呈现稳定性"和"新结构迁移"分开诊断，并研究能否通过训练提升结构鲁棒推理。
3. **已完成的工作：**
   - 系统文献调研（docs/01），定位最接近的工作：Premise Order Matters、DyVal、GSM-∞、iGSM，以及重排增强 / 一致性训练类方法；
   - 形式化框架：z=(G,q,y) 与呈现 r 分离，SS/SF、Δ_T、2×2 分解；
   - 第一轮 pilot 及其教训；
   - 方案重新定位：从 benchmark 转向技术论文，方法从 v1 改到 v2（理由见第 3.1 节）；
   - 新代码框架（求解器验证的数据生成器、负担特征、配对 bootstrap、测试）；
   - E0 结果（如果已经产生，**只用真实数字**；没产生就写"正在运行"）。
4. **技术路线。** C1 诊断 + C2 BGM 方法（第 3.2 节），E0–E3 实验设计（第 4 节），并配一张流程图。
5. **创新点。** 用保守措辞：
   - 首次在训练干预下区分稳定性和迁移（**需以 docs/01 的核实结果为准**；没核实就写"据我们调研"）；
   - 负担引导挖掘（检验"均匀随机重排最好"的结论）。
6. **存在的问题与风险：**
   - 算力只有 8GB 笔记本 GPU；
   - BGM 可能打不过随机重排（退路：以 C1 为主写分析论文）；
   - 合成任务的外部效度。
7. **后续计划与时间表。** 按第 7 节 Step 4–9 排出周计划：E1 → E2/E3 → 写作 → 投稿目标。时间按答辩日期和用户给的截止时间倒推。
8. **预期成果。** 一篇技术论文（英文主稿 + 中文稿），以及开源代码。

**PPT 结构建议（约 12–15 页，按用户给的时长调整）：**

封面 → 研究背景（一张"重排导致翻转"的例子图）→ 研究问题与目标 → 相关工作定位（一张对比表）→ 前期工作与反思 → 形式化框架（2×2 图）→ 方法 BGM（流程图）→ 实验设计 → 当前结果 / 进度 → 风险与对策 → 时间表（甘特图）→ 预期成果 → 致谢 / Q&A

**另外要产出：**
- 每页的讲稿（写在 PPT 备注里，并另存一份 md）；
- 一份"评委可能提问及回答"清单，10–15 条，例如："和 Premise Order Matters 有什么区别？""只用 1.5B 模型结论能推广吗？""如果方法无效怎么办？"

### 10.4 派发 prompt（leader 把【】里的内容填好后原样派发）

```
我是 leader，你是专门负责中期答辩材料的 agent（模型 grok-4.7-high-fast）。
项目：D:\study\科研拔尖项目。

【用户提供的信息：答辩日期＝…；时长＝…分钟；页数要求＝…；模板＝…（路径或"无"）；表格已知信息＝…】

第一步：阅读背景
- 完整阅读 docs/HANDOFF_LEADER.md 的第 1、2、3、4、5、10 节。第 10.3 节是内容方案，照它执行。
- 阅读 docs/01_literature_review.md、docs/02_idea_evaluation.md、docs/03_paper_skeleton.md（存在的话），以及 experiments/ 下已有的真实结果。
- 旧框架材料在 research/、manuscript/、figures/、experiments/mvp_pilot_report.md，也可以从 git tag pre-restart 中查看。

第二步：找到并填写表格
- 在项目根目录找到中期答辩表格（可能是 .docx/.doc/.xlsx/.pdf）。
- 不要修改原件。复制一份到 midterm/ 下填写，文件名加"_已填"。
- 工具：表格用 WPS 文字 COM（`kwps.Application`）直接打开 `.doc` 副本填写并保存，不要转成 docx，不要调用 Microsoft Word。`.xlsx` 如遇到再用 WPS 表格 `ket.Application`。依赖都装在 .venv 里。
- 尽量保留原表格的格式、字体和表格结构。
- 字数限制严格遵守；表中如有字数要求，在汇报里报告每栏字数。
- 个人信息只用上面【】里给的；不知道的栏位留空并标黄，在汇报里列出，不要编。

第三步：做 PPT
- 按第 10.3 节的结构，用 python-pptx 生成 midterm/中期答辩.pptx。有模板就基于模板。
- 中文字体用微软雅黑，每页文字不超过 6 行，以图为主。
- 图：优先复用 figures/ 里已有的图，或者用 matplotlib / drawio 生成新的示意图（流程图、2×2 图、甘特图），源文件放在 midterm/assets/。
- 每页的讲稿写在备注里，并另存一份 midterm/讲稿.md；讲稿时长按【时长】控制。
- 写一份 midterm/评委问答.md，10–15 条。

第四步：检查
- 把 PPT 渲染成图片逐页检查：用 WPS 演示 COM（`kwpp.Application`）导出 PNG。不要调用 Microsoft PowerPoint。重点看文字溢出、重叠、字体缺失。发现问题就修。
- 核对 PPT、表格、讲稿三者的内容一致。

铁律
- 所有实验数字必须来自 experiments/ 下的真实文件，并注明来源。没有结果就写"进行中"，严禁编造或"预估"成已完成结果。
- 引用只用 docs/01 中已核实的文献。
- 不要删除或修改项目中的任何已有文件，只在 midterm/ 下新建文件。
- 不要启动 GPU 任务（实验 worker 可能正在用 GPU）。
- 不要 push。完成后做一次 commit："midterm materials"。
- Shell 命令一律带 required_permissions: ["all"]。

汇报格式
1. 产出文件清单（路径）；
2. 表格中留空 / 待用户确认的栏位；
3. PPT 页数和每页标题；
4. 引用了哪些实验数字，各自来源哪个文件；
5. 渲染检查中发现并修复的问题；
6. 需要用户补充或决定的事项。
```

### 10.5 leader 验收中期材料

- 打开表格副本，核对：栏位齐全；没有编造的个人信息或数字；字数符合要求。
- 浏览 PPT 渲染图，抽查 3–4 页：版面是否溢出；数字是否和 experiments/ 一致。
- 抽查评委问答中"和 Chen et al. 的区别""方法无效怎么办"这两条的回答是否站得住。
- 把留空栏位和需要决定的事项汇总后告诉用户。用户修改意见回来后，resume 答辩 agent 修改。
- E0 / E1 有新结果时，resume 答辩 agent 更新"当前结果"页和表格对应的栏位。
- 在第 5 节记录：答辩 agent 的 id、产出路径、状态。

## 11. 给 worker 的 prompt 模板

```
我是 leader，你是 worker（模型 grok-4.7-high-fast）。
项目：D:\study\科研拔尖项目。先完整阅读 docs/HANDOFF_LEADER.md 的第 0、3、4 节和第 9 节，了解背景、铁律和环境坑。

本轮任务：<从第 7 节对应的 Step 复制，写清楚输入、输出路径、默认配置>
验收标准：<从对应 Step 复制>
停止条件：<做完什么就停下汇报；遇到哪些情况必须立即停下汇报>
禁止：编造数字；push；删除旧文件（除非本轮任务就是清理）；擅自改变研究方向。
Shell 命令一律带 required_permissions: ["all"]；运行 Python 前设 $env:PYTHONPATH="src"，HF_HOME 指向项目内 .cache/huggingface。
长任务放后台，日志写到 build/logs/，结果做到可以断点续跑。

汇报格式：
1. 做了什么，产出了哪些文件（给出路径）；
2. 真实数字（附置信区间），以及产生这些数字的命令和文件；
3. 测试结果，commit hash；
4. 遇到的问题和你的处理；
5. 需要 leader 决策的事项。
```
