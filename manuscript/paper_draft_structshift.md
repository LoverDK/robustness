# Structural Robustness and Reasoning Generalization in Large Language Models

## Abstract

Large language models (LLMs) achieve strong performance on many multi-step reasoning benchmarks, but conventional random splits often preserve the reasoning structures seen during training. This paper studies whether structural robustness affects reasoning generalization. StructShift is used as a solver-validated measurement module for paired structural transformations, while an error-guided structural adversarial LoRA protocol remains a secondary intervention. The study focuses on premise and step order, reasoning depth and branching, unseen combinations of familiar operators, and controlled auxiliary questions. The benchmark construction, evaluation results, and training comparisons are left to the empirical study; no numerical result is claimed in this draft.

## 1 Introduction

Large language models (LLMs) increasingly solve multi-step mathematical, logical, and knowledge-intensive questions by decomposing a problem into intermediate reasoning steps. Yet a correct answer on a conventional random split does not necessarily indicate that a model has learned a reasoning structure that transfers beyond the training presentation. Work on compositional generalization has separated systematicity, productivity, and related forms of structural transfer [1]. In language understanding, compositional splits expose failures that are hidden by random splits [2,3], and scaling alone does not reliably remove the gap [4]. Recent evaluations of mathematical reasoning reach a similar conclusion from a different direction: controlled instantiations and additional clauses can change model behavior even when the underlying task is intended to remain comparable [5,6]. These observations motivate a more specific question: can a model preserve the correct answer when the organization of a reasoning problem changes while the solution remains the same?

Existing evaluations do not yet provide a consistent answer because they perturb different objects under the same broad label of robustness. Compositional-generalization benchmarks hold out new structures, whereas mathematical stress tests vary instantiations or insert distracting clauses [2,5,6]. Other work studies irrelevant context [7], information position in long contexts [8], or the reliability of an already generated chain of thought [9,10]. These settings are related, but they do not share a single threat model. A perturbation may change a task's numerical distribution, its token length, its semantic solution, its context position, or only its rationale; treating all such changes as structural reasoning failures obscures the mechanism being measured. A random instance split can also test whether a model recognizes familiar components without testing whether it can combine them in an unseen way. A benchmark that measures structural robustness therefore needs to preserve the answer by construction, separate structure from nuisance variables, and evaluate held-out combinations rather than only held-out instances.

We study this setting as reasoning under label-preserving structural distribution shift. Each item is generated from a symbolic reasoning graph, rendered as natural language, and paired with an answer verified by a solver. A structural transformation is admitted only when the transformed graph yields the same target answer. The benchmark varies the organization of the task rather than its solution: legal premise reordering, changes in depth or branching, held-out combinations of familiar reasoning operators, and the insertion of a nonessential auxiliary question at a controlled position. This formulation connects the compositionality gap observed in semantic parsing and multi-hop reasoning [4,11] with the distractor sensitivity observed in mathematical and reading-comprehension evaluations [6,7,12]. It also makes the central error measurable. If (R_S(f)) denotes the error rate of a model on a structure group (S), we define a structural gap between an out-of-distribution group and an in-distribution group as (R_{OOD}(f)-R_{IID}(f)). The goal is to create an evaluation setting in which competing explanations can be tested rather than to attribute every performance change to a single internal failure.

Three challenges make this setting nontrivial. First, a natural-language rewrite can silently change the answer or the dependencies in the reasoning graph; surface paraphrases are therefore insufficient as a source of ground truth. Second, structural changes are entangled with nuisance factors. A deeper graph may be longer, a reordered premise may move evidence to a different context position, and a side question may share entities or vocabulary with the main question. Without matched controls, a measured drop cannot be attributed to structure. Third, a benchmark should support diagnosis and improvement rather than only ranking. Generating many variants is easy, but selecting variants that expose a model-specific structural failure while retaining clean-task performance requires a controlled training signal. These challenges are distinct from noisy-rationale robustness, where the object being changed is the intermediate explanation [9,13], and from semantic-changing adversarial attacks, where the label itself may no longer be invariant.

To address these challenges, we build a solver-validated measurement module, called StructShift, and an accompanying diagnostic intervention protocol. The construction pipeline first samples a symbolic reasoning graph, renders a clean natural-language instance, applies a restricted transformation grammar, and re-solves every candidate before release. The evaluation protocol separates IID, Surface-OOD, Structure-OOD, and Composition-OOD splits, and reports clean accuracy, robust accuracy, structural gap, and side-question interference. The companion method uses error-guided structural adversarial examples with LoRA or QLoRA, building on parameter-efficient adaptation [14,15] and targeted adversarial training [16] while preserving the label-preserving constraint. Its purpose is diagnostic: ordinary LoRA, random structure augmentation, targeted mining without structural validation, and full fine-tuning provide controls for determining whether any gain comes from structural selection, additional data, or model capacity.

Our contributions are fourfold. First, we formulate reasoning under label-preserving structural distribution shift and distinguish task-structure perturbations from context noise, rationale interventions, and semantic-changing attacks. Second, we specify a reproducible graph-to-text construction pipeline with solver validation, transformation metadata, nuisance controls, and held-out structural-composition splits. Third, we define a fine-grained evaluation framework for structural gap, worst-group behavior, and side-question interference, designed to reveal which structures current models fail to transfer. Fourth, we provide an error-guided structural LoRA protocol and a matched baseline suite for testing whether parameter-efficient adaptation can reduce the gap without sacrificing clean accuracy. The empirical claims associated with these contributions are established only after the benchmark construction and model evaluation are completed.

## 2 Related Work

### 2.1 Compositional generalization

Compositional generalization studies whether models can apply familiar components in novel combinations. Hupkes et al. decomposed compositionality into several properties that distinguish different forms of generalization [1]. Oren et al. and Herzig and Berant constructed semantic-parsing settings in which test examples contain structures not observed in training [2,3]. Qiu et al. evaluated the effect of model scale and training regime on compositional OOD and found that larger models or standard fine-tuning do not automatically resolve the problem [4]. Press et al. connected the compositionality gap to the distinction between single-hop and multi-hop question answering and examined explicit follow-up questions as a way to narrow that gap [11].

StructShift borrows the idea of holding out structure while changing its target object. Instead of mapping language to executable semantic parses, it evaluates final-answer reasoning under solver-validated graph transformations. The distinction is important because a benchmark can hold out surface templates, semantic parses, or reasoning graphs, and these choices test different capabilities.

### 2.2 Robust reasoning and distractors

GSM-Symbolic generates controlled mathematical variants and examines performance under changes to problem instantiations and clauses [5]. GSM-Plus provides a broader robustness evaluation for mathematical problem solving [6]. Shi et al. study the effect of irrelevant context on LLMs [7], while Jia and Liang establish an earlier precedent for adding distracting sentences to reading-comprehension inputs [12]. Liu et al. show that the position of information in long contexts affects model behavior [8]. These works motivate the use of distractor and position controls, but they do not all preserve the same variables or evaluate unseen structural compositions. StructShift treats irrelevant auxiliary questions as one controlled transformation family and reports their effect separately from depth, branching, and operator composition.

### 2.3 Chain-of-thought robustness and faithfulness

Noisy-rationale work changes the reasoning trace supplied to or produced by a model rather than the input task graph [9]. Paul et al. study whether intermediate reasoning steps causally support the final answer [10], and later work examines direct interventions on generated chains of thought [13]. These studies show why answer correctness and explanation faithfulness should not be conflated. StructShift therefore keeps rationale faithfulness as an optional diagnostic and defines its primary task around the final answer under input-task structural transformations.

### 2.4 Adversarial training and parameter-efficient adaptation

Targeted adversarial training prioritizes examples that expose a model's current errors [16], and self-play provides a related adversarial training precedent for reasoning models [17]. LoRA adapts a frozen model through low-rank updates [14], while QLoRA combines low-rank adaptation with quantization for efficient fine-tuning [15]. StructShift combines these ideas only as an optional companion protocol. The key distinction is that the adversary is restricted to solver-valid, label-preserving structural transformations.

## 3 Task Definition and Design Goals

### 3.1 Problem formulation

Each reasoning item is represented as

\[
x=(G,r,q), \qquad y=\operatorname{Solve}(G,q),
\]

where (G) is a symbolic reasoning graph, (r) is its natural-language realization, (q) is the target question, and (y) is the solver-verified answer. A structural transformation (T_\delta) is admitted to the benchmark threat model only if it preserves the answer:

\[
\mathcal{T}_{LP}=
\{T_\delta : \operatorname{Solve}(T_\delta(G),q)=y\}.
\]

The transformation budget (delta) records which graph or presentation properties may change. The benchmark does not assume that every valid transformation is semantically or linguistically harmless; it validates the target answer and separately checks the generated text for contradictions and ambiguity.

### 3.2 Structural axes

StructShift initially targets four axes:

1. **Order sensitivity:** reorder premises or steps when graph dependencies permit the change.
2. **Structural complexity:** vary depth, branching, or nesting while retaining a solver-verifiable answer.
3. **Composition shift:** hold out combinations of reasoning operators while exposing the individual operators during training.
4. **Side-question interference:** insert a nonessential auxiliary question before, inside, or after the main reasoning context.

The final release should use only transformations that survive the pilot's semantic and difficulty checks.

### 3.3 Design goals

The benchmark follows five goals. G1 is coverage across task families, structural axes, and difficulty. G2 is fine-grained diagnosis through graph metadata and group-wise reporting. G3 is scalability through symbolic generation, deterministic solving, and a versioned transformation grammar. G4 is quality through solver checks, duplicate detection, and expert spot checks. G5 is contamination resistance through hidden evaluation seeds, fresh variants, and an explicit version manifest.

## 4 Construction Pipeline

StructShift combines reverse synthesis with controlled injection. The generator first fixes a valid symbolic answer and reasoning graph, then renders the graph and injects a permitted structural change. The same process yields both clean and transformed items, making paired evaluation possible.

### 4.1 Task-family selection

The pilot should begin with one solver-friendly symbolic family and one natural-language reasoning family. Each family must specify its graph schema, operator set, answer space, legal transformations, and exclusion rules. The benchmark should not expand to additional families until the first family yields stable validation and measurable structure-group differences.

### 4.2 Symbolic seed generation

The generator samples reasoning graphs with controlled depth, branching, operator sequence, and nesting. Each graph is solved before rendering, and the solver trace is retained as metadata. Invalid or non-unique graphs are rejected. The generator should support fixed seeds for reproducibility and hidden seeds for contamination-resistant evaluation.

### 4.3 Natural-language rendering

Each graph is rendered into a clean problem statement using controlled templates and lexical variation. Rendering does not determine the answer; it only realizes the already-solved graph. Candidate realizations are checked for required fields, malformed references, and consistency with the graph. A human spot-check sample should review grammaticality and whether the textual statement expresses the intended relation.

### 4.4 Structural transformation

The transformation grammar applies one controlled change at a time for diagnostic tests and composes multiple changes for Composition-OOD tests. Each candidate records its source graph, transform sequence, perturbation budget, and changed structural attributes. Side-question candidates record count, position, entity overlap, and semantic similarity with the main question.

### 4.5 Automatic validation and splitting

The system re-solves each candidate and retains it only when the target answer is preserved. It then performs duplicate detection and matches nuisance statistics across comparison groups. Splits are constructed by structure rather than by random instance assignment:

* **IID:** same graph distribution, new instances;
* **Surface-OOD:** new wording or entities with the same graph;
* **Structure-OOD:** a held-out depth, branch pattern, order pattern, or side-question position;
* **Composition-OOD:** familiar primitive operators in a held-out combination.

### 4.6 Quality-control protocol

The final paper should report automatic rejection categories, answer-preservation checks, duplicate rates, and the size of the human-reviewed sample. If human judgments are used to assess grammar or side-question non-necessity, the paper should report an agreement statistic and an adjudication rule. A second LLM may be used for cross-validation, but it cannot replace symbolic answer verification.

### 4.7 Planned dataset profile

The release should include counts by split, task family, transform, difficulty tier, graph depth, branching, token length, number range, and side-question position. These quantities are intentionally left unfilled until the pilot is run.

## 5 Evaluation Framework

### 5.1 Metrics

For structure group (S), let (R_S(f)) denote the error rate and (operatorname{Acc}_S(f)) denote accuracy. We report:

\[
\operatorname{StructuralGap}(f)=R_{OOD}(f)-R_{IID}(f),
\]

and, for side-question interference,

\[
\operatorname{InterferenceCost}(f)=
\operatorname{Acc}_{clean}(f)-
\operatorname{Acc}_{side}(f).
\]

The primary evaluation includes clean accuracy, robust accuracy by transformation group, worst-group accuracy, structural gap, interference cost, and confidence intervals across random seeds when model scores are available. Calibration and rationale-faithfulness metrics are secondary analyses and must be validated separately.

### 5.2 Nuisance controls

Every structure comparison should report or match token length, numerical range, lexical template, entity names, and context position where those variables are not themselves the target axis. Side-question experiments should vary position independently from count and semantic similarity. Depth experiments should report whether the additional steps also increase textual length or arithmetic magnitude.

### 5.3 Contamination mitigation

The benchmark should generate evaluation variants from held-out seeds and release generator code with a versioned transformation grammar. Public training data should not contain the final evaluation variants. The release should include a data card describing provenance, intended use, limitations, seed policy, and known contamination risks. Later benchmark versions should add fresh variants rather than only expanding a fixed static test set.

## 6 Companion Method: Structural Adversarial LoRA

The companion method tests whether StructShift provides an actionable training signal. Candidate transformations are generated inside (\mathcal{T}_{LP}), scored by the current model, and retained when they produce high loss or an incorrect answer while the solver confirms the label. The adapter is trained on a matched clean and hard-example mixture.

The method should be compared with a base model, ordinary LoRA, random structure augmentation plus LoRA, targeted mining without the structural constraint, and full fine-tuning under matched token and step budgets. Rank, target modules, learning rate, quantization, training steps, candidate count, and random seeds must be recorded. Ablations should remove hard mining, clean replay, individual transformation families, or solver validation.

The method is not assumed to improve every condition. Its role is to test whether a structure-aware training signal improves held-out structural composition beyond what can be explained by additional data or parameter capacity.

## 7 Experiments

### 7.1 Research questions

* **RQ1:** Which label-preserving structural transformations produce a measurable reasoning gap after nuisance controls?
* **RQ2:** Which structure groups and model conditions distinguish stronger from weaker structural transfer?
* **RQ3:** Does error-guided structural LoRA improve Structure-OOD and Composition-OOD performance while preserving clean accuracy?

### 7.2 Models and prompting

The final study should include multiple open-weight model sizes or architectures and, where access permits, a closed model evaluation. The exact model list and prompt formats should be fixed before the test set is released. Prompting variants should be treated as experimental factors rather than silently mixed into the main comparison.

### 7.3 Overall comparison

| Model | IID Acc. | Surface-OOD Acc. | Structure-OOD Acc. | Composition-OOD Acc. | Worst-group Acc. | Structural Gap | Interference Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base model A | -- | -- | -- | -- | -- | -- | -- |
| Base model B | -- | -- | -- | -- | -- | -- | -- |
| Ordinary LoRA | -- | -- | -- | -- | -- | -- | -- |
| Random augmentation + LoRA | -- | -- | -- | -- | -- | -- | -- |
| Structural adversarial LoRA | -- | -- | -- | -- | -- | -- | -- |
| Full fine-tuning | -- | -- | -- | -- | -- | -- | -- |

### 7.4 Fine-grained analysis

Report performance by order, depth/branching, composition, side-question position, and severity. Include error examples only after the pilot has produced verified pairs. Distinguish a wrong answer with a preserved graph from an invalid generated item or a solver disagreement.

### 7.5 Expected Finding format

The final paper should use data-backed statements such as **Finding 1: [observed transformation] yields the largest gap after [controls].** Planned hypotheses should not be written as findings. If random augmentation matches structural hard mining, the paper should report that the additional structure data, rather than the selection rule, may explain the improvement.

## 8 Discussion and Limitations

StructShift is designed to isolate a specific evaluation dimension, not to measure all forms of reasoning robustness. Symbolic generation improves answer control but can reduce natural-language diversity. Structural complexity is correlated with difficulty and length, so nuisance controls and matched comparisons are necessary. A model can improve on a held-out transform while still relying on other shortcuts. Conversely, a failed transformation may reflect an invalid rendering rather than a model limitation. The benchmark should therefore release graph metadata, validation scripts, rejection logs, and representative paired examples.

The companion LoRA method has an additional attribution risk: gains may come from more training tokens, changed rank, or clean-example replay. These factors must be matched in the experimental protocol. The benchmark also does not by itself establish rationale faithfulness; direct CoT interventions and causal faithfulness tests remain separate analyses [9,10,13].

## 9 Conclusion

This draft defines a study of structural robustness and reasoning generalization under label-preserving structural changes. Its measurement module constructs variants from solver-verified reasoning graphs and evaluates held-out structural and compositional splits. The framework separates task-structure changes from distractor context, rationale noise, position effects, and semantic-changing attacks. An optional structural adversarial LoRA protocol tests whether a robustness change transfers to held-out reasoning. The project should make empirical claims only after the construction pilot demonstrates answer preservation, nuisance control, and measurable differences across the proposed splits.

## References

[1] D. Hupkes, V. Dankers, M. Mul, and E. Bruni, “Compositionality Decomposed: How Do Neural Networks Generalise?,” *Journal of Artificial Intelligence Research*, 2020.

[2] J. Oren, J. Herzig, K. Gupta, M. Gardner, and J. Berant, “Improving Compositional Generalization in Semantic Parsing,” *Findings of EMNLP*, 2020.

[3] J. Herzig and J. Berant, “Span-based Semantic Parsing for Compositional Generalization,” *ACL-IJCNLP*, 2021.

[4] L. Qiu et al., “Evaluating the Impact of Model Scale for Compositional Generalization in Semantic Parsing,” *EMNLP*, 2022.

[5] I. Mirzadeh et al., “GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models,” arXiv preprint, 2024.

[6] Q. Li et al., “GSM-Plus: A Comprehensive Benchmark for Evaluating the Robustness of LLMs as Mathematical Problem Solvers,” *ACL*, 2024.

[7] F. Shi et al., “Large Language Models Can Be Easily Distracted by Irrelevant Context,” arXiv preprint, 2023.

[8] N. F. Liu et al., “Lost in the Middle: How Language Models Use Long Contexts,” *Transactions of the Association for Computational Linguistics*, 2024.

[9] Z. Zhou et al., “Can Language Models Perform Robust Reasoning in Chain-of-thought Prompting with Noisy Rationales?,” *NeurIPS*, 2024.

[10] D. Paul, R. West, A. Bosselut, and B. Faltings, “Making Reasoning Matter: Measuring and Improving Faithfulness of Chain-of-Thought Reasoning,” *Findings of EMNLP*, 2024.

[11] O. Press et al., “Measuring and Narrowing the Compositionality Gap in Language Models,” *Findings of EMNLP*, 2023.

[12] R. Jia and P. Liang, “Adversarial Examples for Evaluating Reading Comprehension Systems,” *EMNLP*, 2017.

[13] A. von Recum, L. Girrbach, and Z. Akata, “Are Reasoning LLMs Robust to Interventions on Their Chain-of-Thought?,” arXiv preprint, 2026.

[14] E. J. Hu et al., “LoRA: Low-Rank Adaptation of Large Language Models,” *ICLR*, 2022.

[15] T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer, “QLoRA: Efficient Finetuning of Quantized LLMs,” *NeurIPS*, 2023.

[16] J. Pereira et al., “Targeted Adversarial Training for Natural Language Understanding,” *NAACL*, 2021.

[17] P. Cheng et al., “Self-playing Adversarial Language Game Enhances LLM Reasoning,” *NeurIPS*, 2024.
