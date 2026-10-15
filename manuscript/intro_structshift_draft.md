# Introduction

Large language models (LLMs) increasingly solve multi-step mathematical, logical, and knowledge-intensive questions by decomposing a problem into intermediate reasoning steps. Yet a correct answer on a conventional random split does not necessarily indicate that a model has learned a reasoning structure that transfers beyond the training presentation. Work on compositional generalization has long separated systematicity, productivity, and related forms of structural transfer [1]. In language understanding, compositional splits expose failures that are hidden by random splits [2,3], and scaling alone does not reliably remove the gap [4]. Recent evaluations of mathematical reasoning reach a similar conclusion from a different direction: controlled instantiations and additional clauses can change model behavior even when the underlying task is intended to remain comparable [5,6]. These observations motivate a more specific question: can a model preserve the correct answer when the organization of a reasoning problem changes while the solution remains the same?

Existing evaluations do not yet provide a consistent answer because they perturb different objects under the same broad label of robustness. Compositional-generalization benchmarks hold out new structures, whereas mathematical stress tests vary instantiations or insert distracting clauses [2,5,6]. Other work studies irrelevant context [7], information position in long contexts [8], or the reliability of an already generated chain of thought [9,10]. These settings are related, but they do not share a single threat model. A perturbation may change a task's numerical distribution, its token length, its semantic solution, its context position, or only its rationale; treating all such changes as structural reasoning failures obscures the mechanism being measured. Moreover, a random instance split can test whether a model recognizes familiar components without testing whether it can combine them in an unseen way. A benchmark that claims to measure structural robustness therefore needs to preserve the answer by construction, separate structure from nuisance variables, and evaluate held-out combinations rather than only held-out instances.

We study this setting as **reasoning under label-preserving structural distribution shift**. Each item is generated from a symbolic reasoning graph, rendered as natural language, and paired with an answer verified by a solver. A structural transformation is admitted only when the transformed graph yields the same target answer. The benchmark varies the organization of the task rather than its solution: legal premise reordering, changes in depth or branching, held-out combinations of familiar reasoning operators, and the insertion of a nonessential auxiliary question at a controlled position. This formulation connects the compositionality gap observed in semantic parsing and multi-hop reasoning [4,11] with the distractor sensitivity observed in mathematical and reading-comprehension evaluations [6,7,12]. It also makes the central error measurable. If (R_S(f)) denotes the error rate of a model on a structure group (S), we define a structural gap between an out-of-distribution group and an in-distribution group as (R_{mathrm{OOD}}(f)-R_{mathrm{IID}}(f)). The goal is not to treat this gap as evidence of a single internal failure, but to create an evaluation setting in which competing explanations can be tested.

Three challenges make this setting nontrivial. First, a natural-language rewrite can silently change the answer or the dependencies in the reasoning graph; surface paraphrases are therefore insufficient as a source of ground truth. Second, structural changes are entangled with nuisance factors. A deeper graph may be longer, a reordered premise may move evidence to a different context position, and a side question may share entities or vocabulary with the main question. Without matched controls, a measured drop cannot be attributed to structure. Third, a benchmark should support diagnosis and improvement rather than only ranking. Generating many variants is easy, but selecting variants that expose a model-specific structural failure while retaining clean-task performance requires a controlled training signal. These challenges are distinct from noisy-rationale robustness, where the object being changed is the intermediate explanation [9,13], and from semantic-changing adversarial attacks, where the label itself may no longer be invariant.

To study this question, we use **StructShift** as a solver-validated measurement module. The construction pipeline samples a symbolic reasoning graph, renders a clean natural-language instance, applies a restricted transformation grammar, and re-solves every candidate before release. The evaluation protocol separates IID, Surface-OOD, Structure-OOD, and Composition-OOD splits, and reports clean accuracy, robust accuracy, structural gap, worst-group accuracy, and side-question interference. An error-guided structural adversarial LoRA protocol is retained as a secondary intervention to test whether robustness changes transfer to held-out reasoning. Ordinary LoRA and random augmentation provide controls for data quantity and coverage.

Our contributions are fourfold. (1) We formulate structural robustness as a measurable dimension of reasoning generalization. (2) We specify a reproducible graph-to-text measurement pipeline with solver validation, transformation metadata, nuisance controls, and held-out structural-composition splits. (3) We define a fine-grained analysis of structural gap, complexity slopes, and side-question interference. (4) We provide a matched intervention protocol for testing whether a robustness change transfers to held-out reasoning. The empirical claims associated with these contributions are intentionally left to model evaluation.

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

The running example and all quantitative preview claims should be replaced with verified pilot evidence before submission.
