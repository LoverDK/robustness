# Evidence Map for the StructShift Draft

| ID | Source | Level | Supports | Cannot support | Planned use | Risk |
|---|---|---|---|---|---|---|
| E1 | Hupkes et al. (2020), verified metadata and title | L3 | compositionality can be decomposed into systematicity/productivity-related properties | exact taxonomy details beyond title-level unless full text is consulted | Introduction, Related Work | metadata-only |
| E2 | Oren et al. (2020), verified ACL record | L3 | compositional generalization in semantic parsing as a distinct evaluation setting | exact model results | Related Work | metadata-only |
| E3 | Herzig & Berant (2021), verified ACL record | L3 | span-based structural representation for compositional generalization | exact result values | Related Work | metadata-only |
| E4 | Qiu et al. (2022), verified ACL record and abstract-level retrieval | L2/L3 | model scale and training regime do not automatically solve compositional OOD | exact scaling values unless paper text is consulted | Introduction, Related Work | bounded claim only |
| E5 | Press et al. (2023), verified ACL record and abstract-level retrieval | L2/L3 | compositionality gap and self-ask/follow-up-question connection | exact gains or broad generalization | Introduction, side-question discussion | bounded claim only |
| E6 | Dziri et al. (2023), verified arXiv abstract | L2 | computation-graph complexity and linearized subgraph matching as an analysis of compositional reasoning limits | universal claim about all LLMs | Related Work, design rationale | abstract-level |
| E7 | Mirzadeh et al. (2024), verified arXiv abstract | L2 | symbolic variants, clause complexity, and irrelevant-looking clauses expose mathematical reasoning fragility | exact percentage unless directly quoted from paper | Introduction, benchmark gap | abstract-level |
| E8 | Li et al. (2024), verified ACL/Crossref metadata | L3 | GSM-Plus is a benchmark for robustness of LLM mathematical problem solving | detailed perturbation taxonomy without full text | Related Work | metadata-only |
| E9 | Shi et al. (2023), verified arXiv metadata/abstract retrieval | L2/L3 | irrelevant context is an adjacent distractor setting | exact benchmark numbers | Related Work, side-question rationale | bounded claim only |
| E10 | Liu et al. (2024), verified TACL record and abstract-level retrieval | L2/L3 | information position in long contexts is an independent confound | reasoning-specific structural causality | Evaluation controls | bounded claim only |
| E11 | Jia & Liang (2017), verified ACL record | L3 | distracting sentences are a precedent for answer-preserving context interference | direct equivalence to side questions | Related Work | scope difference |
| E12 | Zhou et al. (2024), verified NeurIPS/arXiv record and abstract | L2 | noisy/inaccurate rationales are a separate robustness object | task-structure claims | Related Work, threat-model boundary | abstract-level |
| E13 | Paul et al. (2024), verified Crossref/arXiv abstract | L2 | CoT faithfulness and OOD robustness can be studied with causal/faithful reasoning objectives | exact implementation or gains | Related Work | abstract-level |
| E14 | von Recum et al. (2026), verified arXiv abstract | L2 | direct CoT interventions are distinct from input-task perturbations | peer-reviewed status or universal claim | Related Work boundary | preprint |
| E15 | Hu et al. (2022), verified arXiv/ICLR metadata | L3 | LoRA is a low-rank parameter-efficient adaptation mechanism | structural robustness improvement | Method baseline | metadata-only |
| E16 | Dettmers et al. (2023), verified arXiv/NeurIPS metadata | L3 | QLoRA combines quantization with LoRA for efficient adaptation | reasoning robustness benefit | Method baseline | metadata-only |
| E17 | Pereira et al. (2021), verified NAACL record | L3 | targeted adversarial training prioritizes current model errors | effectiveness in structural reasoning | Method rationale | task mismatch |
| E18 | Cheng et al. (2024), verified arXiv abstract | L2 | self-play is an adversarial reasoning-training precedent | label-preserving structural adaptation | Related Work | abstract-level |
| E19 | Wu et al. (2024), verified NAACL record | L3 | counterfactual tasks distinguish reasoning from recitation | structural threat model details | Related Work | metadata-only |
| E20 | User-provided project materials | L1 | proposed graph generator, transforms, splits, metrics, and planned LoRA comparisons | empirical results, final scale, model list | Methods and experimental plan | pending validation |

## Evidence boundaries

* No source in this map supports a numerical result for StructShift.
* No source supports the claim that structural adversarial LoRA is already effective; it is a proposed experiment.
* The literature gap is scoped to the combination found in this search: label-preserving task-structure transformations, held-out structural composition, and adversarial PEFT/LoRA.
* Final submission requires rechecking the newest 2025--2026 literature and replacing abstract-level claims with full-text citations where stronger detail is needed.

