# Vibe Research Workflow for Structural Robustness and Reasoning Generalization

## 1. Phase

- **Primary phase:** mixed figure and experiment preparation.
- **Secondary phases:** benchmark construction, coding, paper maintenance,
  and later result writing.

## 2. Behavioural rules recap

1. AI assistance is limited to literature organisation, code support,
   debugging, and language or layout acceleration.
2. The user owns the research question, task design, method choices,
   experimental plan, conclusions, and novelty claims.
3. Every generated code block, paragraph, figure label, and numerical result
   must be inspected against the actual project state.
4. Citations must be checked by the user in a scholarly source before use.
5. No fabricated data, results, examples presented as observations, or hidden
   plagiarism.
6. Check EACL and institutional AI-disclosure rules before submission.

## 3. Recommended work split

| Work package | User owns | AI/Codex can assist | Verification gate |
|---|---|---|---|
| Research framing | choose the exact claim and scope | compare alternatives and identify risks | advisor/user accepts one frozen framing |
| Benchmark grammar | define legal transformations and exclusions | implement generators and validators | solver checks plus manual sample |
| Dataset splits | decide what counts as seen/unseen | produce split scripts and leakage checks | inspect world/query disjointness |
| Baseline evaluation | choose model, prompt, and interpretation | write inference and parser code | fixed revision, prompt, and raw logs |
| LoRA study | decide whether to run and what counts as success | scaffold training, logging, and ablations | matched tokens, steps, seeds, and clean retention |
| Figures | approve message and captions | generate editable Draw.io and plotting code | inspect at final paper size |
| Writing | verify every claim and citation | polish language and structure | sentence-by-sentence verification |

## 4. Small-step schedule

| Step | Deliverable | User decision |
|---|---|---|
| 1 | Freeze one task family and three legal transformations | approve threat model |
| 2 | Generate a 50--100 item validation pilot | inspect labels and paired examples |
| 3 | Run leakage, answer-preservation, and nuisance checks | approve dataset release candidate |
| 4 | Evaluate one fixed model without training | decide whether the gap is reproducible |
| 5 | Add a second model or task family only if needed | assess external validity |
| 6 | Run clean LoRA and random augmentation controls | check token/step matching |
| 7 | Run offline error-guided QLoRA | decide whether method claims are justified |
| 8 | Replace Figure 1 schematic with a verified item and Figure 3 placeholders with results | approve paper update |

Each step should be implemented, run, inspected, and committed before the next
step begins. Keep raw predictions and configuration files under version
control or an explicitly archived experiment directory.

## 5. Sub-agent coordination

Use sub-agents for bounded, independent work with explicit outputs:

- one agent can audit the literature and citation claims;
- one agent can implement or review the symbolic generator and split checks;
- one agent can review LoRA controls and resource assumptions.

The user remains responsible for merging their recommendations into the final
research design. Agents should not independently change the paper's central
claim or add unverified citations.

## 6. Figure workflow

1. Discuss the single message of each figure with the advisor/user.
2. Freeze the layout and labels before polishing.
3. Keep Figure 1 and Figure 2 editable in Draw.io.
4. Generate Figure 3 from experiment outputs with a reproducible script.
5. Export vector PDF/SVG, inspect text at final size, and keep the source file.

The current repository contains pre-experiment Figure 1 and Figure 2
schematics, plus a no-data Figure 3 template. They are explicitly labelled as
such in the captions.

## 7. Integrity gate before submission

- User verifies every citation and every sentence containing a result.
- User confirms that all reported numbers come from stored experiment outputs.
- User checks the venue's AI policy and discloses assistance as required.
- Advisor reviews the final framing, novelty statement, and interpretation.
- No schematic example or toy-pilot value is presented as an observed model
  result.
