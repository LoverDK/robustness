# Figure Design Specification: Structural Robustness and Reasoning Generalization

This document specifies the three figures that carry the main story of the
explanatory structural-robustness paper. The design follows the current scope:
solver-validated, label-preserving structural shifts; held-out structural
compositions; and an optional structural adversarial LoRA protocol. No figure
may contain numerical results or a qualitative failure claim until the pilot
and model runs are complete.

## Figure 1: Motivated Example

### 1. Figure type

- **Type:** motivated example with paired failure analysis
- **Reason:** the paper's central distinction is easiest to understand from a
  clean reasoning task and a label-preserving structural variant whose answer
  is unchanged.
- **Status:** design-ready, content pending. The concrete item must come from
  the verified pilot and must be reused in the construction walkthrough and a
  later error analysis. Do not invent an example for the camera-ready paper.

### 2. Paradigm recommendation

- **Paradigm:** running example plus failure case, with a compact structural
  comparison.
- **Why:** a paired item makes the invariant answer visible while exposing the
  organization of the reasoning problem as the manipulated object.
- **Alternatives considered and rejected:**
  - **Existing vs. ours:** premature because the empirical relationship is the
    primary claim and the companion LoRA method is optional.
  - **Performance teaser:** unavailable before experiments and would turn a
    design proposal into an unsupported result claim.

### 3. Layout sketch

- **Canvas:** single-column width, approximately 180 mm wide by 68--78 mm
  high; three horizontal panels with a shared baseline.
- **Panel A, left (clean task):**
  - Header: `Clean instance`.
  - Show the pilot item's natural-language premises in a compact text block.
  - Show the target question in a visually separated final line.
  - Below the text, show a small directed reasoning graph with named nodes
    taken from the actual item.
- **Panel B, center (structural transformation):**
  - Header: `Label-preserving structural shift`.
  - Show only the changed organization: premise/step order, depth/branching,
    operator composition, or side-question placement, depending on the pilot
    item selected.
  - Mark unchanged content in neutral gray and the changed relation in amber.
  - Add a small validation badge: `solver answer = y`.
- **Panel C, right (model behavior):**
  - Header: `Same answer, different transfer behavior`.
  - Show the solver answer once, connected to both inputs by a green equality
    marker.
  - If a model failure has been observed, show the model's wrong answer in a
    red outline and the solver answer in green. If no failure has been
    observed, show `model output: pending pilot` outside the final figure and
    omit Panel C's failure annotation.
- **Arrows and connections:**
  - A solid arrow from Panel A to Panel B labelled with the exact transform
    name, such as `premise reorder` or `side question insertion`.
  - Two thin arrows from Panels A and B to the shared solver-answer marker in
    Panel C.
  - No crossing arrows and no decorative arrows.
- **Colour assignment:**
  - Clean content: dark neutral (`#343A40`).
  - Changed structure: amber (`#D95F02`).
  - Solver-verified invariant answer: green (`#1B9E77`).
  - Observed model error: red (`#D73027`).
  - Unchanged or contextual text: light gray (`#6C757D`).
  - Use line style and labels in addition to colour so the figure survives
    grayscale printing.

### 4. Labelling and annotations

- Use the actual task-family name, entity names, operators, and answer format
  from the pilot item. Do not label content as `Entity 1`, `Example A`, or
  `Module X` in the final figure.
- The transformation annotation must state exactly which graph property
  changed and which properties were matched as nuisance controls.
- Add a short bracket under the pair: `target answer preserved by solver`.
- If the item includes a side question, label its position explicitly as
  `before`, `inside`, or `after` the main context.
- Target font sizes: 8.5--9 pt at final placement; no text below 8 pt.
- Keep the body text to one or two lines per premise. Move the full item to
  the appendix if it cannot remain legible at single-column width.

### 5. Tool suggestion

- **Primary:** PowerPoint for the first layout, then export to PDF.
- **Alternative:** Figma for a polished version; TikZ only after the example
  and labels are frozen.
- **Reason:** the figure mixes natural-language text, graph nodes, arrows, and
  annotations. It is not a data plot.

### 6. Caption draft

> **Figure 1:** A solver-validated structural transformation changes the
> organization of a reasoning task while preserving its target answer. The
> clean and transformed instances share the same verified label, but the
> transformation exposes whether a model transfers the underlying reasoning
> structure rather than relying on the training presentation. The concrete
> item and any model output shown here must be taken from the verified pilot.

### 7. Universal rule audit

- [pass] Vector export: PDF or SVG; no raster screenshot of the text.
- [pass] Typography: at least 8 pt after insertion into the paper.
- [pass] Colour-blind safety: colour is paired with labels, outlines, and line
  styles.
- [pass] Caption: self-contained and limited to the claim supported by the
  displayed pair.
- [pass] Axes: not applicable; no quantitative axis is used.
- [pass] Decoration: no 3D effects, gradients, shadows, or ornamental icons.
- [user-verify] The item must be the same running example used in the
  Introduction and Methods.
- [user-verify] A red model-failure marker may be added only after a verified
  model error exists.

## Figure 2: StructShift Construction and Evaluation Pipeline

### 1. Figure type

- **Type:** solution overview, pipeline paradigm
- **Reason:** the benchmark is a sequence of controlled transformations and
  validation steps with explicit train/test split outputs.

### 2. Paradigm recommendation

- **Paradigm:** left-to-right pipeline with a validation gate and a split
  branch.
- **Why:** readers need to locate the source graph, the natural-language
  rendering, the structural transformation, the solver check, and the four
  evaluation splits in one view.
- **Alternatives considered and rejected:**
  - **System architecture:** would overemphasize interactions that are
    sequential in the benchmark generator.
  - **Multi-layer training/inference view:** better suited to the optional
    LoRA procedure, not to the core benchmark construction.

### 3. Layout sketch

- **Canvas:** two-column width, approximately 180 mm wide by 82--95 mm high.
- **Row 1, generation path:** five equal-width stages from left to right:
  1. `Symbolic seed graph` — controlled depth, branching, operator sequence,
     and answer.
  2. `Solver + trace` — deterministic solution and retained trace.
  3. `Natural-language renderer` — templates and lexical variation.
  4. `Structural transform grammar` — order, depth/branching, composition,
     or side-question insertion.
  5. `Candidate pair` — clean item, transformed item, and transform metadata.
- **Row 2, validation and release:**
  - A wide gate below Stage 5 labelled `re-solve transformed graph`.
  - Green branch: `answer preserved` leads to `duplicate check + nuisance
    matching`.
  - Red branch: `answer changed / malformed / ambiguous` leads to `reject +
    rejection log`.
  - The accepted branch ends in four labelled output cards:
    `IID`, `Surface-OOD`, `Structure-OOD`, and `Composition-OOD`.
- **Side annotation:** a thin bracket around the metadata stream labels
  `source graph`, `transform sequence`, `perturbation budget`, `solver trace`,
  and `nuisance statistics`.
- **Optional inset:** a small dashed box at the bottom right labelled
  `evaluation metrics` containing `clean accuracy`, `robust accuracy`,
  `structural gap`, `worst-group accuracy`, and `interference cost`.
- **Arrows and connections:**
  - Solid arrows for item flow.
  - Dashed arrows for metadata flow.
  - The rejection loop returns to the generator only in the conceptual sense;
    do not draw a large feedback loop that obscures the pipeline.
- **Colour assignment:**
  - Data and examples: blue (`#377EB8`).
  - Solvers and validation: green (`#1B9E77`).
  - Transformations: amber (`#D95F02`).
  - Splits and metrics: purple (`#7570B3`).
  - Rejected candidates: red (`#D73027`).
  - Use light fills and dark outlines; colour is always accompanied by text.

### 4. Labelling and annotations

- Every stage name must match a subsection title or a named procedure in the
  paper.
- Show the invariant explicitly at the validation gate:
  `Solve(T_delta(G), q) = Solve(G, q)`.
- Put the split definition inside each output card:
  - `IID: same graph distribution`;
  - `Surface-OOD: new wording/entities`;
  - `Structure-OOD: held-out structure axis`;
  - `Composition-OOD: held-out operator combination`.
- Avoid putting full paragraphs inside boxes. Use one explanatory line per
  stage and move details to the caption.
- Target font sizes: 8.5--9 pt; stage titles may use 9.5 pt semibold.

### 5. Tool suggestion

- **Primary:** diagrams.net (draw.io), exported as PDF or SVG.
- **Alternative:** PowerPoint for an early draft; TikZ for the final version
  if the paper's typography requires LaTeX-native output.
- **Reason:** the figure is a modular pipeline with aligned boxes, branches,
  and a validation gate. Grid snapping and vector export matter more than
  free-form illustration.

### 6. Caption draft

> **Figure 2:** StructShift constructs paired reasoning items by sampling a
> symbolic graph, solving it before rendering, applying a restricted
> structural transformation, and re-solving the candidate. Only candidates
> with preserved answers and valid text pass the release gate; accepted items
> are assigned to IID, Surface-OOD, Structure-OOD, or Composition-OOD splits
> with transformation and nuisance metadata retained for diagnosis.

### 7. Universal rule audit

- [pass] Vector export: PDF/SVG from diagrams.net or TikZ.
- [pass] Typography: boxes are sized for 8.5 pt labels at final width.
- [pass] Colour-blind safety: stage categories also use labels and line types.
- [pass] Caption: states the construction rule and acceptance criterion.
- [pass] Axes: not applicable.
- [pass] Decoration: no gradients, 3D shapes, drop shadows, or clip art.
- [inspection] Pipeline paradigm matches the benchmark construction task.
- [user-verify] Section titles and stage names must remain synchronized after
  the Methods section is revised.

## Figure 3: Experimental Results and Structural Gaps

### 1. Figure type

- **Type:** experimental results, multi-panel quantitative figure
- **Reason:** StructShift needs to show both overall split performance and the
  distribution of failures across structural groups. No values are available
  yet, so this is a plotting specification rather than a result.

### 2. Paradigm recommendation

- **Paradigm:** a two-panel figure combining a grouped bar chart and a
  structure-group heatmap; add a third scatter panel only if the LoRA study is
  completed with matched compute measurements.
- **Why:** grouped bars answer the overall comparison question, while a
  heatmap exposes which transformations account for the aggregate structural
  gap.
- **Alternatives considered and rejected:**
  - **Radar chart:** difficult to compare exact values and weak for a benchmark
    with multiple baselines.
  - **Single bar chart:** hides the difference between IID performance and
    held-out structural transfer.
  - **Line chart:** appropriate for training steps or model scale, but not the
    primary split comparison.

### 3. Layout sketch

- **Canvas:** two-column width, approximately 180 mm wide by 92--105 mm high.
- **Panel A, left (grouped bars):**
  - x-axis: `Evaluation split` with `IID`, `Surface-OOD`, `Structure-OOD`,
    and `Composition-OOD`.
  - y-axis: `Answer accuracy (%)`, with a truthful range chosen after results
    are known. Do not truncate the axis to exaggerate a small difference.
  - Bars: one group per split, one bar per method. Use neutral gray for base
    and control conditions; use a single consistent accent for structural
    adversarial LoRA only if it is a completed method and the paper argues for
    it. If no method dominates, keep all methods in a balanced palette.
  - Error bars: confidence intervals or standard errors across random seeds.
- **Panel B, right (heatmap):**
  - rows: transformation groups, including `premise order`, `depth`,
    `branching`, `operator composition`, and `side-question position` when
    present in the final benchmark.
  - columns: base model, ordinary LoRA, random augmentation + LoRA,
    structural adversarial LoRA, and other preregistered controls.
  - cell value: accuracy or error rate, stated in the legend; use a perceptual
    sequential palette such as Viridis.
  - annotate cells with values only when they remain legible at final scale.
- **Optional Panel C (scatter):**
  - x-axis: training tokens or GPU-hours under the matched budget.
  - y-axis: Structure-OOD accuracy.
  - annotate every point with the method name; use marker shape in addition to
    colour. Include only if compute accounting is available for every method.
- **Connections:** no arrows; panel labels `(a)`, `(b)`, and optional `(c)`.
- **Style:** shared font, line widths, and method order across every results
  figure. Define these in a reusable `plot_utils.py` file.

### 4. Labelling and annotations

- Use exact method names from the experiment table; do not abbreviate a method
  in the plot unless the caption defines the abbreviation.
- Add a horizontal dashed reference only for a preregistered baseline, such as
  `IID accuracy of the base model`; never add a post hoc target line.
- Mark the structural gap in Panel A with a bracket or a small annotation
  between IID and Structure-OOD only after the relevant values are available.
- Use dual encoding: colour plus hatch or marker shape for methods. Hatches
  should be sparse and functional, not decorative.
- Target font sizes: 8--8.5 pt for tick labels and 8.5--9 pt for legend and
  axis labels after final scaling.

### 5. Tool suggestion

- **Primary:** Matplotlib and Seaborn, driven by a version-controlled script.
- **Alternative:** PGFPlots for final LaTeX typography after the data schema
  and plot design are stable.
- **Reason:** all result panels must be regenerated when the benchmark or
  model runs change. Manual chart editing would break reproducibility.

### 6. Caption draft

> **Figure 3:** Overall split accuracy and transformation-level performance
> reveal where models lose accuracy under label-preserving structural shift.
> Panel (a) compares preregistered model conditions across IID and OOD splits
> with uncertainty intervals; Panel (b) decomposes performance by structural
> transformation and model condition. Values, error bars, and any highlighted
> finding must be filled from completed runs and should not be interpreted
> before nuisance matching and solver validation are reported.

### 7. Universal rule audit

- [pass] Vector export: save PDF and, when needed, SVG from Matplotlib.
- [pass] Typography: design at final two-column size with labels at least 8 pt.
- [pass] Colour-blind safety: Viridis for the heatmap; methods use colour plus
  marker or hatch.
- [pass] Caption: first sentence states the supported finding in general form;
  replace it with the observed finding once results exist.
- [user-verify] Axis range must be selected from the actual result distribution
  and reported honestly.
- [user-verify] Error bars require repeated runs or a stated confidence
  interval procedure.
- [user-verify] Do not highlight structural adversarial LoRA as superior unless
  the completed comparison supports that interpretation.
- [pass] No 3D effects, gradients, shadows, or chartjunk.

## Cross-figure quality-control checklist

- [inspection] Figure 1 is a motivated example, Figure 2 is a construction
  pipeline, and Figure 3 is an experimental-results plot.
- [inspection] The layouts are concrete enough to draw without inferring
  missing modules or axes.
- [inspection] Labels use StructShift entities and named procedures; the only
  intentionally unresolved content is the pilot item and future results.
- [inspection] Tool choices match the figure types: PowerPoint/Figma for the
  mixed-content example, diagrams.net/TikZ for the pipeline, and
  Matplotlib/Seaborn for data plots.
- [inspection] No figure design requires an unreported number, benchmark
  result, or model comparison.
- [user-verify] Reuse one verified Figure 1 item consistently across the
  Introduction, construction walkthrough, and qualitative analysis.
- [user-verify] Synchronize method names, split names, and metric definitions
  with the final paper before export.
- [user-verify] Export all paper figures as vector files and inspect them at
  final size for text overlap and legibility.

## Recommended file organization after implementation

```text
figures/
  fig1_motivated_example.pdf
  fig2_pipeline.pdf
  fig3_results.pdf
scripts/
  plot_utils.py
  plot_fig3.py
```

The current repository intentionally contains only the design specification.
Actual figure files should be generated after the pilot supplies the Figure 1
item and the experiments supply the Figure 3 data.
