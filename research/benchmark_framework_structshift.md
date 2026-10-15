# StructShift as a measurement module

StructShift is not the title-level contribution of the project. It is the
controlled construction module used to identify structural robustness and
support the main analysis of reasoning generalization.

## What it must provide

1. Symbolic graphs and solver-verified answers.
2. Legal answer-preserving transformations.
3. Paired base-query IDs and world-level split isolation.
4. Nuisance metadata for length, position, vocabulary, numerical range, depth,
   branching, and operator sequence.
5. IID, Surface-OOD, Structure-OOD, and Composition-OOD test groups.

## What it must not claim

The module does not by itself explain why structural errors occur, establish a
hidden reasoning mechanism, or demonstrate that a training intervention
improves general reasoning. Those claims belong to the mechanism and transfer
experiments in the main paper.

## Required quality checks

- Every transformed item is re-solved.
- Changed answers, malformed references, ambiguous targets, and duplicate
  graphs are rejected.
- A stratified human sample checks semantic preservation and readability.
- Length and target position are matched when they are not the target factor.
- Confidence intervals aggregate by base query rather than rendered row.

## Relationship to the main paper

The data module supplies the independent variable (structural shift), the
outcome (model answer), and the controls needed for residual-error analysis.
The main result is the relationship between the resulting robustness score and
held-out reasoning ability.
