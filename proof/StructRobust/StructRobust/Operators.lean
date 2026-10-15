import StructRobust.Basic
import StructRobust.Closure

namespace StructRobust

def solverYes (facts : List Atom) (rules : List Rule) (query : Atom) : Prop :=
  query ∈ closure rules.length facts rules

theorem solver_iff_derives (facts : List Atom) (rules : List Rule) (query : Atom) :
    solverYes facts rules query ↔ Derives facts rules query :=
  closure_fuel_iff_derives facts rules query (Nat.le_refl _)

theorem sameAtoms_refl (xs : List Atom) : sameAtoms xs xs :=
  fun _ => Iff.rfl

theorem sameRules_refl (xs : List Rule) : sameRules xs xs :=
  fun _ => Iff.rfl

theorem derives_facts_congr {facts facts' : List Atom} {rules : List Rule} {a : Atom}
    (h : sameAtoms facts facts') :
    Derives facts rules a ↔ Derives facts' rules a := by
  constructor
  · intro hd
    induction hd with
    | ofFact hf =>
      exact Derives.ofFact ((h _).1 hf)
    | ofRule hr hp ih =>
      apply Derives.ofRule hr
      intro p hpm
      exact ih p hpm
  · intro hd
    induction hd with
    | ofFact hf =>
      exact Derives.ofFact ((h _).2 hf)
    | ofRule hr hp ih =>
      apply Derives.ofRule hr
      intro p hpm
      exact ih p hpm

/-- A stability transformation preserves the fact set, the rule set, and the query. -/
structure IsStability (t t' : Task) : Prop where
  facts : sameAtoms t.facts t'.facts
  rules : sameRules t.rules t'.rules
  query : t.query = t'.query

theorem stability_preserves (t t' : Task) (h : IsStability t t') :
    solverYes t.facts t.rules t.query ↔ solverYes t'.facts t'.rules t'.query := by
  rw [h.query, solver_iff_derives, solver_iff_derives]
  constructor
  · intro hd
    exact (derives_rules_congr h.rules).1 ((derives_facts_congr h.facts).1 hd)
  · intro hd
    exact (derives_facts_congr h.facts).2 ((derives_rules_congr h.rules).2 hd)

/-- The order operator replaces the rule list by another list with the same members. -/
def reorder (t : Task) (rules' : List Rule) : Task :=
  { t with rules := rules' }

theorem lemma1_reorder_stable (t : Task) (rules' : List Rule)
    (h : sameRules t.rules rules') : IsStability t (reorder t rules') where
  facts := sameAtoms_refl _
  rules := h
  query := rfl

/-- Lemma 1, order: rearranging rules does not change the solver answer. -/
theorem lemma1_order_preserves (facts : List Atom) (rules rules' : List Rule) (query : Atom)
    (h : sameRules rules rules') :
    solverYes facts rules query ↔ solverYes facts rules' query := by
  rw [solver_iff_derives, solver_iff_derives]
  exact derives_rules_congr h

/-- Lemma 1, synonym: one layer of wrapping does not change the denoted rule. -/
theorem lemma1_synonym_denote (p : Phrase) (w : Nat) :
    denote (rephrase p w) = denote p := rfl

theorem lemma1_synonym_extract (ps : List Phrase) (w : Nat) :
    extract (ps.map (fun p => rephrase p w)) = extract ps := by
  induction ps with
  | nil => rfl
  | cons p ps ih =>
    simp [extract, List.map_cons, denote, rephrase, ih]

def dropPremise (r : Rule) : Rule :=
  { r with ps := r.ps.drop 1 }

def ambRule : Rule := { ps := [0], hd := 1 }

def ambTask : Task := { facts := [], rules := [ambRule], query := 1 }

def ambDropped : Task :=
  { facts := [], rules := [dropPremise ambRule], query := 1 }

theorem dropPremise_amb : dropPremise ambRule = { ps := [], hd := 1 } := by
  simp [dropPremise, ambRule]

theorem derives_amb_false : ∀ a, ¬ Derives ([] : List Atom) [ambRule] a := by
  intro a h
  induction h with
  | ofFact hf =>
    cases hf
  | ofRule hr hp ih =>
    simp [List.mem_singleton] at hr
    subst hr
    exact ih 0 (by simp [ambRule])

theorem derives_dropped : Derives ([] : List Atom) [dropPremise ambRule] 1 := by
  rw [dropPremise_amb]
  apply Derives.ofRule (r := { ps := [], hd := 1 })
  · simp
  · intro _ hp
    cases hp

/-- Lemma 1: deleting a premise can change the solver answer on a concrete task. -/
theorem lemma1_ambiguity_changes :
    ¬ solverYes ambTask.facts ambTask.rules ambTask.query ∧
      solverYes ambDropped.facts ambDropped.rules ambDropped.query := by
  constructor
  · intro h
    exact derives_amb_false ambTask.query
      ((solver_iff_derives ambTask.facts ambTask.rules ambTask.query).1 h)
  · exact (solver_iff_derives ambDropped.facts ambDropped.rules ambDropped.query).2
      derives_dropped

/-- Therefore ambiguity is not a stability transformation. -/
theorem lemma1_ambiguity_not_stability : ¬ IsStability ambTask ambDropped := by
  intro h
  have hagree := (stability_preserves ambTask ambDropped h).mpr
  exact lemma1_ambiguity_changes.1 (hagree lemma1_ambiguity_changes.2)

def graphA : Task :=
  { facts := [0], rules := [{ ps := [0], hd := 1 }], query := 1 }

def graphB : Task :=
  { facts := [0], rules := [{ ps := [0], hd := 2 }], query := 2 }

/-- Lemma 1: replacing the rule set changes the graph, so it is not a stability transformation. -/
theorem lemma1_graph_not_stability : ¬ IsStability graphA graphB := by
  intro h
  have hmem : ({ ps := [0], hd := 1 } : Rule) ∈ graphB.rules :=
    (h.rules _).1 (by simp [graphA])
  simp [graphB] at hmem

end StructRobust
