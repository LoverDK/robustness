import StructRobust.Operators

namespace StructRobust

def fAnswer (facts : List Atom) (surface : List Phrase) (query : Atom) : Prop :=
  solverYes facts (extract surface) query

theorem mem_extract (ps : List Phrase) (r : Rule) :
    r ∈ extract ps ↔ ∃ s, s ∈ ps ∧ denote s = r := by
  induction ps with
  | nil =>
    simp [extract]
  | cons s ps ih =>
    rw [extract, List.map_cons, List.mem_cons]
    constructor
    · intro h
      cases h with
      | inl heq =>
        exact ⟨s, List.mem_cons_self, heq.symm⟩
      | inr hr =>
        obtain ⟨s', hs, hd⟩ := ih.mp hr
        exact ⟨s', List.mem_cons_of_mem s hs, hd⟩
    · intro h
      obtain ⟨s', hs, hd⟩ := h
      rcases List.mem_cons.mp hs with rfl | hs
      · exact Or.inl hd.symm
      · exact Or.inr (ih.mpr ⟨s', hs, hd⟩)

theorem extract_same_of_mem {ps qs : List Phrase} (h : ∀ s, s ∈ ps ↔ s ∈ qs) :
    sameRules (extract ps) (extract qs) := by
  intro r
  constructor
  · intro hr
    match (mem_extract ps r).1 hr with
    | ⟨s, hs, hd⟩ =>
      exact (mem_extract qs r).2 ⟨s, (h s).1 hs, hd⟩
  · intro hr
    match (mem_extract qs r).1 hr with
    | ⟨s, hs, hd⟩ =>
      exact (mem_extract ps r).2 ⟨s, (h s).2 hs, hd⟩

/-- On a stability orbit, closure membership of the extracted rules is the solver answer. -/
theorem main_positive_orbit (facts : List Atom) (canon surface : List Phrase) (query : Atom)
    (h : sameRules (extract surface) (extract canon)) :
    fAnswer facts surface query ↔ solverYes facts (extract canon) query := by
  unfold fAnswer
  rw [solver_iff_derives, solver_iff_derives]
  exact derives_rules_congr h

/-- Reordering the surface list does not change the extracted rule set, hence not `f`. -/
theorem main_positive_reorder (facts : List Atom) (ps qs : List Phrase) (query : Atom)
    (h : ∀ s, s ∈ ps ↔ s ∈ qs) :
    fAnswer facts ps query ↔ fAnswer facts qs query := by
  unfold fAnswer
  exact lemma1_order_preserves facts (extract ps) (extract qs) query (extract_same_of_mem h)

/-- Synonym wrapping does not change extraction, hence not `f`. -/
theorem main_positive_synonym (facts : List Atom) (ps : List Phrase) (w query : Nat) :
    fAnswer facts (ps.map (fun p => rephrase p w)) query ↔ fAnswer facts ps query := by
  unfold fAnswer
  rw [lemma1_synonym_extract]

/-- Finite SS: if the canonical writing is answered correctly, every equivalent writing is too. -/
theorem main_positive_finite_ss (facts : List Atom) (canon surface : List Phrase) (query : Atom)
    (h : sameRules (extract surface) (extract canon))
    (hcanon : fAnswer facts canon query) :
    fAnswer facts surface query := by
  have hsol : solverYes facts (extract canon) query :=
    (main_positive_orbit facts canon canon query (sameRules_refl _)).mp hcanon
  exact (main_positive_orbit facts canon surface query h).mpr hsol

theorem main_positive_finite_ss_all (items : List (List Atom × List Phrase × Atom))
    (transform : List Atom × List Phrase × Atom → List Phrase)
    (h : ∀ item, item ∈ items →
      sameRules (extract (transform item)) (extract item.2.1)) :
    ∀ item, item ∈ items →
      fAnswer item.1 item.2.1 item.2.2 → fAnswer item.1 (transform item) item.2.2 := by
  intro item hmem hcanon
  exact main_positive_finite_ss item.1 item.2.1 (transform item) item.2.2 (h item hmem) hcanon

/-- An unseen item that is still extracted correctly is solved in both writings. -/
theorem main_positive_unseen (facts : List Atom) (canon surface : List Phrase) (query : Atom)
    (h : sameRules (extract surface) (extract canon))
    (hsol : solverYes facts (extract canon) query) :
    fAnswer facts canon query ∧ fAnswer facts surface query := by
  constructor
  · exact (main_positive_orbit facts canon canon query (sameRules_refl _)).mpr hsol
  · exact (main_positive_orbit facts canon surface query h).mpr hsol

inductive Writing
  | seen (tag : Nat)
  | unseenCanon

def seenRule : Rule := { ps := [0], hd := 1 }
def unseenRule : Rule := { ps := [], hd := 2 }

def factsOfW : Writing → List Atom
  | .seen _ => [0]
  | .unseenCanon => []

def rulesOfW : Writing → List Rule
  | .seen _ => [seenRule]
  | .unseenCanon => [unseenRule]

def queryOfW : Writing → Atom
  | .seen _ => 1
  | .unseenCanon => 2

def solverOfW (w : Writing) : Prop :=
  solverYes (factsOfW w) (rulesOfW w) (queryOfW w)

/-- Correct on every writing of the seen graph, and the opposite answer on one unseen item. -/
def fArbitrary : Writing → Prop
  | .seen _ => True
  | .unseenCanon => False

theorem seen_derives (_tag : Nat) : Derives [0] [seenRule] 1 := by
  apply Derives.ofRule (r := seenRule)
  · simp [seenRule]
  · intro p hp
    simp [seenRule, List.mem_singleton] at hp
    subst hp
    exact Derives.ofFact (by simp)

theorem unseen_derives : Derives ([] : List Atom) [unseenRule] 2 := by
  apply Derives.ofRule (r := unseenRule)
  · simp [unseenRule]
  · intro p hp
    simp [unseenRule] at hp

theorem main_negative_seen_correct (tag : Nat) :
    fArbitrary (.seen tag) ↔ solverOfW (.seen tag) := by
  constructor
  · intro _
    unfold solverOfW
    rw [solver_iff_derives]
    exact seen_derives tag
  · intro _
    trivial

theorem main_negative_seen_stable (t1 t2 : Nat) :
    fArbitrary (.seen t1) ↔ fArbitrary (.seen t2) :=
  ⟨fun _ => trivial, fun _ => trivial⟩

theorem main_negative_unseen_incorrect :
    ¬ (fArbitrary .unseenCanon ↔ solverOfW .unseenCanon) := by
  intro h
  have hsol : solverOfW .unseenCanon := by
    unfold solverOfW
    rw [solver_iff_derives]
    exact unseen_derives
  exact h.mpr hsol

theorem main_negative_disjoint :
    ¬ sameRules (rulesOfW (.seen 0)) (rulesOfW .unseenCanon) := by
  intro h
  have hmem : seenRule ∈ rulesOfW .unseenCanon :=
    (h seenRule).1 (by simp [rulesOfW, seenRule])
  simp [rulesOfW, unseenRule, List.mem_singleton] at hmem
  have hhd : (1 : Nat) = 2 := by
    have := congrArg Rule.hd hmem
    simpa [seenRule, unseenRule] using this
  cases hhd

end StructRobust
