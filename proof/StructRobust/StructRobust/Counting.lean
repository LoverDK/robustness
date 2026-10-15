import StructRobust.Basic

namespace StructRobust

/-- One observation: canonical correctness `c` and transformed correctness `s`, each 0 or 1. -/
structure Obs where
  c : Bool
  s : Bool
deriving DecidableEq, Repr

def countC (xs : List Obs) (b : Bool) : Nat :=
  xs.countP (fun o => o.c == b)

def countS (xs : List Obs) (b : Bool) : Nat :=
  xs.countP (fun o => o.s == b)

def countCS (xs : List Obs) (c s : Bool) : Nat :=
  xs.countP (fun o => o.c == c && o.s == s)

theorem countC_split (xs : List Obs) :
    countC xs true = countCS xs true false + countCS xs true true := by
  induction xs with
  | nil =>
    simp [countC, countCS]
  | cons o xs ih =>
    simp only [countC, countCS] at ih ⊢
    simp at ih
    cases o with
    | mk c s =>
      cases c <;> cases s <;> simp [List.countP_cons]
      all_goals (try rw [ih])
      all_goals omega

theorem countS_split (xs : List Obs) :
    countS xs true = countCS xs false true + countCS xs true true := by
  induction xs with
  | nil =>
    simp [countS, countCS]
  | cons o xs ih =>
    simp only [countS, countCS] at ih ⊢
    simp at ih
    cases o with
    | mk c s =>
      cases c <;> cases s <;> simp [List.countP_cons]
      all_goals (try rw [ih])
      all_goals omega

/-- Lemma 2: the accuracy gap is the count of flips minus the count of rescues. -/
theorem lemma2_count_gap (xs : List Obs) :
    (countC xs true : Int) - (countS xs true : Int) =
      (countCS xs true false : Int) - (countCS xs false true : Int) := by
  rw [countC_split xs, countS_split xs]
  omega

/-- The same identity multiplied by the number of canonically correct items.
`countCS true false` is the flip count; `countCS false true` is the rescued count. -/
theorem lemma2_gap_times_correct (xs : List Obs) :
    ((countC xs true : Int) - (countS xs true : Int)) * (countC xs true : Int) =
      (countCS xs true false : Int) * (countC xs true : Int) -
        (countCS xs false true : Int) * (countC xs true : Int) := by
  rw [lemma2_count_gap xs, Int.sub_mul]

def obsTT : Obs := { c := true, s := true }
def obsTF : Obs := { c := true, s := false }
def obsFT : Obs := { c := false, s := true }
def obsFF : Obs := { c := false, s := false }

def popStable : List Obs :=
  List.replicate 60 obsTT ++ List.replicate 40 obsFF

def popFlip : List Obs :=
  List.replicate 30 obsTT ++ List.replicate 30 obsTF ++
    List.replicate 30 obsFT ++ List.replicate 10 obsFF

theorem lemma2_stable_counts :
    popStable.length = 100 ∧
      countC popStable true = 60 ∧
      countS popStable true = 60 ∧
      countCS popStable true false = 0 := by
  simp [popStable, countC, countS, countCS, obsTT, obsFF,
    List.length_append, List.length_replicate, List.countP_append, List.countP_replicate]

theorem lemma2_flip_counts :
    popFlip.length = 100 ∧
      countC popFlip true = 60 ∧
      countS popFlip true = 60 ∧
      countCS popFlip true false = 30 ∧
      countC popFlip true = 60 := by
  simp [popFlip, countC, countS, countCS, obsTT, obsTF, obsFT, obsFF,
    List.length_append, List.length_replicate, List.countP_append, List.countP_replicate]

/-- Two populations of 100 items, both with accuracy 60/100.
One has SF numerator 0; the other has SF numerator 30 out of 60.
Equal accuracy therefore does not force SF to be zero. -/
theorem lemma2_same_accuracy_different_sf :
    popStable.length = 100 ∧ popFlip.length = 100 ∧
      countC popStable true = 60 ∧ countS popStable true = 60 ∧
      countC popFlip true = 60 ∧ countS popFlip true = 60 ∧
      countCS popStable true false = 0 ∧
      countCS popFlip true false = 30 ∧
      (countC popFlip true : Int) - countS popFlip true = 0 ∧
      countCS popFlip true false ≠ 0 := by
  obtain ⟨hlenS, hcS, hsS, hsfS⟩ := lemma2_stable_counts
  obtain ⟨hlenF, hcF, hsF, hsfF, _⟩ := lemma2_flip_counts
  refine ⟨hlenS, hlenF, hcS, hsS, hcF, hsF, hsfS, hsfF, ?_, ?_⟩
  · omega
  · omega

end StructRobust
