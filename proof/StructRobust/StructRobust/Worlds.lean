import StructRobust.Counting

namespace StructRobust

structure FiniteWorld where
  seen : List Obs
  unseenCanon : List Bool

def worldSame : FiniteWorld where
  seen := [obsTT, obsTT]
  unseenCanon := [true, true]

def worldDiff : FiniteWorld where
  seen := [obsTF, obsTT]
  unseenCanon := [true, true]

def canonSeen (w : FiniteWorld) : Nat :=
  countC w.seen true

def canonUnseen (w : FiniteWorld) : Nat :=
  w.unseenCanon.countP (fun b => b)

def sfNum (w : FiniteWorld) : Nat :=
  countCS w.seen true false

def sfDen (w : FiniteWorld) : Nat :=
  countC w.seen true

/-- Lemma 3: two finite worlds have the same canonical seen and unseen accuracies,
hence the same transfer gap, but different flip counts on the seen graph. -/
theorem lemma3_same_transfer_different_sf :
    worldSame.seen.length = worldSame.unseenCanon.length ∧
      canonSeen worldSame = canonUnseen worldSame ∧
      worldDiff.seen.length = worldDiff.unseenCanon.length ∧
      canonSeen worldDiff = canonUnseen worldDiff ∧
      ((canonSeen worldSame : Int) - canonUnseen worldSame) =
        ((canonSeen worldDiff : Int) - canonUnseen worldDiff) ∧
      sfNum worldSame = 0 ∧ sfDen worldSame = 2 ∧
      sfNum worldDiff = 1 ∧ sfDen worldDiff = 2 ∧
      sfNum worldSame ≠ sfNum worldDiff := by
  simp [worldSame, worldDiff, canonSeen, canonUnseen, sfNum, sfDen,
    countC, countCS, obsTT, obsTF, List.countP_nil]

end StructRobust
