namespace StructRobust

def interaction (asc ass auc aus : Int) : Int :=
  (asc - ass) - (auc - aus)

/-- Difference, across the two presentations, of the seen-minus-unseen gaps. -/
def transferPresentationDiff (asc ass auc aus : Int) : Int :=
  (ass - aus) - (asc - auc)

/-- Lemma 4: the interaction contrast and the difference of transfer gaps are negatives. -/
theorem lemma4_interaction_opposite (asc ass auc aus : Int) :
    interaction asc ass auc aus = - transferPresentationDiff asc ass auc aus := by
  unfold interaction transferPresentationDiff
  omega

end StructRobust
