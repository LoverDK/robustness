namespace StructRobust

def invCount : List Nat → Nat
  | [] => 0
  | x :: xs => xs.countP (fun y => decide (y < x)) + invCount xs

def cross (xs ys : List Nat) : Nat :=
  match xs with
  | [] => 0
  | x :: xs => ys.countP (fun y => decide (y < x)) + cross xs ys

/-- Canonical order of `m` rule indices: `0, 1, ..., m - 1`. -/
def canonOrder : Nat → List Nat
  | 0 => []
  | m + 1 => canonOrder m ++ [m]

/-- Full reversal: `m - 1, ..., 1, 0`. -/
def revOrder : Nat → List Nat
  | 0 => []
  | m + 1 => m :: revOrder m

theorem inv_append (xs ys : List Nat) :
    invCount (xs ++ ys) = invCount xs + invCount ys + cross xs ys := by
  induction xs with
  | nil =>
    simp [invCount, cross]
  | cons x xs ih =>
    simp [invCount, cross, List.countP_append, ih, Nat.add_assoc, Nat.add_left_comm]

theorem cross_append (xs ys zs : List Nat) :
    cross (xs ++ ys) zs = cross xs zs + cross ys zs := by
  induction xs with
  | nil =>
    simp [cross]
  | cons x xs ih =>
    simp [cross, ih, Nat.add_assoc]

theorem cross_lt_singleton (xs : List Nat) (m : Nat) (h : ∀ a, a ∈ xs → a < m) :
    cross xs [m] = 0 := by
  induction xs with
  | nil =>
    rfl
  | cons a xs ih =>
    have ha : a < m := h a List.mem_cons_self
    simp [cross, List.countP_cons, List.countP_nil,
      ih (fun b hb => h b (List.mem_cons_of_mem _ hb)),
      decide_eq_false (Nat.lt_asymm ha)]

theorem canon_mem_lt (m : Nat) : ∀ a, a ∈ canonOrder m → a < m := by
  induction m with
  | zero =>
    intro a ha
    cases ha
  | succ m ih =>
    intro a ha
    simp [canonOrder, List.mem_append, List.mem_singleton] at ha
    cases ha with
    | inl ha =>
      exact Nat.lt_succ_of_lt (ih a ha)
    | inr heq =>
      rw [heq]
      exact Nat.lt_succ_self _

theorem inv_singleton (n : Nat) : invCount [n] = 0 := by
  simp [invCount, List.countP_nil]

/-- Lemma 5: canonical order has no inversions. -/
theorem lemma5_canon_inversions (m : Nat) : invCount (canonOrder m) = 0 := by
  induction m with
  | zero =>
    rfl
  | succ m ih =>
    have hstep : canonOrder (m + 1) = canonOrder m ++ [m] := rfl
    rw [hstep, inv_append, ih, inv_singleton, cross_lt_singleton _ _ (canon_mem_lt m)]

theorem rev_length (m : Nat) : (revOrder m).length = m := by
  induction m with
  | zero =>
    rfl
  | succ m ih =>
    simp [revOrder, ih]

theorem rev_mem_lt {m a : Nat} (h : a ∈ revOrder m) : a < m := by
  induction m generalizing a with
  | zero =>
    simp [revOrder] at h
  | succ m ih =>
    simp [revOrder, List.mem_cons] at h
    cases h with
    | inl heq =>
      rw [heq]
      exact Nat.lt_succ_self _
    | inr h =>
      exact Nat.lt_succ_of_lt (ih h)

theorem rev_count_lt (m : Nat) :
    (revOrder m).countP (fun y => decide (y < m)) = m := by
  have hall :
      (revOrder m).countP (fun y => decide (y < m)) = (revOrder m).length :=
    (List.countP_eq_length).mpr (fun a ha => decide_eq_true (rev_mem_lt ha))
  rw [hall, rev_length]

theorem two_m_identity (m : Nat) : 2 * m + m * (m - 1) = (m + 1) * m := by
  cases m with
  | zero =>
    simp
  | succ m =>
    rw [Nat.succ_sub_one]
    calc
      2 * (m + 1) + (m + 1) * m
          = (m + 1) * 2 + (m + 1) * m := by rw [Nat.mul_comm]
      _ = (m + 1) * (2 + m) := by rw [← Nat.mul_add]
      _ = (m + 1) * (m + 2) := by rw [Nat.add_comm 2 m]
      _ = (m + 2) * (m + 1) := by rw [Nat.mul_comm]

theorem two_mul_inv_rev (m : Nat) : 2 * invCount (revOrder m) = m * (m - 1) := by
  induction m with
  | zero =>
    simp [revOrder, invCount]
  | succ m ih =>
    rw [revOrder, invCount, rev_count_lt m, Nat.mul_add, ih, two_m_identity m]
    simp

/-- The reversed list is the list reversal of the canonical order. -/
theorem lemma5_rev_eq_reverse (m : Nat) : revOrder m = (canonOrder m).reverse := by
  induction m with
  | zero =>
    rfl
  | succ m ih =>
    simp [revOrder, canonOrder, List.reverse_append, ih]

/-- Lemma 5: full reversal of `m` indices has `m * (m - 1) / 2` inversions. -/
theorem lemma5_reverse_inversions (m : Nat) :
    invCount (revOrder m) = m * (m - 1) / 2 := by
  have h := two_mul_inv_rev m
  have hdiv : 2 * invCount (revOrder m) / 2 = invCount (revOrder m) :=
    Nat.mul_div_cancel_left _ (by decide : 0 < 2)
  rw [← hdiv, h]

theorem lemma5_reverse_two : invCount (revOrder 2) = 1 := by
  decide

theorem lemma5_reverse_three : invCount (revOrder 3) = 3 := by
  decide

/-- Depth-2 families with 2 and 3 core rules do not receive the same reversal dose. -/
theorem lemma5_reverse_two_ne_three :
    invCount (revOrder 2) ≠ invCount (revOrder 3) := by
  rw [lemma5_reverse_two, lemma5_reverse_three]
  decide

end StructRobust
