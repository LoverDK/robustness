import StructRobust.Basic

namespace StructRobust

theorem contains_true_iff (as : List Atom) (a : Atom) :
    as.contains a = true ↔ a ∈ as :=
  List.elem_iff

theorem premisesHold_iff (known : List Atom) (r : Rule) :
    premisesHold known r = true ↔ ∀ p, p ∈ r.ps → p ∈ known := by
  unfold premisesHold
  simp only [List.all_eq_true]
  constructor
  · intro h p hp
    exact (contains_true_iff known p).1 (h p hp)
  · intro h p hp
    exact (contains_true_iff known p).2 (h p hp)

theorem addHead_superset (known : List Atom) (r : Rule) {a : Atom}
    (ha : a ∈ known) : a ∈ addHead known r := by
  unfold addHead
  split
  · split
    · exact ha
    · exact List.mem_cons_of_mem _ ha
  · exact ha

theorem addHead_fires (known : List Atom) (r : Rule)
    (hp : ∀ p, p ∈ r.ps → p ∈ known) : r.hd ∈ addHead known r := by
  unfold addHead
  split
  · rename_i _
    split
    · rename_i hin
      exact hin
    · exact List.mem_cons_self
  · rename_i hnot
    exact False.elim (hnot hp)

theorem fold_superset (rs : List Rule) :
    ∀ (known : List Atom) {a}, a ∈ known → a ∈ rs.foldl addHead known := by
  induction rs with
  | nil =>
    intro known a ha
    simpa using ha
  | cons r rs ih =>
    intro known a ha
    simp only [List.foldl_cons]
    exact ih (addHead known r) (addHead_superset known r ha)

theorem fold_fires (rs : List Rule) (known : List Atom) (r : Rule)
    (hr : r ∈ rs) (hp : ∀ p, p ∈ r.ps → p ∈ known) :
    r.hd ∈ rs.foldl addHead known := by
  induction rs generalizing known with
  | nil =>
    cases hr
  | cons r0 rs ih =>
    simp only [List.foldl_cons]
    simp only [List.mem_cons] at hr
    cases hr with
    | inl heq =>
      subst heq
      exact fold_superset rs (addHead known r) (addHead_fires known r hp)
    | inr hr =>
      exact ih (addHead known r0) hr
        (fun p hpm => addHead_superset known r0 (hp p hpm))

theorem closure_succ_fold (rules : List Rule) (k : Nat) :
    ∀ facts, closure (k + 1) facts rules = foldRules (closure k facts rules) rules := by
  induction k with
  | zero =>
    intro facts
    simp [closure, foldRules]
  | succ k ih =>
    intro facts
    have hdef :
        closure ((k + 1) + 1) facts rules =
          closure (k + 1) (foldRules facts rules) rules := rfl
    rw [hdef, ih (foldRules facts rules)]
    simp [closure]

theorem closure_step_superset (rules : List Rule) (k : Nat) (facts : List Atom) :
    ∀ a, a ∈ closure k facts rules → a ∈ closure (k + 1) facts rules := by
  intro a ha
  rw [closure_succ_fold rules k facts]
  exact fold_superset rules (closure k facts rules) ha

theorem closure_le (rules : List Rule) {k k' : Nat} (hk : k ≤ k') (facts : List Atom) :
    ∀ a, a ∈ closure k facts rules → a ∈ closure k' facts rules := by
  induction hk with
  | refl =>
    intro a ha
    exact ha
  | step _ ih =>
    intro a ha
    exact closure_step_superset rules _ facts a (ih a ha)

theorem closure_add (rules : List Rule) (n m : Nat) :
    ∀ facts, closure (n + m) facts rules = closure m (closure n facts rules) rules := by
  induction n with
  | zero =>
    intro facts
    simp [closure]
  | succ n ih =>
    intro facts
    have hsum : (n + 1) + m = (n + m) + 1 := Nat.succ_add n m
    rw [hsum]
    have hdef :
        closure ((n + m) + 1) facts rules =
          closure (n + m) (foldRules facts rules) rules := rfl
    rw [hdef, ih (foldRules facts rules)]
    have hstep :
        closure (n + 1) facts rules = closure n (foldRules facts rules) rules := rfl
    rw [hstep]

theorem uniformPremises (facts : List Atom) (rules : List Rule) (ps : List Atom)
    (h : ∀ p, p ∈ ps → ∃ k, p ∈ closure k facts rules) :
    ∃ k, ∀ p, p ∈ ps → p ∈ closure k facts rules := by
  induction ps with
  | nil =>
    exact ⟨0, fun p hp => nomatch hp⟩
  | cons a as ih =>
    match h a List.mem_cons_self with
    | ⟨k1, hk1⟩ =>
      match ih (fun p hp => h p (List.mem_cons_of_mem a hp)) with
      | ⟨k2, hk2⟩ =>
        refine ⟨max k1 k2, ?_⟩
        intro p hp
        cases List.mem_cons.mp hp with
        | inl heq =>
          subst heq
          exact closure_le rules (Nat.le_max_left k1 k2) facts _ hk1
        | inr hp =>
          exact closure_le rules (Nat.le_max_right k1 k2) facts p (hk2 p hp)

theorem derives_mem (facts : List Atom) (rules : List Rule) {a : Atom}
    (h : Derives facts rules a) : ∃ k, a ∈ closure k facts rules := by
  induction h with
  | ofFact hf =>
    exact ⟨0, by simpa [closure] using hf⟩
  | ofRule hr hp ih =>
    match uniformPremises facts rules _ ih with
    | ⟨k, hk⟩ =>
      refine ⟨k + 1, ?_⟩
      rw [closure_succ_fold rules k facts]
      exact fold_fires rules (closure k facts rules) _ hr hk

theorem addHead_sound {facts known : List Atom} {rules : List Rule} {r : Rule}
    (hknown : ∀ a, a ∈ known → Derives facts rules a) (hr : r ∈ rules) {a : Atom}
    (ha : a ∈ addHead known r) : Derives facts rules a := by
  unfold addHead at ha
  split at ha
  · rename_i hprem
    split at ha
    · exact hknown a ha
    · cases ha with
      | head =>
        apply Derives.ofRule hr
        intro p hp
        exact hknown p (hprem p hp)
      | tail _ ha =>
        exact hknown a ha
  · exact hknown a ha

theorem mem_fold_sound (facts : List Atom) (rules : List Rule) (known : List Atom) (rs : List Rule)
    (hknown : ∀ a, a ∈ known → Derives facts rules a)
    (hsub : ∀ r, r ∈ rs → r ∈ rules) :
    ∀ a, a ∈ rs.foldl addHead known → Derives facts rules a := by
  induction rs generalizing known with
  | nil =>
    intro a ha
    exact hknown a (by simpa using ha)
  | cons r rs ih =>
    intro a ha
    simp only [List.foldl_cons] at ha
    exact ih (addHead known r)
      (fun b hb => addHead_sound hknown (hsub r List.mem_cons_self) hb)
      (fun r' hr' => hsub r' (List.mem_cons_of_mem r hr')) a ha

theorem derives_widen {facts facts' : List Atom} {rules : List Rule}
    (hbase : ∀ b, b ∈ facts' → Derives facts rules b) {a : Atom}
    (h : Derives facts' rules a) : Derives facts rules a := by
  induction h with
  | ofFact hf =>
    exact hbase _ hf
  | ofRule hr hp ih =>
    apply Derives.ofRule hr
    intro p hpm
    exact ih p hpm

theorem mem_closure_sound (rules : List Rule) (k : Nat) :
    ∀ facts a, a ∈ closure k facts rules → Derives facts rules a := by
  induction k with
  | zero =>
    intro facts a ha
    exact Derives.ofFact (by simpa [closure] using ha)
  | succ k ih =>
    intro facts a ha
    have ha' : a ∈ closure k (foldRules facts rules) rules := by
      simpa [closure] using ha
    have hfold : ∀ b, b ∈ foldRules facts rules → Derives facts rules b :=
      mem_fold_sound facts rules facts rules (fun b hb => Derives.ofFact hb) (fun r hr => hr)
    exact derives_widen hfold (ih (foldRules facts rules) a ha')

theorem contains_eq_of_same {k1 k2 : List Atom} (h : sameAtoms k1 k2) (a : Atom) :
    k1.contains a = k2.contains a := by
  cases h1 : k1.contains a with
  | false =>
    cases h2 : k2.contains a with
    | false => rfl
    | true =>
      have hm : a ∈ k2 := (contains_true_iff k2 a).1 h2
      have hm1 : a ∈ k1 := (h a).2 hm
      have h1' : k1.contains a = true := (contains_true_iff k1 a).2 hm1
      rw [h1] at h1'
      cases h1'
  | true =>
    cases h2 : k2.contains a with
    | true => rfl
    | false =>
      have hm : a ∈ k1 := (contains_true_iff k1 a).1 h1
      have hm2 : a ∈ k2 := (h a).1 hm
      have h2' : k2.contains a = true := (contains_true_iff k2 a).2 hm2
      rw [h2] at h2'
      cases h2'

theorem premisesHold_eq_of_same {k1 k2 : List Atom} (h : sameAtoms k1 k2) (r : Rule) :
    premisesHold k1 r = premisesHold k2 r := by
  simp only [premisesHold]
  induction r.ps with
  | nil => rfl
  | cons p ps ih =>
    simp only [List.all_cons, contains_eq_of_same h p, ih]

theorem addHead_congr {k1 k2 : List Atom} (h : sameAtoms k1 k2) (r : Rule) :
    sameAtoms (addHead k1 r) (addHead k2 r) := by
  intro a
  have hprem : (∀ p, p ∈ r.ps → p ∈ k1) ↔ (∀ p, p ∈ r.ps → p ∈ k2) := by
    constructor
    · intro h1 p hp
      exact (h p).1 (h1 p hp)
    · intro h2 p hp
      exact (h p).2 (h2 p hp)
  have hhd : r.hd ∈ k1 ↔ r.hd ∈ k2 := h r.hd
  unfold addHead
  by_cases hp2 : ∀ p, p ∈ r.ps → p ∈ k2
  · have hp1 : ∀ p, p ∈ r.ps → p ∈ k1 := hprem.2 hp2
    by_cases hm2 : r.hd ∈ k2
    · have hm1 : r.hd ∈ k1 := hhd.2 hm2
      rw [ite_eq_left hp1, ite_eq_left hp2, ite_eq_left hm1, ite_eq_left hm2]
      exact h a
    · have hm1 : r.hd ∉ k1 := fun hin => hm2 (hhd.1 hin)
      rw [ite_eq_left hp1, ite_eq_left hp2, ite_eq_right hm1, ite_eq_right hm2]
      simp only [List.mem_cons]
      constructor
      · intro ha
        cases ha with
        | inl heq => exact Or.inl heq
        | inr ha => exact Or.inr ((h a).1 ha)
      · intro ha
        cases ha with
        | inl heq => exact Or.inl heq
        | inr ha => exact Or.inr ((h a).2 ha)
  · have hp1 : ¬ ∀ p, p ∈ r.ps → p ∈ k1 := fun h1 => hp2 (hprem.1 h1)
    rw [ite_eq_right hp1, ite_eq_right hp2]
    exact h a

theorem fold_congr (rs : List Rule) :
    ∀ {f1 f2 : List Atom}, sameAtoms f1 f2 →
      sameAtoms (rs.foldl addHead f1) (rs.foldl addHead f2) := by
  induction rs with
  | nil =>
    intro f1 f2 h
    simpa using h
  | cons r rs ih =>
    intro f1 f2 h
    simp only [List.foldl_cons]
    exact ih (addHead_congr h r)

theorem closure_congr (rules : List Rule) (m : Nat) :
    ∀ {f1 f2 : List Atom}, sameAtoms f1 f2 →
      sameAtoms (closure m f1 rules) (closure m f2 rules) := by
  induction m with
  | zero =>
    intro f1 f2 h
    simpa [closure] using h
  | succ m ih =>
    intro f1 f2 h a
    simp only [closure]
    exact ih (fold_congr rules h) a

theorem addHead_mem_of_closed {facts known : List Atom} {rules : List Rule} {r : Rule}
    (hc : closed facts rules) (hr : r ∈ rules)
    (hknown : ∀ a, a ∈ known → a ∈ facts) {a : Atom}
    (ha : a ∈ addHead known r) : a ∈ facts := by
  unfold addHead at ha
  split at ha
  · rename_i hprem
    split at ha
    · exact hknown a ha
    · cases ha with
      | head =>
        exact hc r hr (fun p hp => hknown p (hprem p hp))
      | tail _ ha =>
        exact hknown a ha
  · exact hknown a ha

theorem fold_mem_of_closed {facts : List Atom} {rules : List Rule}
    (hc : closed facts rules) (rs : List Rule)
    (hsub : ∀ r, r ∈ rs → r ∈ rules) :
    ∀ known, (∀ a, a ∈ known → a ∈ facts) →
      ∀ a, a ∈ rs.foldl addHead known → a ∈ facts := by
  induction rs with
  | nil =>
    intro known hknown a ha
    exact hknown a (by simpa using ha)
  | cons r rs ih =>
    intro known hknown a ha
    simp only [List.foldl_cons] at ha
    exact ih (fun r' hr' => hsub r' (List.mem_cons_of_mem r hr'))
      (addHead known r)
      (fun b hb => addHead_mem_of_closed hc (hsub r List.mem_cons_self) hknown hb)
      a ha

theorem fold_same_of_closed {facts : List Atom} {rules : List Rule}
    (hc : closed facts rules) : sameAtoms (foldRules facts rules) facts := by
  intro a
  constructor
  · intro ha
    exact fold_mem_of_closed hc rules (fun r hr => hr) facts (fun b hb => hb) a ha
  · intro ha
    exact fold_superset rules facts ha

theorem closure_same_of_closed {facts : List Atom} {rules : List Rule}
    (hc : closed facts rules) (m : Nat) :
    sameAtoms (closure m facts rules) facts := by
  induction m with
  | zero =>
    intro a
    simp [closure]
  | succ m ih =>
    intro a
    simp only [closure]
    have hfold : sameAtoms (foldRules facts rules) facts := fold_same_of_closed hc
    have hcl : sameAtoms (closure m (foldRules facts rules) rules) (closure m facts rules) :=
      closure_congr rules m hfold
    constructor
    · intro ha
      exact (ih a).1 ((hcl a).1 ha)
    · intro ha
      exact (hcl a).2 ((ih a).2 ha)

theorem closed_congr {a b : List Atom} {rules : List Rule}
    (h : sameAtoms a b) (hc : closed a rules) : closed b rules := by
  intro r hr hp
  exact (h r.hd).1 (hc r hr (fun p hpm => (h p).2 (hp p hpm)))

theorem isClosed_iff (known : List Atom) (rules : List Rule) :
    isClosed known rules = true ↔ closed known rules := by
  unfold isClosed closed
  simp only [List.all_eq_true]
  constructor
  · intro h r hr hp
    have hdisj := h r hr
    have hpre : premisesHold known r = true := (premisesHold_iff known r).2 hp
    simp [hpre] at hdisj
    exact hdisj
  · intro h r hr
    cases hpre : premisesHold known r with
    | false =>
      simp
    | true =>
      have hc : r.hd ∈ known := h r hr ((premisesHold_iff known r).1 hpre)
      simp
      exact hc

theorem exists_violation (known : List Atom) :
    ∀ rules, isClosed known rules = false →
      ∃ r, r ∈ rules ∧ premisesHold known r = true ∧ known.contains r.hd = false := by
  intro rules
  induction rules with
  | nil =>
    intro h
    simp [isClosed] at h
  | cons r rs ih =>
    intro h
    simp only [isClosed, List.all_cons] at h
    cases h1 : (!premisesHold known r || known.contains r.hd) with
    | false =>
      cases hp : premisesHold known r with
      | false =>
        simp [hp] at h1
      | true =>
        cases hc : known.contains r.hd with
        | false =>
          exact ⟨r, List.mem_cons_self, hp, hc⟩
        | true =>
          exfalso
          have htrue : (!premisesHold known r || known.contains r.hd) = true := by
            rw [hp, hc]
            rfl
          rw [h1] at htrue
          cases htrue
    | true =>
      have htail : isClosed known rs = false := by
        rw [h1, Bool.true_and] at h
        simpa [isClosed] using h
      match ih htail with
      | ⟨r', hr', hp', hc'⟩ =>
        exact ⟨r', List.mem_cons_of_mem r hr', hp', hc'⟩

theorem not_closed_bool {facts : List Atom} {rules : List Rule}
    (h : ¬ closed facts rules) : isClosed facts rules = false := by
  cases hclosed : isClosed facts rules with
  | true =>
    exact False.elim (h ((isClosed_iff facts rules).1 hclosed))
  | false =>
    rfl

theorem countP_le_of_imp {α : Type} (wide narrow : α → Bool) (l : List α)
    (himp : ∀ a, a ∈ l → narrow a = true → wide a = true) :
    l.countP narrow ≤ l.countP wide := by
  induction l with
  | nil =>
    simp [List.countP_nil]
  | cons a as ih =>
    have ih' := ih (fun b hb hn => himp b (List.mem_cons_of_mem a hb) hn)
    cases hn : narrow a with
    | false =>
      cases hw : wide a with
      | false =>
        rw [List.countP_cons, List.countP_cons]
        simp [hn, hw]
        exact ih'
      | true =>
        rw [List.countP_cons, List.countP_cons]
        simp [hn, hw]
        exact Nat.le_trans ih' (Nat.le_succ _)
    | true =>
      cases hw : wide a with
      | false =>
        have hwide : wide a = true := himp a List.mem_cons_self hn
        rw [hw] at hwide
        cases hwide
      | true =>
        rw [List.countP_cons, List.countP_cons]
        simp [hn, hw]
        exact ih'

theorem countP_lt_of_imp {α : Type} (wide narrow : α → Bool) (l : List α)
    (himp : ∀ a, a ∈ l → narrow a = true → wide a = true)
    (hex : ∃ a, a ∈ l ∧ wide a = true ∧ narrow a = false) :
    l.countP narrow < l.countP wide := by
  induction l with
  | nil =>
    match hex with
    | ⟨_, ha, _, _⟩ => cases ha
  | cons a as ih =>
    match hex with
    | ⟨b, hb, hw, hn⟩ =>
      cases List.mem_cons.mp hb with
      | inl heq =>
        subst heq
        have hle :=
          countP_le_of_imp wide narrow as
            (fun c hc hcn => himp c (List.mem_cons_of_mem _ hc) hcn)
        rw [List.countP_cons, List.countP_cons]
        simp [hw, hn]
        exact Nat.lt_succ_of_le hle
      | inr hb =>
        have ih' :=
          ih (fun c hc hcn => himp c (List.mem_cons_of_mem a hc) hcn)
            ⟨b, hb, hw, hn⟩
        cases hna : narrow a with
        | false =>
          cases hwa : wide a with
          | false =>
            rw [List.countP_cons, List.countP_cons]
            simp [hna, hwa]
            exact ih'
          | true =>
            rw [List.countP_cons, List.countP_cons]
            simp [hna, hwa]
            exact Nat.lt_succ_of_lt ih'
        | true =>
          cases hwa : wide a with
          | false =>
            have hwide : wide a = true := himp a List.mem_cons_self hna
            rw [hwa] at hwide
            cases hwide
          | true =>
            rw [List.countP_cons, List.countP_cons]
            simp [hna, hwa]
            exact ih'

theorem pot_lt_of_new_head {facts facts' : List Atom} {rules : List Rule}
    (hsub : ∀ a, a ∈ facts → a ∈ facts') (r : Rule) (hr : r ∈ rules)
    (hnew : r.hd ∉ facts) (hgot : r.hd ∈ facts') :
    pot facts' rules < pot facts rules := by
  have himp : ∀ s, s ∈ rules → headMissing facts' s = true → headMissing facts s = true := by
    intro s _ hs
    have hnin' : s.hd ∉ facts' := of_decide_eq_true (by simpa [headMissing] using hs)
    have hnin : s.hd ∉ facts := fun hin => hnin' (hsub s.hd hin)
    simpa [headMissing] using (decide_eq_true_iff.mpr hnin)
  have hex : ∃ s, s ∈ rules ∧ headMissing facts s = true ∧ headMissing facts' s = false := by
    refine ⟨r, hr, ?_, ?_⟩
    · simpa [headMissing] using (decide_eq_true_iff.mpr hnew)
    · simp only [headMissing, decide_eq_false_iff_not]
      intro hnin
      exact hnin hgot
  simpa [pot] using countP_lt_of_imp (headMissing facts) (headMissing facts') rules himp hex

theorem pot_lt_of_not_closed {facts : List Atom} {rules : List Rule}
    (h : ¬ closed facts rules) :
    pot (foldRules facts rules) rules < pot facts rules := by
  match exists_violation facts rules (not_closed_bool h) with
  | ⟨r, hr, hp, hc⟩ =>
    have hprem : ∀ p, p ∈ r.ps → p ∈ facts := (premisesHold_iff facts r).1 hp
    have hnew : r.hd ∉ facts := by
      intro hin
      have ht : facts.contains r.hd = true := (contains_true_iff facts r.hd).2 hin
      rw [hc] at ht
      cases ht
    have hgot : r.hd ∈ foldRules facts rules := fold_fires rules facts r hr hprem
    exact pot_lt_of_new_head (fun a ha => fold_superset rules facts ha) r hr hnew hgot

theorem pot_zero_head {facts : List Atom} {rules : List Rule} {r : Rule}
    (hz : pot facts rules = 0) (hr : r ∈ rules) : r.hd ∈ facts := by
  have hnone : ∀ s, s ∈ rules → ¬ headMissing facts s = true := by
    simpa [pot, List.countP_eq_zero] using hz
  have hmiss : ¬ headMissing facts r = true := hnone r hr
  by_cases hin : r.hd ∈ facts
  · exact hin
  · exact False.elim (hmiss (by simpa [headMissing] using (decide_eq_true_iff.mpr hin)))

theorem closure_closed_of_ge (rules : List Rule) (m : Nat) :
    ∀ facts, pot facts rules ≤ m → closed (closure m facts rules) rules := by
  induction m with
  | zero =>
    intro facts hm
    have hz : pot facts rules = 0 := Nat.eq_zero_of_le_zero hm
    intro r hr _
    exact pot_zero_head hz hr
  | succ m ih =>
    intro facts hm
    cases hclosed : isClosed facts rules with
    | true =>
      have hc : closed facts rules := (isClosed_iff facts rules).1 hclosed
      exact closed_congr (fun x => (closure_same_of_closed hc (m + 1) x).symm) hc
    | false =>
      have hnot : ¬ closed facts rules := by
        intro hc
        have ht : isClosed facts rules = true := (isClosed_iff facts rules).2 hc
        rw [hclosed] at ht
        cases ht
      have hlt : pot (foldRules facts rules) rules < pot facts rules :=
        pot_lt_of_not_closed hnot
      have hle : pot (foldRules facts rules) rules ≤ m :=
        Nat.le_of_succ_le_succ (Nat.le_trans (Nat.succ_le_of_lt hlt) hm)
      simpa [closure] using ih (foldRules facts rules) hle

theorem pot_le_length (known : List Atom) (rules : List Rule) :
    pot known rules ≤ rules.length :=
  List.countP_le_length (p := headMissing known) (l := rules)

theorem derives_mem_pot (facts : List Atom) (rules : List Rule) {a : Atom}
    (h : Derives facts rules a) : a ∈ closure (pot facts rules) facts rules := by
  match derives_mem facts rules h with
  | ⟨k, hk⟩ =>
    let p := pot facts rules
    have hp : p ≤ max k p := Nat.le_max_right k p
    have hkmax : k ≤ max k p := Nat.le_max_left k p
    have ha : a ∈ closure (max k p) facts rules :=
      closure_le rules hkmax facts a hk
    have hcl : closed (closure p facts rules) rules :=
      closure_closed_of_ge rules p facts (Nat.le_refl p)
    have hsum : p + (max k p - p) = max k p := Nat.add_sub_of_le hp
    have ha2 : a ∈ closure (max k p - p) (closure p facts rules) rules := by
      have hmem : a ∈ closure (p + (max k p - p)) facts rules := by
        simpa [hsum] using ha
      simpa [closure_add rules p (max k p - p) facts] using hmem
    exact (closure_same_of_closed hcl (max k p - p) a).1 ha2

/-- When the fuel is at least the number of rules, fuel-bounded forward chaining
is exactly definite-clause derivability. -/
theorem closure_fuel_iff_derives (facts : List Atom) (rules : List Rule) (a : Atom)
    {fuel : Nat} (hfuel : rules.length ≤ fuel) :
    a ∈ closure fuel facts rules ↔ Derives facts rules a := by
  constructor
  · intro h
    exact mem_closure_sound rules fuel facts a h
  · intro h
    exact closure_le rules (Nat.le_trans (pot_le_length facts rules) hfuel) facts a
      (derives_mem_pot facts rules h)

theorem derives_rules_congr {facts : List Atom} {rules rules' : List Rule} {a : Atom}
    (h : sameRules rules rules') :
    Derives facts rules a ↔ Derives facts rules' a := by
  constructor
  · intro hd
    induction hd with
    | ofFact hf =>
      exact Derives.ofFact hf
    | ofRule hr hp ih =>
      apply Derives.ofRule ((h _).1 hr)
      intro p hpm
      exact ih p hpm
  · intro hd
    induction hd with
    | ofFact hf =>
      exact Derives.ofFact hf
    | ofRule hr hp ih =>
      apply Derives.ofRule ((h _).2 hr)
      intro p hpm
      exact ih p hpm

end StructRobust

