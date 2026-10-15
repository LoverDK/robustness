/-
Atoms, definite clauses, and the fuel-bounded forward chaining operator.
Finite sets are represented by lists; membership is `∈`.
-/

namespace StructRobust

abbrev Atom := Nat

/-- A definite clause: premises together with one conclusion. -/
structure Rule where
  ps : List Atom
  hd : Atom
deriving DecidableEq, Repr

/-- A task is a finite list of facts, a finite list of rules, and a query. -/
structure Task where
  facts : List Atom
  rules : List Rule
  query : Atom
deriving Repr

def premisesHold (known : List Atom) (r : Rule) : Bool :=
  r.ps.all (fun p => known.contains p)

def addHead (known : List Atom) (r : Rule) : List Atom :=
  if ∀ p, p ∈ r.ps → p ∈ known then
    if r.hd ∈ known then
      known
    else
      r.hd :: known
  else
    known

/-- One left-to-right pass: if a rule's premises are already known, add its head. -/
def foldRules (known : List Atom) (rules : List Rule) : List Atom :=
  rules.foldl addHead known

/-- Fuel-bounded closure. Each unit of fuel is one pass over the rule list. -/
def closure : Nat → List Atom → List Rule → List Atom
  | 0, known, _ => known
  | fuel + 1, known, rules => closure fuel (foldRules known rules) rules

/-- The definite-clause closure, as an inductive derivation. -/
inductive Derives (facts : List Atom) (rules : List Rule) : Atom → Prop
  | ofFact {a : Atom} (h : a ∈ facts) : Derives facts rules a
  | ofRule {r : Rule} (hr : r ∈ rules)
      (hp : ∀ p, p ∈ r.ps → Derives facts rules p) : Derives facts rules r.hd

def sameAtoms (a b : List Atom) : Prop :=
  ∀ x, x ∈ a ↔ x ∈ b

def sameRules (a b : List Rule) : Prop :=
  ∀ r, r ∈ a ↔ r ∈ b

/-- Deductively closed: every rule whose premises are known already has its head known. -/
def closed (known : List Atom) (rules : List Rule) : Prop :=
  ∀ r, r ∈ rules → (∀ p, p ∈ r.ps → p ∈ known) → r.hd ∈ known

/-- Boolean test for `closed`, so case splits stay constructive. -/
def isClosed (known : List Atom) (rules : List Rule) : Bool :=
  rules.all (fun r => !premisesHold known r || known.contains r.hd)

/-- Rules whose head is not yet known. Its length bounds the number of productive passes. -/
def headMissing (known : List Atom) (r : Rule) : Bool :=
  decide (r.hd ∉ known)

def pot (known : List Atom) (rules : List Rule) : Nat :=
  rules.countP (headMissing known)

/-- One layer of sentence wrapping. The denotation is the rule inside. -/
structure Phrase where
  wrap : Nat
  rule : Rule
deriving DecidableEq, Repr

def denote (p : Phrase) : Rule :=
  p.rule

def rephrase (p : Phrase) (wrap' : Nat) : Phrase :=
  { p with wrap := wrap' }

def extract (ps : List Phrase) : List Rule :=
  ps.map denote

end StructRobust
