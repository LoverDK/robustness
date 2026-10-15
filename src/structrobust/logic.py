"""Task family A: propositional definite-clause reasoning (ProofWriter / SimpleLogic style).

A latent task z = (G, q, y) is a set of facts and rules over atoms, a query atom q,
and the solver-verified label y. A presentation r is an ordering of the same
sentences. Every presentation of z has the same label and the same solver trace.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field

Rule = tuple[tuple[str, ...], str]

ADJ_TRAIN = (
    "red blue green kind nice rough smart quiet young cold big round furry white "
    "gentle bright calm brave clever eager fancy happy jolly lively proud silly "
    "witty zesty bold crisp damp fuzzy grand hollow humble lucky noble plain "
    "rapid sharp shiny sleek soft stiff sturdy sweet"
).split()
ADJ_EVAL = (
    "amber azure coral ivory olive scarlet violet golden silver bronze dusty misty "
    "rusty sandy snowy stormy sunny windy frosty icy muddy rocky salty spicy sour "
    "bitter mellow sober tidy weary wild wise loyal modest polite rude shy stern "
    "tense vivid warm"
).split()
NAMES_TRAIN = "Anne Bob Charlie Dave Erin Fiona Gary Harry Iris Jack".split()
NAMES_EVAL = "Kara Liam Mona Nate Olga Paul Rosa Sam Tina Victor".split()

SEEN_FAMILIES = ("chain", "conj_fact")
UNSEEN_FAMILIES = ("tree", "diamond")
PRESENTATIONS = ("aligned", "reversed", "random", "random_distract")


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def closure(facts: set[str] | frozenset[str], rules: list[Rule]) -> list[tuple[int, Rule]]:
    """Forward chaining in rounds. Returns derivation steps (round, rule) in a
    presentation-independent order (round, then conclusion name)."""
    known = set(facts)
    steps: list[tuple[int, Rule]] = []
    rnd = 0
    while True:
        rnd += 1
        fire = sorted(
            {(tuple(s), t) for s, t in rules if t not in known and all(x in known for x in s)},
            key=lambda r: (r[1], r[0]),
        )
        new: list[Rule] = []
        for s, t in fire:
            if t not in known and all(x in known for x in s):
                new.append((s, t))
        if not new:
            return steps
        seen_t: set[str] = set()
        for s, t in new:
            if t in seen_t:
                continue
            seen_t.add(t)
            steps.append((rnd, (s, t)))
        known.update(seen_t)


def solve(facts: set[str] | frozenset[str], rules: list[Rule], query: str) -> bool:
    if query in facts:
        return True
    return any(t == query for _, (_, t) in closure(facts, rules))


@dataclass
class Latent:
    task_id: str
    family: str
    depth: int
    label: str  # "yes" | "no"
    entity: str
    facts: tuple[str, ...]
    core_rules: tuple[Rule, ...]  # relevant rules, in proof (derivation) order
    other_rules: tuple[Rule, ...]  # decoys and base distractors
    extra_distractors: tuple[Rule, ...]  # only shown in the random_distract presentation
    query: str
    other_slots: tuple[int, ...]  # fixed sentence slots of other_rules among all rules
    vocab: str = "train"
    meta: dict = field(default_factory=dict)

    def all_rules(self, with_extra: bool = False) -> list[Rule]:
        rules = list(self.core_rules) + list(self.other_rules)
        if with_extra:
            rules += list(self.extra_distractors)
        return rules

    def to_json(self) -> dict:
        d = self.__dict__.copy()
        return json.loads(json.dumps(d))


def _proof_order(rules: list[Rule], facts: set[str], query: str) -> list[Rule]:
    """Post-order DFS from the query: premises before the conclusion."""
    by_target: dict[str, Rule] = {}
    for r in rules:
        by_target.setdefault(r[1], r)
    order: list[Rule] = []
    visited: set[str] = set()

    def visit(atom: str) -> None:
        if atom in visited or atom in facts or atom not in by_target:
            return
        visited.add(atom)
        r = by_target[atom]
        for p in r[0]:
            visit(p)
        order.append(r)

    visit(query)
    return order


def make_latent(
    family: str,
    depth: int,
    label: str,
    seed: int,
    vocab: str = "train",
    n_base_distract: int = 2,
    n_extra_distract: int = 6,
) -> Latent:
    rng = random.Random(seed)
    adjs = list(ADJ_TRAIN if vocab == "train" else ADJ_EVAL)
    names = NAMES_TRAIN if vocab == "train" else NAMES_EVAL
    rng.shuffle(adjs)
    pool = iter(adjs)

    def fresh() -> str:
        return next(pool)

    entity = rng.choice(names)
    facts: list[str] = []
    rules: list[Rule] = []
    query: str

    if family == "chain":
        spine = [fresh() for _ in range(depth + 1)]
        facts.append(spine[0])
        rules = [((spine[i],), spine[i + 1]) for i in range(depth)]
        query = spine[-1]
    elif family == "conj_fact":
        spine = [fresh() for _ in range(depth + 1)]
        facts.append(spine[0])
        for i in range(depth):
            if i % 2 == 1:
                f = fresh()
                facts.append(f)
                rules.append(((spine[i], f), spine[i + 1]))
            else:
                rules.append(((spine[i],), spine[i + 1]))
        query = spine[-1]
    elif family == "tree":
        left = [fresh() for _ in range(depth)]
        rlen = max(1, (depth - 1) // 2)
        right = [fresh() for _ in range(rlen + 1)]
        query = fresh()
        facts += [left[0], right[0]]
        rules = [((left[i],), left[i + 1]) for i in range(depth - 1)]
        rules += [((right[i],), right[i + 1]) for i in range(rlen)]
        rules.append(((left[-1], right[-1]), query))
    elif family == "diamond":
        # spine with one fork-and-rejoin: a_m -> x, a_m -> y, (x and y) -> a_{m+1}
        n_spine = max(1, depth - 1)
        spine = [fresh() for _ in range(n_spine + 1)]
        m = rng.randrange(0, n_spine)
        x, y = fresh(), fresh()
        facts.append(spine[0])
        for i in range(n_spine):
            if i == m:
                rules += [((spine[i],), x), ((spine[i],), y), ((x, y), spine[i + 1])]
            else:
                rules.append(((spine[i],), spine[i + 1]))
        query = spine[-1]
    else:
        raise ValueError(family)

    core = _proof_order(rules, set(facts), query)
    assert len(core) == len(rules)
    others: list[Rule] = []
    # A decoy that concludes a core atom from an underivable premise (present for both labels).
    u_extra = fresh()
    core_atoms = [r[1] for r in core]
    others.append(((u_extra,), rng.choice(core_atoms)))
    if label == "no":
        # Break the proof: replace one core rule by a rule with the same conclusion
        # whose premise is underivable. Rule count stays identical across labels.
        k = rng.randrange(len(core))
        u = fresh()
        s, t = core[k]
        new_s = (u,) if len(s) == 1 else (u, s[1])
        core[k] = (new_s, t)
    # Base distractors: outgoing edges from core/fact atoms to fresh atoms (cannot affect q).
    sources = list(facts) + core_atoms[:-1]
    for _ in range(n_base_distract):
        others.append(((rng.choice(sources),), fresh()))
    extra: list[Rule] = []
    g = fresh()
    chain_src = g
    for j in range(n_extra_distract):
        if j % 2 == 0:
            tgt = fresh()
            extra.append(((chain_src,), tgt))
            chain_src = tgt
        else:
            extra.append(((rng.choice(sources),), fresh()))
    extra_facts = [g]

    n_rules = len(core) + len(others)
    other_slots = tuple(sorted(rng.sample(range(n_rules), len(others))))
    all_facts = tuple(facts)
    base_rules = core + others
    truth = solve(set(all_facts), base_rules, query)
    truth_extra = solve(set(all_facts) | set(extra_facts), base_rules + extra, query)
    assert truth == (label == "yes") == truth_extra, (family, depth, label, seed)
    tid = hashlib.sha256(
        json.dumps([family, depth, label, seed, vocab, base_rules, all_facts]).encode()
    ).hexdigest()[:12]
    return Latent(
        task_id=tid,
        family=family,
        depth=depth,
        label=label,
        entity=entity,
        facts=all_facts,
        core_rules=tuple(core),
        other_rules=tuple(others),
        extra_distractors=tuple(extra),
        query=query,
        other_slots=other_slots,
        vocab=vocab,
        meta={"extra_facts": extra_facts},
    )


def order_core(lat: Latent, presentation: str, perm_seed: int | None = None) -> list[int]:
    m = len(lat.core_rules)
    idx = list(range(m))
    if presentation == "aligned":
        return idx
    if presentation == "reversed":
        return idx[::-1]
    rng = random.Random(stable_seed(lat.task_id, presentation, perm_seed))
    for _ in range(20):
        rng.shuffle(idx)
        if m < 2 or idx != list(range(m)):
            break
    return idx


def rule_sequence(lat: Latent, presentation: str, perm_seed: int | None = None,
                  core_order: list[int] | None = None) -> list[tuple[str, int]]:
    """Ordered list of ("core", i) / ("other", j) / ("extra", k) rule references."""
    if core_order is None:
        core_order = order_core(lat, presentation, perm_seed)
    n = len(lat.core_rules) + len(lat.other_rules)
    seq: list[tuple[str, int] | None] = [None] * n
    for j, slot in enumerate(lat.other_slots):
        seq[slot] = ("other", j)
    it = iter(core_order)
    for i in range(n):
        if seq[i] is None:
            seq[i] = ("core", next(it))
    out = [s for s in seq if s is not None]
    if presentation == "random_distract":
        rng = random.Random(stable_seed(lat.task_id, "extra", perm_seed))
        for k in range(len(lat.extra_distractors)):
            out.insert(rng.randrange(len(out) + 1), ("extra", k))
    return out


def _atom(entity: str, a: str) -> str:
    return f"{entity} is {a}"


def rule_text(entity: str, rule: Rule) -> str:
    s, t = rule
    if len(s) == 1:
        return f"If {_atom(entity, s[0])}, then {_atom(entity, t)}."
    return f"If {_atom(entity, s[0])} and {_atom(entity, s[1])}, then {_atom(entity, t)}."


def render(lat: Latent, presentation: str, perm_seed: int | None = None,
           core_order: list[int] | None = None) -> dict:
    seq = rule_sequence(lat, presentation, perm_seed, core_order)
    facts = list(lat.facts)
    if presentation == "random_distract":
        facts = facts + list(lat.meta["extra_facts"])
    lookup = {"core": lat.core_rules, "other": lat.other_rules, "extra": lat.extra_distractors}
    fact_lines = [_atom(lat.entity, f) + "." for f in facts]
    rule_lines = [rule_text(lat.entity, lookup[k][i]) for k, i in seq]
    text = (
        "Facts:\n" + "\n".join(fact_lines) + "\n\nRules:\n" + "\n".join(rule_lines)
        + f"\n\nQuestion: Is it true that {_atom(lat.entity, lat.query)}?"
    )
    core_pos = {}
    for pos, (k, i) in enumerate(seq):
        if k == "core":
            core_pos[i] = pos
    return {
        "text": text,
        "presentation": presentation,
        "core_positions": [core_pos[i] for i in range(len(lat.core_rules))],
        "n_rules": len(seq),
        "rule_seq": seq,
    }


def solver_trace(lat: Latent) -> str:
    """Presentation-independent forward-chaining trace ending with the answer."""
    rules = lat.all_rules(with_extra=False)
    e = lat.entity
    lines = ["Known facts: " + ", ".join(_atom(e, f) for f in lat.facts) + "."]
    found = lat.query in lat.facts
    for _, (s, t) in closure(set(lat.facts), rules):
        prem = " and ".join(_atom(e, x) for x in s)
        lines.append(f"Since {prem}, {_atom(e, t)}.")
        if t == lat.query:
            found = True
            break
    if found:
        lines.append(f"So {_atom(e, lat.query)} is proved.")
        lines.append("Answer: yes")
    else:
        lines.append(f"Nothing else can be derived, so {_atom(e, lat.query)} cannot be proved.")
        lines.append("Answer: no")
    return "\n".join(lines)


def make_pool(families, depths, n_per_cell: int, seed: int, vocab: str, tag: str) -> list[Latent]:
    """Balanced pool: for every (family, depth) n_per_cell latents, half yes and half no."""
    out = []
    for fam in families:
        for d in depths:
            for i in range(n_per_cell):
                label = "yes" if i % 2 == 0 else "no"
                out.append(make_latent(fam, d, label, stable_seed(tag, seed, fam, d, i), vocab))
    return out
