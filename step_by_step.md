
## What we’re trying to achieve (concise design summary)

**Goal (Stage‑1):** Given a **messy token list (already cleaned)** and the **canonical right‑to‑left suffix trie** for a postcode block (e.g., `WD4 9H*`), return a **UPRN** iff we can match with **near‑100% precision** and **small edit tolerance**; otherwise return “no match” so a Stage‑2 (heavier) method can try.

**Core ideas we will implement now (working + safe):**

1. **Tail peeling by counts (deterministic):** iteratively drop 1–2 final tokens from the messy address when that “joins a larger subtree” in the trie (your `peel_end_tokens` rule). This discards obviously redundant tails like `... HERTFORDSHIRE ENGLAND`.

2. **Right‑to‑left trie walk with bounded edits (tiny + readable):**

   * **Exact consume** (cost 0) when the next messy token is a child.
   * **Fuzzy consume** (cost 1) if *no exact child* but a child label is within **Damerau‑Levenshtein ≤ 1** (one typo / adjacent transposition).
   * **Skip noisy messy token** (cost 0 or 1): drop the next messy token; make it **cost 0** only when the trie counts say the token is redundant at this anchor (same logic as tail peeling), otherwise **cost 1**.
   * *(Optional later) Insert a missing canonical token (cost 1) for low‑information children.* (We’ll add this after the basic version works.)

3. **Conservative accept rule (precision first):**

   * **Only accept** if we are at a node with **`count == 1`** (i.e., a **unique suffix**), it carries a **`uprn`**, and we **can’t** or **don’t need to** go deeper with the next messy token.
   * Require at least **one exact numeric match** (e.g., house/flat number) on the accepted path for extra safety.
     *(This is a strong precision guard, and still practical for Stage‑1.)*

4. **Small cost budget + margin (keep search tight):**

   * Total cost ≤ **2**.
   * Keep the **best** and **runner‑up** candidates; accept only if the best has a **strict margin ≥ 1** in cost.

The result is a compact, understandable Python implementation that mirrors what you’ll later code in C++ as a DuckDB UDF.

---

## Step‑by‑step implementation plan (small, verifiable steps)

> **File names referenced below**: we’ll add a new `matcher_stage1.py` for the walk, and extend `trie_builder.py` with some helpers. You can keep experimenting in `try.py`.



### [x] Step 1 — Add small helpers to the trie (no behavior change)

**What:** Convenience methods used by the matcher; keeps matching code readable.

**Change (`trie_builder.py`):**

* Add on `TrieNode`:

  * `def child(self, tok: str) -> Optional[TrieNode]`
  * `def child_count(self, tok: str) -> int` (0 if no such child)
  * `def has_child(self, tok: str) -> bool`
  * `def iter_children(self): return self.children.items()`
* Add a **wrapper** function that matches your “suffix” semantics:

  ```python
  def count_tail_L2R(root: TrieNode, suffix_tokens_L2R: Sequence[str]) -> int:
      # Converts L2R suffix to our R2L trie path
      path = list(reversed([t for t in suffix_tokens_L2R if t is not None]))
      return root.count_for_path(path)
  ```

**Verify:** With your Love Lane trie:

* `count_tail_L2R(root, ["LANGLEY"])` returns `7`.
* `count_tail_L2R(root, ["LANGLEY","HERTFORDSHIRE","ENGLAND"])` returns `0`.

---

### [x] Step 2 — Implement `peel_end_tokens` exactly (standalone function + tests)

**What:** Bring your reference function into a new module `matcher_stage1.py` as is, parameterized by a callable `count_tail`.

**Change (`matcher_stage1.py`):** Copy the function (Python from your spec) and host a thin wrapper:

```python
def peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2):
    return peel_end_tokens(tokens, lambda tail: count_tail_L2R(root, tail), steps, max_k)
```

**Verify (toy tests in `try.py`):**

* Input: `["KIMS","NAILS","4","LOVE","LANE","KINGS","LANGLEY","HERTFORDSHIRE","ENGLAND"]`
  Output: `["KIMS","NAILS","4","LOVE","LANE","KINGS","LANGLEY"]`.
* Input with no redundant tail: returns unchanged.

---

### [x] Step 3 — Exact right‑to‑left walk (no fuzz, no skips)

**What:** A baseline matcher that consumes tokens R→L with **exact** child matches only and accepts only at **unique** leaves (`count==1` and `uprn` present).

**Change (`matcher_stage1.py`):**

* Add `walk_exact(tokens_L2R, root)`:

  1. `t = list(reversed(tokens_L2R))`
  2. Traverse from `root` with exact `child(tok)` for each `tok` in `t`.
  3. Acceptance: if at any point the current node satisfies either:
     * Unique & blocked (strict): `node.uprn is not None` and `node.count == 1` and (either we consumed all tokens or the next token cannot descend), or
     * Exact‑exhausted terminal: all messy tokens consumed and `node.uprn is not None` (regardless of `count`).
  4. Otherwise, return `None`.

**Verify (Love Lane):**

* `"4 LOVE LANE KINGS LANGLEY"` → returns UPRN for 4.
* `"7 LOVE LANE KINGS LANGLEY"` → returns UPRN for 7 (**exact‑exhausted terminal**).
* `"ANNEX 7 LOVE LANE KINGS LANGLEY"` → returns UPRN for ANNEX address.
* `"KIMS NAILS 4 LOVE LANE KINGS LANGLEY"` → returns UPRN for 4 (business name ignored because acceptance triggers when unique and cannot descend).

---

### [x] Step 4 — Wire peeling + exact walk in a top‑level function

**What:** Build the simplest “Stage‑1” function combining Steps 2–3.

**Change (`matcher_stage1.py`):**

```python
def match_stage1_exact_only(tokens, root):
    toks = peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2)
    return walk_exact(toks, root)
```

**Verify:** Re‑run the Step‑3 tests but call `match_stage1_exact_only`.

---

### [x] Step 5 — Add **skip** (delete messy token) with **count‑aware cost**

**What:** Introduce a tiny state search with **costs**. We’ll allow skipping messy tokens:

* **Skip cost 0** if the token is **redundant** at this anchor:

  * Let `c_anchor = node.count` and `c_combo = node.child_count(tok)`; if `c_anchor > c_combo` and `c_anchor / max(1,c_combo) ≥ 2.0`, treat skip as **0**.
* Else **skip cost 1**.

We still **do not** add fuzzy matches in this step—keeps the change small.

**Change (`matcher_stage1.py`):**

* New function `match_stage1_with_skips(tokens, root, max_cost=2)`:

  * Reverse tokens to `t`.
  * Dijkstra (or a min‑heap) over states `(cost, node, i, exact_hits, saw_numeric)` where `i` is the index in `t` (number of tokens left to consume).
  * Transitions:

    1. **Exact consume** if `i>0` and `child(t[i-1])` exists → `(cost+0, child, i-1, exact_hits+1, saw_numeric or is_numeric(t[i-1]))`.
    2. **Skip messy** if `i>0` → `(cost + skip_cost(node,t[i-1]), node, i-1, exact_hits, saw_numeric)`.
  * Acceptance condition (identical to Step 3) plus two guards:

    * **require at least one numeric match** on the path: `saw_numeric == True`.
    * **exact\_hits ≥ 2** (at least two exact tokens matched).
  * Keep track of **best** and **runner‑up** candidates by cost (UPRN + cost).
  * When done: accept if best exists, `best.cost ≤ max_cost`, and either no runner‑up or `runner.cost ≥ best.cost + 1`.

**Verify (Love Lane + synthetics):**

* `"KIMS NAILS 4 LOVE LANE KINGS LANGLEY"` → exact path, accepted (cost 0).
* `"4 LOVE LANE KINGS LANGLEY EXTRA"` → peeled at tail; if placed inside (e.g., `"4 LOVE EXTRA LANE KINGS LANGLEY"`), the skip op handles it with cost 0 or 1 depending on counts. In Love Lane, it’s cost 0 (redundant at that anchor: child_count(EXTRA)=0).
* `"7 LOVE LANE KINGS LANGLEY"` → accepted via Step‑3’s exact‑exhausted terminal rule.

---

### Step 6 — Add tiny **fuzzy consume** (Damerau–Levenshtein ≤ 1)

**What:** If no exact child for `t[i-1]`, allow stepping into a child whose label is within DL distance 1 from the token; cost +1.

**Change (`matcher_stage1.py`):**

* Implement `damerau_levenshtein_at_most(a, b, k=1)` (early exit once distance > k). Keep the function **small and readable**; no optimization necessary.
* In the state expansion, **after** trying exact child:

  * Loop over all `node.children` and if `DL≤1`, enqueue `(cost+1, child, i-1, exact_hits, saw_numeric or is_numeric(child_label))`.

**Verify:**

* Create canonical `"HAYDN PARK ROAD"` and messy `"HADYN PARK ROAD"` within a toy trie; fuzzy should bridge `HADYN`↔`HAYDN` with cost 1 and still accept if unique and numeric present.
* Ensure fuzz is only used when no exact child exists for that token (keeps paths stable).

---

### Step 7 — Expose a single Stage‑1 API and keep Step‑5/6 knobs strict

**What:** Create one exported function with conservative defaults and the acceptance guards.

**Change (`matcher_stage1.py`):**

```python
DEFAULT_PARAMS = dict(
    max_cost=2,
    min_exact_hits=2,
    require_numeric=True,
    skip_redundant_ratio=2.0,  # 0-cost skip if c_anchor/c_combo >= 2
)

def match_stage1(tokens, root, params=DEFAULT_PARAMS):
    toks = peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2)
    return match_stage1_core(toks, root, params)  # wraps the search from Steps 5–6
```

Return a **structured result**:

```python
{"matched": True, "uprn": <int>, "cost": <int>, "path_tokens": [...]}  # or matched=False
```

**Verify:** Re-run earlier tests through this single entry point.

---

### Step 8 — (Optional, safe) One **adjacent transposition** transition

**What:** Handle swaps like `"7 ANNEX"` vs `"ANNEX 7"` cheaply. To avoid overgeneralization, **gate it**:

* Only allow one transposition **and** only when **one of the two tokens is numeric** (house/flat anchor).

**Change (`matcher_stage1.py`):**

* In expansion, if `i > 1` and not `used_transpose`:

  * Let `a=t[i-1]`, `b=t[i-2]`.
  * If `is_numeric(a) or is_numeric(b)` and there exist children `child(b)` then `child(b).child(a)`, allow transition to that grandchild with `i-2`, cost+1, and `used_transpose=True`.

**Verify:** In Love Lane, test messy `"7 ANNEX LOVE LANE KINGS LANGLEY"` vs canonical `"ANNEX 7 ..."`: should accept annex UPRN at cost 1.

*(If you prefer, you can defer this step until after you’re happy with Steps 1–7.)*

---

### Step 9 — (Optional) Insert missing canonical token (low‑info only)

**What:** Allow “virtual insertion” of a canonical token with cost 1, but **only for low‑information children**:

* Define **info gain**: `IG = parent.count / child.count`.
* Allow insertion only if `IG ≤ 1.3` (tunable) and try **at most 1–2 children** per node.

**Change (`matcher_stage1.py`):**

* Add an “insert” expansion that keeps `i` unchanged and moves to child with `cost+1`.

**Verify:** Synthetic trie where county is present canonically but omitted in messy; the algorithm can still reach a unique leaf.

*(Again, optional for v1. Keep off until you need it.)*



### Step 10 — Integrate into `try.py` (simple demo harness)

**What:** Show end‑to‑end on:

* your ***toy*** Love Lane trie,
* and optionally **one real postcode block** from your parquet.

**Change (`try.py`):**

* Import `match_stage1`.
* Define a few messy examples (from your prompt).
* Print the result object (UPRN, cost, path, debug trace).

**Verify:** See expected UPRNs on the toy set; on a real block, confirm that exact/near‑exact inputs succeed and ambiguous ones refuse.

---


## Notes / defaults you can copy into code comments

* **Orientation:** The trie is **right‑to‑left**; always reverse the messy tokens before walking. `count_tail_L2R` internally reverses the argument so callers can think in L2R.
* **Acceptance (precision guard):** Step‑3 (exact‑only) accepts either unique‑blocked or exact‑exhausted terminal nodes. Later, in the costed search (Step‑5+), also enforce: **require a numeric** token and `≥ 2` exact token hits on the accepted path, and maintain margin over the runner‑up.
* **Costs:** exact=0, fuzzy=1, skip=0/1 (0 only when counts say redundant), (optional) insert=1, (optional) transpose=1; **max total cost=2**; **margin ≥ 1** against runner‑up.
* **Fuzzy:** Keep to **DL ≤ 1** only, and use **only if no exact** child exists for that token.
* **Numeric detector:** `is_numeric(tok)` should treat `"191A"` / `"23A"` as numeric‑ish (e.g., `re.fullmatch(r"\d+[A-Z]?", tok)`).

---

## (Optional) Minimal function signatures

This can help you lay out the code without writing it all at once:

```python
# matcher_stage1.py

from dataclasses import dataclass
from typing import Optional, Sequence, Dict, Any
from trie_builder import TrieNode, count_tail_L2R

def peel_end_tokens(tokens, count_tail, steps=4, max_k=2): ...
def peel_end_tokens_with_trie(tokens, root: TrieNode, steps=4, max_k=2): ...

def is_numeric(tok: str) -> bool: ...
def dl_at_most(a: str, b: str, k: int = 1) -> bool: ...  # returns True/False

@dataclass
class Params:
    max_cost: int = 2
    min_exact_hits: int = 2
    require_numeric: bool = True
    skip_redundant_ratio: float = 2.0
    allow_transpose: bool = False  # enable after Step 8
    allow_insert_lowinfo: bool = False  # enable after Step 9
    lowinfo_gain_thresh: float = 1.3
    debug: bool = False

def walk_exact(tokens_L2R: Sequence[str], root: TrieNode) -> Optional[int]: ...

def match_stage1(tokens_L2R: Sequence[str], root: TrieNode, params: Params = Params()) -> Dict[str, Any]:
    # 1) peel
    # 2) Dijkstra over states with transitions: exact, (fuzzy), skip, (transpose), (insert)
    # 3) acceptance + margin; build debug trace
    ...
```

---

## How you’ll know it’s “good enough” for Stage‑1

* **Passes** all toy tests above.
* On a handful of **real** postcode groups:

  * **Exact/near‑exact** messy inputs return the expected UPRN with **cost ≤ 1**.
  * **Ambiguous stems** that are not exact‑exhausted (or require edits/skips) are **rejected** when there is no clear margin; exact‑exhausted terminals like “7 LOVE LANE …” are **accepted** in Step‑3.
  * A few realistic typos (one character) are **recovered** (cost 1) when a numeric anchor is present.

Once you’re happy, you can start toggling the optional transitions (transpose, insert low‑info) and tuning thresholds. The Python logic directly mirrors a small C++ UDF: a priority queue over `(node*, i, cost, flags)` with 3–4 transitions and the count‑aware rules you already have.

If you want, I can next draft the `matcher_stage1.py` code skeleton with the small DL≤1 function and the Dijkstra loop ready to paste into your repo.
