Below is a **step‑by‑step, implementation‑ready plan** for refactoring your matcher into small, pluggable **transition rules** while preserving behavior, traces, and APIs. Each step is **small, self‑contained, and verifiable**. Follow them in order; run the full test suite after each step.

---

## Ground rules (do not violate)

* **APIs must not change**: `match_stage1`, `match_stage1_with_skips`, `match_address`, `Params` defaults.
* **Event payloads must remain byte‑for‑byte compatible** with current code:

  * `EXACT_DESCEND`: `{"action","messy","canon","m_index","child_count","anchor_count"}`
  * `SKIP_REDUNDANT`/`SKIP_PENALIZED`: `{"action","messy","m_index","anchor_count","child_count","ratio","threshold"}`
  * `FUZZY_CONSUME`: `{"action","messy","canon","m_index","edit_type","child_count"}`
  * Accept/Stop events are produced exactly as today by the engine logic.
* **Search semantics unchanged**: peeling first; then best‑first search; rule order remains **Exact → Skip → Fuzzy**; acceptance guards and runner‑up margin unchanged.
* **Determinism and performance**: same keys in `seen`, same heap order tie‑breakers (preserve `seq` usage).

---

## Step 0 — Pre‑flight checkpoint

**Action:** Commit current working tree. Run the full test suite to establish the baseline.

**Verify:** All tests pass.

---

## Step 1 — Introduce light “transition” types (no behavior change)

**File:** `matcher/matcher_stage1.py`

**Action:**

* Define lightweight *internal* types (no exports):

  * `State` (read‑only snapshot): `(node: TrieNode, i: int, exact_hits: int, saw_num_any: bool, saw_num_exact: bool)`.
  * `Move` (proposed transition): fields to carry **target node**, `i_delta`, `exact_delta`, updated numeric flags, `cost_delta` (0/1), **trace event dict**, and optional `last_consume_m_index` & `last_canon_label` hints for accept star placement.
  * `RuleFunc` protocol/type alias: callable `(state, tokens_r2l, n, params-like) -> iterable of Move`.

**Rationale:** Gives a shared vocabulary without changing logic.

**Verify:** `pytest -q` still passes (no code path uses these yet).

---

## Step 2 — Extract `skip_cost` into a tiny helper (behavior‑preserving)

**File:** `matcher/matcher_stage1.py`

**Action:**

* Move the body of the inner `skip_cost` into a top‑level helper (e.g., `_calc_skip_cost(node, tok, skip_redundant_ratio)`), preserving the exact semantics (including **only 0‑cost when `child_count(tok) > 0`** and ratio gate).
* Update `_search_with_skips` to call this helper; remove the inner version.

**Verify:** Tests still pass. This de‑duplicates logic you’ll need inside the skip rule.

---

## Step 3 — Factor a single push helper (behavior‑preserving)

**File:** `matcher/matcher_stage1.py`

**Action:**

* Inside `_search_with_skips`, introduce a tiny local `"push_move"` helper that:

  * Accepts: current key/state, a **Move**, and current `cost/seq`.
  * Pushes `(new_cost, seq, new_node, new_i, new_exact, new_saw_any, new_saw_exact)` onto the heap.
  * Writes `parents[next_key] = {"parent": cur_key, "event": move.event, "g_cost": new_cost, "node": move.node, ...}` exactly as today.
  * **Only when the move’s action is a consume** (`EXACT_DESCEND` or `FUZZY_CONSUME`), also attach `"last_consume_m_index"` and `"last_canon_label"` to that parents entry, using the move’s optional hints if available.

**Rationale:** This prevents drift as you replace inline blocks with rules.

**Verify:** Use `push_move` for **one** of the existing inline transitions (e.g., exact) but without removing the original logic yet (call the helper inside that block). Tests pass.

---

## Step 4 — Implement `rule_exact` and switch to it

**File:** `matcher/matcher_stage1.py`

**Action:**

* Implement a pure function `rule_exact(state, t, n, params)` that:

  * If `state.i >= n` or no exact child for `t[state.i]`: **yield nothing**.
  * Else, **yield exactly one Move** with cost 0, `i_delta=1`, `exact_delta=1`, numeric flags updated from the messy token, and a trace event identical to current exact block (include both `child_count` and `anchor_count`).
  * Set `last_consume_m_index` & `last_canon_label` for the move.

* In `_search_with_skips`:

  * **Replace** the current “exact child” inline block with: build `State` and iterate `rule_exact`, then pass each yielded move to `push_move`.

**Verify:** All tests pass. Compare a few traces by eye in `try.py` to confirm identical `EXACT_DESCEND` events.

---

## Step 5 — Implement `rule_skip` and switch to it

**File:** `matcher/matcher_stage1.py`

**Action:**

* Implement `rule_skip(state, t, n, params)` that:

  * If `state.i >= n`: **yield nothing**.
  * Else compute `cost = _calc_skip_cost(state.node, tok, params.skip_redundant_ratio)`.
  * **Always yield one Move** with `i_delta=1`, `exact_delta=0`, numeric flags **unchanged**, `cost_delta=cost`, and trace event matching current skip block (`SKIP_REDUNDANT` or `SKIP_PENALIZED` with counts/ratio/threshold).
  * **Do not** set last‑consume hints for skip.

* In `_search_with_skips`:

  * **Replace** the current skip inline block (including heap push and parents event) with an iteration over `rule_skip` using `push_move`.

**Verify:** Tests still pass, especially those asserting `SKIP_PENALIZED` vs `SKIP_REDUNDANT`. Confirm that the ratio and threshold text in the alignment remains unchanged.

---

## Step 6 — Implement `rule_fuzzy` and switch to it

**File:** `matcher/matcher_stage1.py`

**Action:**

* Implement `rule_fuzzy(state, t, n, params)` that:

  * If `state.i >= n` or `state.node.has_child(tok)` (i.e., exact is available): **yield nothing** (fuzzy only when exact is absent).
  * Else, for each `(lbl, child)` where `dl1_edit_type(tok, lbl)` returns a non‑`"exact"` edit—emit **one Move per child** with `i_delta=1`, `exact_delta=0`, `cost_delta=1`.
  * Numeric flags: `saw_num_any` flips true if `lbl` is numeric; `saw_num_exact` **does not** change.
  * Trace event must match current fuzzy block precisely (`FUZZY_CONSUME` with `edit_type` and `child_count`).
  * Set last‑consume hints for the move.

* In `_search_with_skips`:

  * **Replace** the current fuzzy inline block (the `if child is None: for lbl, ch ...`) with iteration over `rule_fuzzy` using `push_move`.

**Verify:** All tests pass, especially those checking fuzzy acceptance and “transpose/substitution” reasons in the alignment.

---

## Step 7 — Centralize rule ordering (still behavior‑preserving)

**File:** `matcher/matcher_stage1.py`

**Action:**

* After building `State`, call the three rules **in order**: `exact → skip → fuzzy`. (You have already done this implicitly by replacing inline blocks in that order.)
* Optionally, gather them into a local list `rules = [rule_exact, rule_skip, rule_fuzzy]` and iterate; keep the order fixed.

**Verify:** Tests pass. This makes the engine’s loop visually tiny: accept check → `for rule in rules: for move in rule(...): push_move(move)`.

---

## Step 8 — Keep acceptance & star placement exactly as today

**File:** `matcher/matcher_stage1.py`

**Action:**

* **Do not** change the acceptance predicate or runner‑up margin.
* Ensure accept star placement continues to prefer the recorded `"last_consume_m_index"` from the parents entry if available; otherwise, use the existing fallback (`i-1` mapping).
* Confirm that `match_stage1`’s post‑processing (path reconstruction & `STOP_*` enrichment) remains untouched.

**Verify:** Tests that assert ACCEPT\_\* placement and STOP\_\* reasons pass unchanged. Try the demo cases in `try.py`.

---

## Step 9 — Light clean‑up (no external changes)

**Files:** `matcher/matcher_stage1.py` (and only here)

**Action:**

* Remove the now‑dead inline transition code.
* Ensure all newly added helpers (`State`, `Move`, `RuleFunc`, `_calc_skip_cost`, `rule_*`, `push_move`) are **module‑private** (prefix `_` where appropriate). Do **not** export them from `__init__.py`.
* Update internal docstrings to explain the rule pattern and the fixed rule order, but **do not** change function signatures or public names.

**Verify:** Full test suite passes.

---

## Step 10 — Sanity checks in the demo script

**File:** `try.py`

**Action:**

* Run the existing scenarios; visually ensure:

  * Vertical alignment table looks the same as before (since it uses the same events).
  * “Result summary” and consumed path summary unchanged.

**Verify:** Outputs match expectations you’ve been using (by eye is fine; tests are the real guard).

---

## Optional Step 11 — (Documentation only) Brief design note

**File:** `AGENTS.md` or a short `matcher/ARCHITECTURE.md`

**Action:**

* Document in 5–7 lines that the search is a uniform‑cost search over an implicit graph, and each fault tolerance is encoded as a **transition rule**. Note that the refactor improves separation of concerns and makes it easy to add future behaviors without touching the engine.

**Verify:** No code; just for future readers.

---

## What you’ll see when you’re done

* `_search_with_skips` reads almost linearly:

  1. pop state;
  2. accept/guards;
  3. if tokens remain: **for each rule** produce **moves** and push them;
  4. continue; then margin decision at the end.
* All “fault tolerance” logic is **localized** in three small rule functions, each with a single responsibility and a single event shape.
* Tests and demos behave identically.

---

## Common pitfalls (watch for these as you implement)

* **Event payload drift**: a missing `anchor_count` or `child_count` will break alignment/consumed‑path helpers. Keep keys exact.
* **m\_index math**: must remain `m_index = (n - 1) - i` for the token consumed in that transition.
* **Fuzzy precondition**: only offer fuzzy moves when **no exact child** exists.
* **Skip semantics**: 0‑cost only when the messy token is a **known child** and ratio condition holds; otherwise cost 1.
* **Parents map overwrite**: preserve the current “take the better `g_cost`” rule when writing `parents[next_key]`.

---

### Final note

This refactor deliberately **does not add features**. Its main benefits are clarity and maintainability. A side benefit is that, later, adding any additional behavior becomes “write one new rule, add it to the ordered list,” leaving the engine and existing rules untouched.
