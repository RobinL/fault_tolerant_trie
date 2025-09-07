# Goal (short)

Augment the Stage-1 matcher to return:

* the **consumed canonical path** (the labels actually traversed in the trie, in L2R order),
* the **node count at each step** and at the **final node**,
* and, when **no match** and the final node’s candidate set is **small (≤ limit)**, a list of **UPRNs** under that node.

Keep search/accept logic unchanged. Implement path reconstruction and candidate enumeration as **post-processing helpers**.



## Step 1 — Add lightweight helpers (no behavior changes)

**Files:** `matcher/trace_utils.py`

**Add functions (pure):**

1. `reconstruct_consumed_events(best_state, parents) -> list[dict]`

   * Walk back from `best_state` using `parents`, collect events whose `action` is in `{"EXACT_DESCEND","FUZZY_CONSUME"}`.
   * Reverse to get chronological order.
   * Return the list of those consumption events.

2. `events_to_consumed_path(consumed_events) -> tuple[list[str], list[int]]`

   * From each event, take:

     * **label**: `event["canon"]` (canonical token you descended into).
     * **count**: prefer `event["child_count"]` if present; else `None`.
   * Return `(consumed_path_L2R, consumed_counts_L2R)`. (Order = events order.)

3. `final_node_from_state(best_state, parents) -> TrieNode | None`

   * Look up `parents[best_state]["node"]` if present; else `None`.
   * (We already attach `"node"` into `parents` for transitions/accept in your code; this is a thin accessor.)

4. `collect_uprns(node: TrieNode, limit: int) -> list[int]`

   * DFS/BFS over `node`’s subtree; append `child.uprn` when `child.uprn is not None`.
   * Stop as soon as `len(uprns) > limit`; then return an **empty list** (signal “too many”).
   * If traversal finishes with `len(uprns) ≤ limit`, return that list (sorted for stability).

**Verify:** Add a tiny in-file doctest-style comment for each function showing input/expected behavior. No imports or callers yet.

---

## Step 2 — Ensure parents carry enough info (tiny augmentation, search logic unchanged)

**Files:** `matcher/matcher_stage1.py` (inside `_search_with_skips` where you already build `parents`)

**Change:**

* When pushing **EXACT\_DESCEND**: you already set `{"node": child}` in `parents[next_key]`. Also ensure `event` includes `canon` and `child_count` (you already do).
* When pushing **FUZZY\_CONSUME**: ensure you also set `{"node": ch}` in `parents[next_key3]` and include `canon` and (if easy) `child_count=ch.count`.
  *(This keeps fuzzy on par with exact for path reconstruction and counts.)*
* When writing **accept\_event** on an accepting state: ensure `parents[cur_key_acc]` has `"node": node` (you already set).
* **No changes** to costs/guards/logic.

**Verify:** Run `try.py`. Output/behavior should be **identical** to current (tables & matches unchanged).

---

## Step 3 — Post-process result: compute consumed path & counts

**Files:** `matcher/matcher_stage1.py`

**Change (inside `match_stage1`)**:

* Add a new optional param to `Params`: `max_uprns_to_return: int = 10`.
* After `_search_with_skips(...)` returns and you already reconstruct the ordered events into `trace`:

  * Call `reconstruct_consumed_events(best_state, parents)` → `consumed_events`.
  * Call `events_to_consumed_path(consumed_events)` → `(consumed_path, consumed_counts)`.
  * Compute `final_node = final_node_from_state(best_state, parents)` and `final_node_count = int(final_node.count)` if not `None`.

**Return dict additions:**

* `consumed_path`: `list[str]` (canonical labels, L2R order)
* `consumed_path_counts`: `list[int|None]` (same length; counts after stepping into each label; `None` if not captured)
* `final_node_count`: `int|None` (count at last consumed node)
* **Rename** `peeled_tokens` → `search_tokens` (keep **back-compat alias** `peeled_tokens` holding the same list and add a TODO to remove later).

**Verify:** With your existing `try.py`, print the result dict for:

* A **match** case (e.g., `... 4 LOVE LANE ...`) → `consumed_path` should include the house number and street etc. `final_node_count` should reflect the count at the leaf.
* A **no-match** case (e.g., `700 LOVE ...`) → `consumed_path` stops at `LOVE`, `final_node_count` equals that node’s count.

Behavior/matches remain unchanged.

---

## Step 4 — Add small candidate set enumeration (only on no-match)

**Files:** `matcher/matcher_stage1.py`

**Change (still in `match_stage1`)**:

* If `matched` is `False` **and** `final_node` is not `None` **and** `final_node_count` is not `None`:

  * If `final_node_count ≤ params.max_uprns_to_return`:

    * Call `collect_uprns(final_node, params.max_uprns_to_return)` → `cand`.
    * If `cand` is **non-empty**, set `candidate_uprns = cand`.
    * If `cand` is **empty** (means traversal exceeded limit early), **omit** `candidate_uprns`.
  * Else: omit `candidate_uprns`.

**Return dict additions (conditionally):**

* `candidate_uprns`: `list[int]` when present
* `limit_used`: `int` (echo `params.max_uprns_to_return`) — helps the caller reason about omission.

**Verify:**

* `700 LOVE LANE KINGS LANGLEY ENGLAND` → `matched=False`, `final_node_count` reflects subtree; if ≤ limit, `candidate_uprns` present and length ≤ limit; else absent.
* A match case remains unchanged (no `candidate_uprns`).

---

## Step 5 — Wire through a thin convenience wrapper

**Files:** `matcher/matcher_stage1.py`

**Change:**

* Update `match_address(...)` to pass through a `Params` (so callers can tune `max_uprns_to_return`).
* Return the **full result dict on match** and also on **no-match** (i.e., **remove** the `None` fallback), because the enriched failure info is now valuable.

  * If you must keep the old contract, add `match_address_full(...)` as a new function, and leave `match_address(...)` behavior as-is.

**Verify:** Call `match_address(tokens, trie, Params(max_uprns_to_return=5))` in `try.py` for both success and failure; ensure the dict includes the new fields.

---

## Step 6 — Update the rendering (optional, purely cosmetic)

**Files:** `matcher/trace_utils.py`

**Change:** No changes required for alignment text. (It already visualizes what happened.)
Optionally, add a small helper `render_consumed_summary(consumed_path, counts, final_count)` that prints:

```
Consumed path (L→R): LOVE → LANE → KINGS → LANGLEY
Counts along path:   7 → 7 → 7 → 7   | final=7
```

**Verify:** In `try.py`, after printing the alignment table, print this summary once.

---

## Step 7 — Try-script demos

**Files:** `try.py`

**Add printouts:**

* For **match success** and **no-match** cases, print the new fields:

  * `consumed_path`, `consumed_path_counts`, `final_node_count`, and `candidate_uprns` (when present).
* Keep existing alignment prints intact.

**Verify (spot-checks):**

* Success (`… 4 LOVE …`):

  * `matched=True`, `consumed_path` includes `4`, star sits under `4`, `candidate_uprns` absent.
* No-match (`700 LOVE …`):

  * `matched=False`, `consumed_path=['LOVE','LANE','KINGS','LANGLEY']`, `final_node_count` matches trie, and `candidate_uprns` returned only if `final_node_count ≤ limit`.

---

## Step 8 — Back-compat + docs

**Files:** `matcher/matcher_stage1.py`, `matcher/__init__.py`, README/inline docstrings

**Change:**

* Export any new helpers only if you want them public; otherwise keep them internal.
* In docstrings for `match_stage1` and `match_address`, document the new fields and the `max_uprns_to_return` behavior.
* Note the alias: `search_tokens` (preferred) and `peeled_tokens` (temporary alias).

**Verify:** `from matcher import match_address` still works; calling code doesn’t break. New fields appear for new callers.

---

## Step 9 — Edge-cases & guardrails

**Checklist:**

* If `best_state` or `parents` is `None` (e.g., tracing disabled), return new fields as `[]`/`None` without error.
* If **fuzzy** events exist, ensure `consumed_path` uses `event["canon"]` (canonical label), not the messy token.
* If any event lacks `child_count`, allow `None` in `consumed_path_counts`.

**Verify:** Temporarily disable tracing (set `trace=None`) and ensure the function **still returns** gracefully with `consumed_path=[]`, `final_node_count=None`, etc.

---

## Step 10 — Final quick performance sanity

**Action:**

* Ensure `collect_uprns` returns **empty** when traversal exceeds `limit` (i.e., we do not build huge lists).
* No change in complexity for normal matches (subtree walk only happens on **no-match + small subtree**).

**Verify:** Create a synthetic node with many children and confirm `collect_uprns(node, limit=10)` returns `[]` quickly (early abort).

---

### Done Criteria

* For success inputs, previous behavior is unchanged; new fields are present and coherent.
* For failures, the result includes a clear **consumed path**, a **final node count**, and (when small) **candidate UPRNs**.
* No changes to search/accept logic; all additions are post-processing/utility.
* `try.py` demo prints confirm both success and failure flows.
