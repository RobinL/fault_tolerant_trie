# Step-by-step plan: clean tracing & alignment table

> **High-level objective**
>
> Add a clean, event-driven tracing layer to Stage-1 so we can explain, step-by-step, how the algorithm moved through the R→L suffix trie—**without polluting** the core control logic. The trace is **optional** and lives in a separate module, but when enabled, it can render a readable alignment like:
>
> ```
> Messy (R→L):   ENGLAND   HERTFORDSHIRE   LANGLEY    KINGS     LANE     EXTRA     LOVE     4      NAILS    KIMS
> Action:          ⌫            ⌫             ✓         ✓         ✓        ·        ✓       ✓★       ·        ·
> Canonical:        —            —          LANGLEY    KINGS     LANE       —      LOVE      4        —        —
> Reason:        peel         peel        exact      exact     exact   redundant   exact   unique     post-   post-
>                                                                                          leaf      accept   accept
>                                                                                                     skip     skip
> ```
>
> * **Core behavior does not change** when tracing is off.
> * Tracing is **opt-in**, minimal overhead, and confined to `matcher/trace_utils.py`.

---

## Conventions & constraints

* Orientation: messy tokens are provided **L2R**, but rendered **R→L** for the table.
* Deterministic layout: every event addressing a messy token includes its **L2R index** `m_index`. The table uses **columns = tokens in R→L order**.
* No `print()` in library code. Rendering lives in helpers and `try.py`.
* Accept remains unchanged (unique UPRN + guards). Tracing is passive/observational.

---

## Event taxonomy (this iteration)

We will emit only the events needed to produce the table above:

* `PEEL_TAIL`: `{action:"PEEL_TAIL", removed_tokens: List[str], k: int}`
* `EXACT_DESCEND`: `{action:"EXACT_DESCEND", messy: str, canon: str, m_index: int, child_count: int}`
* `SKIP_REDUNDANT`: `{action:"SKIP_REDUNDANT", messy: str, m_index: int, anchor_count: int, child_count: int}`
* `SKIP_PENALIZED`: `{action:"SKIP_PENALIZED", messy: str, m_index: int}`
* `ACCEPT_UNIQUE`: `{action:"ACCEPT_UNIQUE", uprn: int, at_m_index: int}`  (star is placed on this column)
* `ACCEPT_TERMINAL`: `{action:"ACCEPT_TERMINAL", uprn: int, at_m_index: int}`

*(Later: `FUZZY_CONSUME` with `edit_type` when you want to show typos.)*

---

## Step 0 — Create the tracing module (new file, no behavior change) [DONE]

**File:** `matcher/trace_utils.py`

**What to add:**

1. A tiny event type and **Trace** collector:

```python
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Event = Dict[str, Any]

@dataclass
class Trace:
    enabled: bool = True
    tokens_l2r: Optional[List[str]] = None
    events: List[Event] = field(default_factory=list)

    def set_tokens(self, tokens_l2r: Sequence[str]) -> None:
        if self.enabled:
            self.tokens_l2r = list(tokens_l2r)

    def add(self, event: Event) -> None:
        if self.enabled:
            self.events.append(event)
```

2. A **table builder** that converts events → 4 aligned rows:

```python
ICONS = {
    "exact": "✓",
    "star":  "✓★",
    "peel":  "⌫",
    "dot":   "·",
}
DASH = "—"

def build_alignment_table(tokens_l2r: Sequence[str], events: List[Event]) -> Dict[str, List[str]]:
    """
    Returns a dict with four equal-length lists:
      - "messy_r2l": tokens printed R→L
      - "action":    icon per column (⌫, ✓, ✓★, ·)
      - "canonical": canonical token aligned under exact matches; DASH otherwise
      - "reason":    short reason labels
    """
    n = len(tokens_l2r)
    # Initialize columns
    messy_r2l = list(reversed(list(tokens_l2r)))
    action = [ICONS["dot"]] * n
    canonical = [DASH] * n
    reason = [""] * n

    def col_from_m_index(m_index: int) -> int:
        # Column 0 is the rightmost messy token (L2R index n-1)
        return (n - 1) - m_index

    # First, mark peeled tail (those have no m_index; they are the rightmost k tokens in R→L)
    for ev in events:
        if ev.get("action") == "PEEL_TAIL":
            k = int(ev.get("k", 0))
            # Rightmost k columns (R→L) correspond to L2R indices n-1, n-2, ...
            for j in range(k):  # j=0 => rightmost
                idx = j  # R→L index
                action[idx] = ICONS["peel"]
                reason[idx] = "peel"

    # Next, mark exact / skips along the chosen path
    for ev in events:
        a = ev.get("action")
        if a == "EXACT_DESCEND":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["exact"]
            canonical[j] = str(ev.get("canon", DASH))
            reason[j] = "exact"
        elif a == "SKIP_REDUNDANT":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "redundant"
        elif a == "SKIP_PENALIZED":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "skip"

    # Finally, mark acceptance star at the consumed token column
    for ev in events:
        if ev.get("action") in ("ACCEPT_UNIQUE", "ACCEPT_TERMINAL"):
            j = col_from_m_index(int(ev["at_m_index"]))
            action[j] = ICONS["star"]
            reason[j] = "unique leaf" if ev["action"] == "ACCEPT_UNIQUE" else "terminal"

            # Post-accept leftover tokens to the left (in R→L) are "post-accept skip"
            for k in range(j + 1, n):
                if reason[k] == "":
                    reason[k] = "post-accept"
            break  # only one accept per path

    return {"messy_r2l": messy_r2l, "action": action, "canonical": canonical, "reason": reason}
```

3. A **text renderer** (column-aligned, simple monospace):

```python
def render_alignment_text(table: Dict[str, List[str]]) -> str:
    rows = [
        ("Messy (R→L):", table["messy_r2l"]),
        ("Action:",      table["action"]),
        ("Canonical:",   table["canonical"]),
        ("Reason:",      table["reason"]),
    ]
    # Compute column widths
    cols = len(table["messy_r2l"])
    widths = [0] * cols
    for _, values in rows:
        for i, v in enumerate(values):
            widths[i] = max(widths[i], len(v))
    # Build lines
    lines = []
    for label, values in rows:
        parts = [label.ljust(13)]  # pad label for neatness
        for i, v in enumerate(values):
            parts.append(str(v).rjust(widths[i] + 2))
        lines.append("".join(parts))
    return "\n".join(lines)
```

**Verify:** Import `Trace`, build a fake table with a few hardcoded events and ensure `render_alignment_text` prints reasonable columns.

---

## Step 1 — Thread an optional `trace` through the public API

**Files:** `matcher/matcher_stage1.py`, `matcher/__init__.py`

**Changes:**

* Add `trace: Optional[Trace] = None` param (default `None`) to:

  * `peel_end_tokens_with_trie(...)`
  * `match_stage1(...)`
  * internal `_search_with_skips(...)`

* Do **not** change behavior when `trace` is `None`.

Example signatures:

```python
def peel_end_tokens_with_trie(tokens: Sequence[str], root: TrieNode, steps: int = 4, max_k: int = 2, trace: Optional[Trace] = None) -> List[str]: ...
def match_stage1(tokens_L2R: Sequence[str], root: TrieNode, params: Params = Params(), trace: Optional[Trace] = None) -> Dict[str, Any]: ...
def _search_with_skips(..., trace: Optional[Trace] = None) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[Dict[str, Any]]]: ...
```

Return type of `_search_with_skips` will soon include a **best state handle** to reconstruct the path (see Step 3). For now, return unchanged; we’ll upgrade in Step 3.

**Verify:** `pytest` still passes; `try.py` runs unchanged when `trace=None`.

---

## Step 2 — Emit `PEEL_TAIL` from peeling [DONE]

**File:** `matcher/matcher_stage1.py`

Inside `peel_end_tokens_with_trie`:

* Before returning, if tokens were removed and `trace` is set:

```python
peeled = peel_end_tokens(tokens, _count_tail, steps=steps, max_k=max_k)
if trace is not None:
    trace.set_tokens(tokens)
    k = len(tokens) - len(peeled)
    if k > 0:
        removed = tokens[-k:]
        trace.add({"action": "PEEL_TAIL", "removed_tokens": removed, "k": k})
return peeled
```

**Verify:** In `try.py`, create a `Trace`, call `peel_end_tokens_with_trie(..., trace=trace)`, then:

```python
from matcher.trace_utils import build_alignment_table, render_alignment_text
tbl = build_alignment_table(tokens, trace.events)
print(render_alignment_text(tbl))
```

You should see the rightmost columns marked `⌫` / `peel`.

---

## Step 3 — Add predecessor pointers (only when tracing) to `_search_with_skips` [DONE]

We need the **chosen best path** to emit `EXACT_DESCEND`, `SKIP_*`, `ACCEPT_*`.

**File:** `matcher/matcher_stage1.py`

**What to add:**

* Define a minimal **state key** and a **parent map** only when `trace` is provided:

```python
from typing import Tuple

StateKey = Tuple[int, int, int, bool, bool]  # (node_id, i, exact_hits, any_num, exact_num)
ParentMap = Dict[StateKey, Dict[str, Any]]   # stores: {"parent": StateKey|None, "event": Event}
```

* When pushing a transition into the heap, if `trace` is enabled, **record** its parent and the **event** that led to it:

  * For exact consume: event = `{"action":"EXACT_DESCEND","messy": tok, "canon": lbl, "m_index": m_idx, "child_count": child.count}`
  * For skip:

    * compute `s_cost = skip_cost(node, tok)`
    * event = `{"action": "SKIP_REDUNDANT" ...}` if `s_cost == 0`, else `{"action": "SKIP_PENALIZED" ...}`
  * For fuzzy (later—optional).

* The `m_index` can be computed in the search:

  * `n = len(t)` (R→L tokens)
  * consuming `t[i]` corresponds to `m_index = (n-1) - i`

* When accepting a candidate as **best**, save the **terminal state key** (e.g., `best_state: Optional[StateKey]`) alongside cost.

* **Return** that `best_state` and `parent_map` to the caller (extend the return tuple).

Signature change:

```python
def _search_with_skips(..., trace: Optional[Trace] = None
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[StateKey], Optional[ParentMap]]:
    ...
    return best_uprn, best_cost, runner_cost, best_state, parents if trace else (best_uprn, best_cost, runner_cost, None, None)
```

**Verify:** No rendering yet; just unit test that `best_state` is not `None` on a simple exact case.

---

## Step 4 — Reconstruct the chosen path (only if tracing)

**File:** `matcher/matcher_stage1.py`, inside `match_stage1(...)`

* After calling `_search_with_skips(..., trace=trace)`, if `trace` and `best_state` exist, **walk back** parents from `best_state` to `None`, **collecting events in reverse order**. Do **not** emit them to `trace` yet; we want to **normalize** first.

* Determine **accept position**:

  * The accept event corresponds to the **last consumed token** (if unique-blocked) or the **last token** (if terminal). We already added an accept event in `_search_with_skips` parent chain for the accepted state. If not, emit it here using the derived `m_index`.

* Finally, **append** the ordered events to `trace.events`:

```python
if trace is not None and best_state is not None and parents is not None:
    ordered: List[Event] = []
    cur = best_state
    while cur is not None:
        info = parents[cur]
        ev = info.get("event")
        if ev:
            ordered.append(ev)
        cur = info.get("parent")
    ordered.reverse()

    # Ensure we have an ACCEPT_* at the end; if not, add ACCEPT_UNIQUE or ACCEPT_TERMINAL with at_m_index
    if not ordered or not ordered[-1]["action"].startswith("ACCEPT_"):
        # compute at_m_index from the terminal state’s consumed index
        # (You stored i when creating StateKey; reconstruct from it)
        # Fallback to ACCEPT_UNIQUE for now
        at_m_index = parents[best_state]["m_index_accept"]
        ordered.append({"action": "ACCEPT_UNIQUE", "uprn": best_uprn, "at_m_index": at_m_index})

    for ev in ordered:
        trace.add(ev)
```

**Note:** When you record a transition into `parents[next_state]`, also store any **auxiliary** fields you’ll need later (e.g., `m_index` at that step; for acceptance, also store `m_index_accept` on the accepting state).

**Verify:** For a pure exact path (`"4 LOVE LANE KINGS LANGLEY"`), events should include EXACT\_DESCEND for `LANGLEY, KINGS, LANE, LOVE, 4`, plus an ACCEPT event on `4`.

---

## Step 5 — Emit accept events with the correct flavor

**File:** `_search_with_skips` (accept logic)

* When `accept(node, i, ...)` is `True` and `trace` is enabled, determine the accept **type** and **where**:

  * `unique_blocked = node.count == 1 and (i >= n or not node.has_child(t[i]))`
  * `exact_exhausted = accept_terminal_if_exhausted and i >= n`

* Compute `at_m_index` as:

  * If at least one token was consumed (`i > 0`), use the **last consumed** token index: `m_index = (n-1) - (i-1)`.
  * If `i == 0` (edge case), set to `(n-1)`.

* Store **a parent entry** for the accepting state with:

  * `event = {"action": "ACCEPT_UNIQUE" or "ACCEPT_TERMINAL", "uprn": node.uprn, "at_m_index": m_index}`
  * plus `"parent": prev_state_key` and e.g. `"m_index_accept": m_index`.

**Verify:** With `"KIMS NAILS 4 LOVE LANE KINGS LANGLEY ..."`, accept happens on `4` (unique leaf). The ACCEPT event should carry `at_m_index` pointing to the `4` column.

---

## Step 6 — Add `SKIP_REDUNDANT` / `SKIP_PENALIZED` transitions into the parent map

**File:** `_search_with_skips`

* When generating the **skip** transition, compute `s_cost = skip_cost(node, tok)`. For the **child state** after skip, if `trace` is enabled, record a parent entry:

```python
if trace is not None:
    ev = {"action": "SKIP_REDUNDANT" if s_cost == 0 else "SKIP_PENALIZED",
          "messy": tok, "m_index": m_index}
    if s_cost == 0:
        ev.update({"anchor_count": node.count, "child_count": node.child_count(tok)})
    parents[next_key] = {"parent": cur_key, "event": ev}
```

* Do the same for `EXACT_DESCEND`:

```python
if trace is not None:
    ev = {"action": "EXACT_DESCEND",
          "messy": tok, "canon": tok, "m_index": m_index, "child_count": child.count}
    parents[next_key] = {"parent": cur_key, "event": ev}
```

**Verify:** With `"KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY ..."`, the chosen path should include a `SKIP_REDUNDANT` for `EXTRA`.

---

## Step 7 — Build and render the alignment table from the final path

**Files:** `try.py`

* Construct a `Trace`, pass it through `match_stage1(..., trace=trace)`.

* After calling, build and render:

```python
from matcher.trace_utils import build_alignment_table, render_alignment_text

tbl = build_alignment_table(addr.split(), trace.events)
print(render_alignment_text(tbl))
```

**Expected output** (columns aligned similarly to your sample):

```
Messy (R→L):   ENGLAND   HERTFORDSHIRE   LANGLEY    KINGS     LANE     EXTRA     LOVE     4      NAILS    KIMS
Action:          ⌫            ⌫             ✓         ✓         ✓        ·        ✓       ✓★       ·        ·
Canonical:        —            —          LANGLEY    KINGS     LANE       —      LOVE      4        —        —
Reason:        peel         peel        exact      exact     exact   redundant   exact   unique     post-   post-
                                                                                      leaf      accept   accept
```

**Verify:** Run with both:

* `KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND`
* `KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND`

…confirm that `EXTRA` is labeled “redundant” and star appears on `4`.

---

## Step 8 — Tests (lightweight)

**Files:** `tests/test_trace_alignment.py` (new)

* Create a tiny canonical Love Lane trie (as in your tests).

* Case 1 (no EXTRA): assert that:

  * Rightmost two columns are `⌫` + `peel`.
  * Columns for `LANGLEY,KINGS,LANE,LOVE,4` contain `✓` and `canonical` row equals those tokens.
  * `4` column has `✓★` and reason `unique leaf`.
  * Leftward tokens (`NAILS`, `KIMS`) are `post-accept`.

* Case 2 (with EXTRA): assert the `EXTRA` column is `·` with reason `redundant`.

---

## Step 9 — Keep trace optional & non-invasive

* Ensure all new params default to `None` and tracing **does not alter** match results.
* Keep **no prints** in matcher; only return events.
* Keep the `Trace` footprint minimal—no heavy objects in state keys.

---

## Step 10 — (Optional next) Add FUZZY events

When you’re ready:

* Emit `{"action":"FUZZY_CONSUME","messy": tok, "canon": child_lbl, "edit_type": "transpose|substitution|insert|delete", "m_index": m_index}`
* Extend `build_alignment_table`:

  * `action` remains `✓`
  * `reason` becomes `fuzzy:{edit_type}` (or short label like `fuzzy`)

Add small tests for a `HADYN`↔`HAYDN` swap.

---

## Developer notes (index math & acceptance)

* **Indexing:** when consuming `t[i]` (R→L), its L2R index is `m_index = (n-1) - i`. Use this consistently in events.
* **Acceptance star location:**

  * `unique_blocked`: star the **last consumed** token (`m_index = (n-1) - (i-1)`).
  * `terminal/exhausted`: same rule; if `i==0`, set to `n-1`.
* **Post-accept reason:** all columns **left** of the star in the R→L row (i.e., larger column indices) that remain unlabeled → `"post-accept"`.

---

## Definition of done

* `matcher/trace_utils.py` exists (collector + builder + renderer).
* `peel_end_tokens_with_trie` emits `PEEL_TAIL`.
* `_search_with_skips` records parents and events for **the chosen path** when `trace` is provided.
* `match_stage1` reconstructs and appends events to the `Trace`.
* `try.py` prints the final table identical in structure to the sample.
* Tests prove peel/descend/skip/accept mapping → table rows.

---

That’s it. This keeps the matching code clean (a few guarded `parents[...] = {...}` lines per transition), defers formatting to a single place, and gives you the exact table you asked for.
