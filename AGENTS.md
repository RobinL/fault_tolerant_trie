# AGENTS.md

A tiny Python project to prototype **Stage-1 UK address matching** using a **right-to-left suffix trie** over canonical addresses.

!!Priority: a **clear working version**, not performance.
!!Priority: code is clear and split into well-named functions i.e. the implementation is understandable by a HUMAN



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

---

## Quick start

```bash
# Python 3.11+ recommended

# With uv (preferred)
uv venv
source .venv/bin/activate
uv pip install duckdb pytest

# Or with pip
python -m venv .venv
source .venv/bin/activate
pip install duckdb pytest
```

Run the demo:

```bash
python try.py
```

NOTE IMPORTANTLY:  we are using the vs code interactive window and ipykernel to run the code so in general code will be run in try.py by selecting it and shift enter, NOT by typing python

That means we DO NOT

> The demo loads a random messy/canonical pair or a provided one, builds a small suffix trie from OS AddressBase rows (same postcode group), and prints an ASCII tree.

---

## Repo layout

```
get_data.py        # pulls sample rows from local Parquet (FHRS + OS AddressBase)
trie_builder.py    # minimal token-trie (right→left), insert/count/pretty-print
try.py             # tiny script wiring data + trie for smoke testing
```

Data paths used by `get_data.py`:

* `OS_PARQUET` and `FHRS_PATH` point to local Parquet files.
* Prefer passing paths via function args to avoid editing constants.

---

## What to build (Stage-1 matcher)

Extend  try.py to build a funciton:

```python
match_address(tokens: list[str], trie) -> dict | None
```



## Code style & conventions

* Python with **type hints**; keep functions short and explicit.
* Favor pure functions and small dataclasses for state.
* No heavy deps; standard lib + `duckdb` only.
* Keep constants & thresholds at the top of the module.

---



## Guardrails (do not relax)

* **Never** return a match unless at a **unique UPRN**.
* Keep edit budget **≤ 2** in Stage-1.
* Require **numeric anchor** and **≥2 exact tokens** on accept.
* If multiple candidates tie on cost → **reject** (send to Stage-2).
* Do **not** add heavy global alignment/ML here (that’s Stage-2).

---

## Data & privacy

* Paths in `get_data.py` point to local files; **do not commit** data.
* If you need to change paths, pass them as function parameters (don’t hard-code secrets).

---

## Suggested next tasks for agents

1. Implement `peel_end_tokens(...)` (from the prompt) as a utility.
2. Implement `stage1_matcher.py` with the bounded costed walk (exact + tiny fuzzy + skip).
3. Wire `try.py` to run `match_address(...)` and print the chosen UPRN (or “No match”).
4. Add unit tests for the cases above.
5. Keep thresholds configurable via a small `Params` dataclass.

---
