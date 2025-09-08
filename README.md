# Fault-Tolerant Trie Matcher

A prototype of a Stage‑1 UK address matcher built around a right‑to‑left suffix trie. The matcher consumes cleaned address tokens and returns a Unique Property Reference Number (UPRN) when a precise match is found under a small edit tolerance. The design emphasises correctness and clarity over performance and mirrors the logic destined for a future C++ implementation.

This diagram illustatrates the operation:

Given a trie:

```
-- BLETCHLEY (count=256)
    |-- CLOSE (count=153)
    |   |-- CARDIGAN (count=20)
    |   |   |-- 1 (count=1, uprn=a)
    |   |   |   ...
    |   |-- CUMBRIA (count=25)
    |   |   |-- 13 (count=1, uprn=b)
    |   |   |-- ...
    |   |   `-- HOUSE (count=13)
    |   |       `-- LANARK (count=13, uprn=c)
    |   |           |-- 1 (count=1, uprn=d)
    |   |           ...
    |   |-- ESSEX (count=31)
    |   |   ...
    |   |   |-- 20 (count=1, uprn=e)

```

```
Messy:      SUES NAILS 20 ESSEX CLOS NORTH BLETCHLEY MILTON KEYNES
Canonical:  20 ESSEX CLOSE BLETCHLEY
idx  messy      action  canonical  reason        condition                        
---  ---------  ------  ---------  ------------  ---------------------------------
0    KEYNES     ⌫       —          peel          KEYNES:0→BLETCHLEY:256 (k=2)     
1    MILTON     ⌫       —          peel                                           
2    BLETCHLEY  ✓       BLETCHLEY  exact         child_count=256, anchor_count=256
3    NORTH      ·       —          skip          ratio=256/1=256.00 ≥ 2.0         
4    CLOS       ✓       CLOSE      fuzzy:insert  edit=insert                      
5    ESSEX      ✓       ESSEX      exact         child_count=31, anchor_count=153 
6    20         ✓★      20         unique leaf   child_count=1, anchor_count=31   
7    NAILS      ·       —          post-accept                                    
8    SUES       ·       —          post-accept                                    

Result summary:
  matched=True uprn=25038431 cost=2
Consumed path (L→R): BLETCHLEY → CLOSE → ESSEX → 20
Counts along path:   256 → 153 → 31 → 1   | final=1
```

## Repository structure

- `matcher/trie_builder.py` – dataclass-based suffix trie with insert/count helpers and utilities for building tries from canonical addresses and rendering ASCII views.
- `matcher/matcher_stage1.py` – tail peeling, uniform-cost search and rule-based fault tolerance (exact, skip, fuzzy, canonical insert, swap). Exposes the public matching APIs.
- `matcher/trace_utils.py` – tracing primitives used to build alignment tables and consumed-path summaries for debugging.
- `matcher/get_data.py` – helpers to extract sample messy and canonical address data using DuckDB.
- `try.py` – demonstration script showing alignment output for common scenarios.
- `tests/` – pytest suite covering peeling, rule behaviour and end-to-end matching.

## Core algorithm

1. **Tail peeling** – deterministically remove redundant tokens from the end of the messy address when counts in the canonical trie show that stepping back joins a larger subtree.
2. **Uniform‑cost search** – traverse the trie right‑to‑left keeping total cost ≤2 and a margin of at least 1 between the best and runner‑up. The search state records the trie node, position, exact hits and numeric flags.
3. **Transition rules** – fault tolerance is expressed as small rule functions:
   - `exact` – consume an exact child token with cost 0.
   - `skip` – drop a messy token, cost 0 if clearly redundant, else 1.
   - `fuzzy` – Damerau–Levenshtein ≤1 consume when no exact child exists, cost 1.
   - `canonical_insert` – optionally insert one canonical token before consuming the current messy token.
   - `swap_adjacent` – optional swap of two adjacent messy tokens.
4. **Conservative acceptance** – only accept when reaching a UPRN under strict guards (minimum exact hits, numeric presence, and unique‑blocked or exact‑exhausted terminal).

Tracing events are collected during peeling and search to drive alignment tables and consumed-path summaries.

## Public API

```python
from matcher import build_trie_from_canonical, match_address, Params

# build a suffix trie from canonical rows (UPRN, tokens, postcode)
root = build_trie_from_canonical(canonical_rows, reverse=True)

# match a cleaned token list
result = match_address(tokens, root, Params())
if result["matched"]:
    print("UPRN:", result["uprn"])
```

- `Params` configures search thresholds (max cost, numeric requirements, optional swap/insert behaviours, candidate enumeration limits).
- `match_address(tokens, trie, params)` returns a dictionary describing the match decision, consumed path and optional candidate UPRNs when no match is found.
- `match_stage1(tokens, trie, params, trace)` offers the same matching logic but accepts a `Trace` instance for detailed alignment and diagnostics.

## Search parameters

`Params` is a dataclass of knobs that tune the search behaviour. All fields are optional and default to conservative values:

- `max_cost` *(2)* – maximum total edit cost permitted during search.
- `min_exact_hits` *(2)* – minimum number of exact token matches required to accept a UPRN.
- `require_numeric` *(True)* – demand that a numeric token is present and consumed if the input contains any numbers.
- `numeric_must_be_exact` *(True)* – the numeric requirement is satisfied only by an exact numeric match (no fuzzy numbers).
- `skip_redundant_ratio` *(2.0)* – skip a token at zero cost when dropping it increases the candidate pool by at least this ratio; otherwise skipping costs 1.
- `accept_terminal_if_exhausted` *(True)* – allow acceptance when all input tokens are consumed at a terminal node, even if siblings exist.
- `accept_unique_subtree_if_blocked` *(False)* – accept a node with `count == 1` when the next messy token has no matching child.
- `max_uprns_to_return` *(10)* – cap on UPRNs returned when enumerating candidates from a small no-match subtree.
- `allow_swap_adjacent` *(False)* – enable a rule that swaps two adjacent messy tokens.
- `swap_cost` *(1)* – cost applied to an adjacent swap.
- `allow_canonical_insert` *(True)* – permit insertion of a missing canonical token before consuming the current messy token.
- `canonical_insert_cost` *(1)* – cost of a canonical insert move.
- `canonical_insert_allow_fuzzy` *(False)* – after inserting, allow fuzzy matching of the next token instead of requiring an exact child.
- `canonical_insert_max_candidates` *(3)* – maximum number of canonical children considered for an insert.
- `canonical_insert_disallow_numeric` *(True)* – disable canonical inserts when the candidate label is numeric.

## Development and testing

Python 3.11+ with minimal dependencies. Create a virtual environment and install requirements:

```bash
uv venv
source .venv/bin/activate
uv pip install duckdb pytest
```

Run the test suite:

```bash
pytest -q
```

The `try.py` script provides smoke‑test scenarios and renders alignment tables for visual inspection.

