from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import (
    peel_end_tokens_with_trie,
    match_stage1_exact_only,
    match_stage1_with_skips,
    match_stage1,
    Params,
)
from matcher.trace_utils import Trace, build_alignment_table, render_alignment_text


# messy_address, canonical_addresses = get_random_address_data(print_output=True)

# addr = "20 Essex Close Bletchley, Milton Keynes"
# pc = "MK3 7ET"

# messy_address, canonical_addresses = get_address_data_from_messy_address(
#     addr, pc, print_output=True
# )

# root = build_trie_from_canonical(
#     canonical_addresses[:10], reverse=True
# )  # suffix trie
# print_trie(root)


canonical_love_lane = [
    (1, ["5", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (2, ["9", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (3, ["8", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (4, ["7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (5, ["ANNEX", "7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (6, ["6", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    (7, ["4", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
]
root = build_trie_from_canonical(canonical_love_lane, reverse=True)


# addr = "20 Essex Close Bletchley, Milton Keynes"
# pc = "MK3 7ET"

# addr = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
# pc = "WD4 9HW"

# messy_address, canonical_addresses = get_address_data_from_messy_address(
#     addr, pc, print_output=True
# )


# --- Step 7: Stage‑1 API (structured result) demo ---
params = Params()  # defaults: strict guards, numeric must be exact


def run_alignment(
    addr: str, *, params_override: Params | None = None, title: str | None = None
) -> None:
    if title:
        print(f"\n=== {title} ===\n")
    tokens = addr.split()
    trace = Trace(enabled=True)

    # Then run matcher and show full alignment
    _ = match_stage1(tokens, root, params_override or params, trace=trace)
    tbl = build_alignment_table(tokens, trace.events)
    print()
    print(render_alignment_text(tbl))


# Case 1: baseline success (unique leaf on 4)
addr1 = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr1, title="Success: unique leaf at 4")

print("\n" + "-" * 80 + "\n")

# Case 2: redundant skip (EXTRA between LOVE and LANE)
addr2 = "KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr2, title="Redundant skip: EXTRA between LOVE and LANE")

print("\n" + "-" * 80 + "\n")

# Case 3: terminal accept on 7
addr3 = "KIMS NAILS 7 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr3, title="Terminal accept: exhausted at 7")

print("\n" + "-" * 80 + "\n")

# Case 4: penalized skip by raising redundancy threshold → EXTRA counted as skip
addr4 = "KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
params_penalized = Params(skip_redundant_ratio=100.0)
run_alignment(
    addr4,
    params_override=params_penalized,
    title="Penalized skip: EXTRA counted as skip (not redundant)",
)

print("\n" + "-" * 80 + "\n")

# Case 5: stop with no child (unknown house number)
addr5 = "KIMS NAILS 500 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr5, title="Stop: no child for 500 (no-child)")

print("\n" + "-" * 80 + "\n")

# Case 6: stop incomplete (no number present)
addr6 = "KIMS NAILS LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr6, title="Stop: incomplete (no terminal UPRN)")

print("\n" + "-" * 80 + "\n")

# Case 7: fuzzy consume (adjacent transpose KINSG → KINGS)
addr7 = "KIMS NAILS 4 LOVE LANE KINSG LANGLEY HERTFORDSHIRE ENGLAND"
run_alignment(addr7, title="Fuzzy: transpose KINSG → KINGS")

print("\n" + "-" * 80 + "\n")

# Case 8: skip after accept scenario (ANNEXE after 7)
addr8 = "ANNEXE 7 LOVE LANE KINGS LANGLEY"
run_alignment(addr8, title="Skip after accept: ANNEXE after 7 (star under 7)")


addr9 = "ANNEX 7 LOVE LANE KINGS LANGLEY"
run_alignment(addr9)
