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


addr = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"


addr = "500 LOVE LANE KINGS LANGLEY ENGLAND"

tokens = addr.split()

# Step 2 verification: demonstrate PEEL_TAIL tracing
trace = Trace(enabled=True)
_peeled = peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2, trace=trace)
tbl = build_alignment_table(tokens, trace.events)
print(render_alignment_text(tbl))

# Run Stage‑1 with tracing to add EXACT_DESCEND/ACCEPT events, then render again
_ = match_stage1(tokens, root, params, trace=trace)
tbl = build_alignment_table(tokens, trace.events)
print()
print(render_alignment_text(tbl))
