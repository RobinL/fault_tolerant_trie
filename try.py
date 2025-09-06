from matcher.get_data import (
    get_random_address_data,
    get_address_data_from_messy_address,
)
from matcher.trie_builder import build_trie_from_canonical, print_trie, count_tail_L2R
from matcher.matcher_stage1 import (
    peel_end_tokens_with_trie,
    match_stage1_exact_only,
    match_stage1_with_skips,
)


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

print("\n=== Step 1: Love Lane trie (sanity check) ===")
print_trie(root)


# addr = "20 Essex Close Bletchley, Milton Keynes"
# pc = "MK3 7ET"

# addr = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
# pc = "WD4 9HW"

# messy_address, canonical_addresses = get_address_data_from_messy_address(
#     addr, pc, print_output=True
# )


def log(msg: str) -> None:
    print(msg)


print("\n=== Step 2: Peeling demo ===")
messy_str = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
print("Input:", messy_str)
peeled = peel_end_tokens_with_trie(messy_str.split(), root, steps=4, max_k=2, debug=log)
print("Peeled:", " ".join(peeled))


print("\n=== Step 3: Exact walk demo (with trace) ===")
for addr in [
    "4 LOVE LANE KINGS LANGLEY",
    "7 LOVE LANE KINGS LANGLEY",
    "ANNEX 7 LOVE LANE KINGS LANGLEY",
]:
    print(f"\nInput: {addr}")
    uprn = match_stage1_exact_only(addr.split(), root)
    # Re-run with explicit exact trace to show steps
    _ = match_stage1_exact_only(addr.split(), root)  # quiet
    from matcher.matcher_stage1 import walk_exact

    _ = walk_exact(addr.split(), root, accept_terminal_if_exhausted=True, debug=log)
    print(f"Result UPRN: {uprn}")


print("\n=== Step 5: Skips (search) demo with debug ===")
for addr in [
    "4 LOVE EXTRA LANE KINGS LANGLEY",  # inner noise → skip
    "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE",  # business + redundant county
]:
    print(f"\nInput: {addr}")
    # Show peel step first
    _ = peel_end_tokens_with_trie(addr.split(), root, steps=4, max_k=2, debug=log)
    # Then run the skip-enabled matcher with a concise expansion trace
    uprn = match_stage1_with_skips(addr.split(), root, debug=log)
    print(f"Result UPRN: {uprn}")


# --- Step 6: Fuzzy consume (DL≤1) demo ---
print("\n=== Step 6: Fuzzy consume demo (DL≤1) ===")
canonical_haydn = [
    (1001, ["12", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
    (1002, ["10", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
]
root_h = build_trie_from_canonical(canonical_haydn, reverse=True)
for addr in [
    "12 HADYN PARK ROAD",  # transposition: AD vs DA
    "10 HAYEN PARK ROAD",  # substitution: DN vs EN
    "HADYN PARK ROAD",     # no numeric → guard blocks
]:
    print(f"\nInput: {addr}")
    res = match_stage1_with_skips(addr.split(), root_h, debug=log)
    print(f"Result UPRN: {res}")
