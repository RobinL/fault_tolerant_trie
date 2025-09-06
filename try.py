from get_data import get_random_address_data, get_address_data_from_messy_address
from trie_builder import build_trie_from_canonical, print_trie, count_tail_L2R
from matcher_stage1 import peel_end_tokens_with_trie


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


messy = [
    "KIMS",
    "NAILS",
    "4",
    "LOVE",
    "LANE",
    "KINGS",
    "LANGLEY",
    "HERTFORDSHIRE",
    "ENGLAND",
]
peeled = peel_end_tokens_with_trie(messy, root, steps=4, max_k=2)
print("Original tokens:", messy)
print("Peeled tokens:  ", peeled)
assert peeled == [
    "KIMS",
    "NAILS",
    "4",
    "LOVE",
    "LANE",
    "KINGS",
    "LANGLEY",
]

# No redundant tail → unchanged
no_tail = ["4", "LOVE", "LANE", "KINGS", "LANGLEY"]
no_tail_after = peel_end_tokens_with_trie(no_tail, root)
print("No-tail input:  ", no_tail)
print("No-tail output: ", no_tail_after)
assert no_tail_after == no_tail

print("Step 2 peeling verified OK.\n")
