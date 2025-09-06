from get_data import get_random_address_data, get_address_data_from_messy_address
from trie_builder import build_trie_from_canonical, print_trie

messy_address, canonical_addresses = get_random_address_data(print_output=True)

addr = "20 Essex Close Bletchley, Milton Keynes"
pc = "MK3 7ET"

messy_address, canonical_addresses = get_address_data_from_messy_address(
    addr, pc, print_output=True
)

# canonical_addresses = [
#     (1, "5 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (2, "9 LOVE LANE KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (3, "8 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (4, "7 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (5, "ANNEX 7 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (6, "6 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
#     (7, "4 LOVE LANE, KINGS LANGLEY".split(" "), "WD4 9HW"),
# ]


root = build_trie_from_canonical(canonical_addresses[:10], reverse=True)  # suffix trie
print_trie(root)
