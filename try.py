import duckdb
from get_data import get_random_address_data, get_address_data_from_messy_address
from trie_builder import build_trie_from_canonical, print_trie

messy_address, canonical_addresses = get_random_address_data()

addr = "20 Essex Close Bletchley, Milton Keynes"
pc = "MK3 7ET"

messy_address, canonical_addresses = get_address_data_from_messy_address(
    addr, pc, print_output=False
)


root = build_trie_from_canonical(canonical_addresses[:10], reverse=True)  # suffix trie
print_trie(root)
