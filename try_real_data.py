import duckdb
from matcher.get_data import (
    get_address_data_from_messy_address,
    get_random_address_data,
    OS_PARQUET,
)
from matcher.trie_builder import build_trie_from_canonical, print_trie
from matcher.matcher_stage1 import (
    match_stage1,
    Params,
)


messy_address, canonical_addresses = get_random_address_data(print_output=True)

# addr = "SUES NAILS 20 Essex Close Bletchley, Milton Keynes, UK"
# pc = "MK3 7ET"

# messy_address, canonical_addresses = get_address_data_from_messy_address(
#     addr, pc, print_output=False
# )


# --- Build suffix trie from canonical addresses for this postcode block ---
root = build_trie_from_canonical(canonical_addresses, reverse=True)
messy_address


def log(msg: str) -> None:
    print(msg)


# Extract the messy tokens from the input row and run Stage‑1 matcher
messy_tokens = list(messy_address[1])
print("\n=== Stage‑1 match on real data ===")
print("Messy cleaned:", " ".join(messy_tokens) + " " + messy_address[2])
res = match_stage1(messy_tokens, root, Params(), debug=False)

if res["matched"]:
    print("Match:")
    print(" ".join([a for a in canonical_addresses if a[0] == res["uprn"]][0][1]))
else:
    print("no match")

# in_postcode = [c for c in canonical_addresses if c[2] == messy_address[2]]
# [(a[0], " ".join(a[1])) for a in in_postcode]

# import duckdb
# ddbdf = duckdb.read_parquet(OS_PARQUET)
# sql = f"""
# select fulladdress
# from ddbdf
# where postcode = '{messy_address[2]}'
# """
# duckdb.sql(sql).show()
# res
