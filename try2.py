


from matcher.trie_builder import build_trie_from_canonical, print_trie
from matcher.matcher_stage1 import (
    peel_end_tokens_with_trie,
    match_stage1_exact_only,
    match_stage1_with_skips,
    match_stage1,
    Params,
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


# === Hayes Cricket Club canonical data ===
canonical_hayes = [
    (100023416072, ["HAYES", "CRICKET", "CLUB", "THE", "GREEN", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    (10091101198, ["HAYES", "CRICKET", "CLUB", "PAVILLION", "THE", "GREEN", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    (
        10092982659,
        ["SMARTYS", "NURSERY", "SMARTYS", "NURSERY", "HAYES", "HAYES", "CRICKET", "CLUB", "PAVILLION", "WOOD", "END", "HAYES"],
        "UB3 2RJ",
    ),
    (100021440394, ["6", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    (100021440395, ["7", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    (100021440396, ["8", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    (100021440393, ["5", "WOOD", "END", "HAYES"], "UB3 2RJ"),
]
root = build_trie_from_canonical(canonical_hayes, reverse=True)

print("\n=== Step 1: Hayes CC trie (sanity check) ===")
print_trie(root)


def log(msg: str) -> None:
    print(msg)


print("\n=== Step 2: Peeling demo ===")
messy_str = "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES UB3 2RJ"
print("Input:", messy_str)
peeled = peel_end_tokens_with_trie(messy_str.split(), root, steps=4, max_k=2, debug=log)
print("Peeled:", " ".join(peeled))


print("\n=== Step 3: Exact walk demo (with trace) ===")
for addr in [
    "HAYES CRICKET CLUB THE GREEN WOOD END HAYES",
    "HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES",
    "SMARTYS NURSERY SMARTYS NURSERY HAYES HAYES CRICKET CLUB PAVILLION WOOD END HAYES",
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
    "HAYES CRICKET CLUB PAVILLION THE GREEN EXTRA WOOD END HAYES",  # inner noise → skip
    "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES HERTFORDSHIRE",  # business + redundant county
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
    "HADYN PARK ROAD",  # no numeric → guard blocks
]:
    print(f"\nInput: {addr}")
    res = match_stage1_with_skips(addr.split(), root_h, debug=log)
    print(f"Result UPRN: {res}")


# --- Step 7: Stage-1 API (structured result) demo ---
print("\n=== Step 7: Stage-1 API demo (structured result) ===")
params = Params(numeric_must_be_exact=False)  # defaults: strict guards, numeric must be exact

for addr in [

    "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES",

]:
    print(f"\nInput: {addr}")
    result = match_stage1(addr.split(), root, params, debug=False)
    print("Structured result:", result)




HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES