from get_data import get_random_address_data, get_address_data_from_messy_address
from trie_builder import build_trie_from_canonical, print_trie, count_tail_L2R
from matcher_stage1 import peel_end_tokens_with_trie


def verify_step1_helpers() -> None:
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

    # Root should only have 'LANGLEY' with count 7
    print("root.has_child('LANGLEY'):", root.has_child("LANGLEY"))
    print("root.child_count('LANGLEY'):", root.child_count("LANGLEY"))
    assert root.has_child("LANGLEY")
    assert root.child_count("LANGLEY") == 7

    # Wrapper count checks
    c1 = count_tail_L2R(root, ["LANGLEY"])  # expect 7
    c2 = count_tail_L2R(root, ["LANGLEY", "HERTFORDSHIRE", "ENGLAND"])  # expect 0
    print("count_tail_L2R(['LANGLEY']):", c1)
    print("count_tail_L2R(['LANGLEY','HERTFORDSHIRE','ENGLAND']):", c2)
    assert c1 == 7
    assert c2 == 0

    # Check deeper structure: under LOVE, '7' has 2 (plain 7 + ANNEX 7)
    node = root.child("LANGLEY").child("KINGS").child("LANE").child("LOVE")
    print("LOVE children counts:", {k: v.count for k, v in node.iter_children()})
    assert node.child_count("7") == 2

    print("Step 1 helpers verified OK.\n")


def verify_step2_peeling() -> None:
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


def run_data_demo() -> None:
    messy_address, canonical_addresses = get_random_address_data(print_output=True)

    addr = "20 Essex Close Bletchley, Milton Keynes"
    pc = "MK3 7ET"

    messy_address, canonical_addresses = get_address_data_from_messy_address(
        addr, pc, print_output=True
    )

    root = build_trie_from_canonical(
        canonical_addresses[:10], reverse=True
    )  # suffix trie
    print_trie(root)


if __name__ == "__main__":
    # Always run the small, deterministic verifications (no external data required).
    verify_step1_helpers()
    verify_step2_peeling()

    # Then try the original data-backed demo; ignore if local data not available.
    try:
        run_data_demo()
    except Exception as e:
        print("Data demo skipped (error):", e)
