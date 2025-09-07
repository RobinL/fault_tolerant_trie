import pytest
from matcher.trie_builder import (
    TrieNode,
    build_trie,
    build_trie_from_canonical,
)



def test_counts_and_uprns_for_canonical(love_lane_root: TrieNode):
    # Root should not include postcode tokens as children
    assert not love_lane_root.has_child("WD4")
    assert not love_lane_root.has_child("9HW")

    # Root anchor token is LANGLEY with total count 7
    assert love_lane_root.has_child("LANGLEY")
    assert love_lane_root.child_count("LANGLEY") == 7

    kings = love_lane_root.child("LANGLEY").child("KINGS")
    lane = kings.child("LANE")
    love = lane.child("LOVE")

    # '7' has 2 paths (plain 7 and ANNEX 7); '4' has 1 path
    c7 = love.child("7")
    c4 = love.child("4")
    assert c7 is not None and c7.count == 2 and c7.uprn == 4
    assert c4 is not None and c4.count == 1 and c4.uprn == 7

    # ANNEX under 7 is a single terminating path with its own UPRN
    annex = c7.child("ANNEX")
    assert annex is not None and annex.count == 1 and annex.uprn == 5


def test_has_path_and_count_for_path(love_lane_root: TrieNode):
    # Paths must be provided R2L to match internal trie orientation
    path_r2l_4 = ["LANGLEY", "KINGS", "LANE", "LOVE", "4"]
    assert love_lane_root.has_path(path_r2l_4)

    path_r2l_7 = ["LANGLEY", "KINGS", "LANE", "LOVE", "7"]
    path_r2l_7_annex = path_r2l_7 + ["ANNEX"]
    assert love_lane_root.count_for_path(path_r2l_7) == 2
    assert love_lane_root.count_for_path(path_r2l_7_annex) == 1

    # Negative cases
    assert not love_lane_root.has_path(["LANGLEY", "KINGS", "LANE", "LOVES"])  # misspelled
    assert love_lane_root.count_for_path(["LANGLEY", "KINGS", "LANE", "LOVE", "10"]) == 0


def test_build_trie_reverse_flag():
    paths = [["A", "B"], ["A", "C"]]

    # reverse=False → root children are first tokens ('A')
    root_l2r = build_trie(paths, reverse=False)
    assert root_l2r.has_child("A")
    assert root_l2r.child_count("A") == 2
    assert root_l2r.child("A").has_child("B")
    assert root_l2r.child("A").has_child("C")

    # reverse=True → root children are last tokens ('B' and 'C')
    root_r2l = build_trie(paths, reverse=True)
    assert root_r2l.has_child("B")
    assert root_r2l.has_child("C")
    assert root_r2l.child_count("B") == 1
    assert root_r2l.child_count("C") == 1


def test_root_count_after_inserts():
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
    assert root.count == len(canonical_love_lane)
