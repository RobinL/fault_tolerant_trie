"""
Pytest configuration and shared fixtures.

- Adds project root to sys.path so tests can import the local package.
- Provides shared tries used across tests to avoid duplication.
"""
import os
import sys
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from matcher.trie_builder import build_trie_from_canonical

@pytest.fixture(scope="session")
def love_lane_root():
    canonical = [
        (1, ["5", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (2, ["9", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (3, ["8", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (4, ["7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (5, ["ANNEX", "7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (6, ["6", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (7, ["4", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    ]
    return build_trie_from_canonical(canonical, reverse=True)


@pytest.fixture(scope="session")
def haydn_root():
    canonical = [
        (1001, ["12", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
        (1002, ["10", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
    ]
    return build_trie_from_canonical(canonical, reverse=True)
