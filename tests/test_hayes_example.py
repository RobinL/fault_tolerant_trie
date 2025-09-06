import pytest

from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1, Params


@pytest.fixture()
def hayes_root():
    # Canonical tokens for UB3 2RJ (postcode itself will be stripped by builder)
    canonical = [
        # HAYES CRICKET CLUB, THE GREEN, WOOD END, HAYES
        (
            100023416072,
            [
                "HAYES",
                "CRICKET",
                "CLUB",
                "THE",
                "GREEN",
                "WOOD",
                "END",
                "HAYES",
            ],
            "UB3 2RJ",
        ),
        # HAYES CRICKET CLUB PAVILLION, THE GREEN, WOOD END, HAYES
        (
            10091101198,
            [
                "HAYES",
                "CRICKET",
                "CLUB",
                "PAVILLION",
                "THE",
                "GREEN",
                "WOOD",
                "END",
                "HAYES",
            ],
            "UB3 2RJ",
        ),
        # SMARTY'S NURSERY ... HAYES CRICKET CLUB PAVILLION, WOOD END, HAYES
        (
            10092982659,
            [
                "SMARTYS",
                "NURSERY",
                "HAYES",
                "HAYES",
                "CRICKET",
                "CLUB",
                "PAVILLION",
                "WOOD",
                "END",
                "HAYES",
            ],
            "UB3 2RJ",
        ),
        # Some numeric addresses on the street
        (100021440394, ["6", "WOOD", "END", "HAYES"], "UB3 2RJ"),
        (100021440395, ["7", "WOOD", "END", "HAYES"], "UB3 2RJ"),
        (100021440396, ["8", "WOOD", "END", "HAYES"], "UB3 2RJ"),
        (100021440393, ["5", "WOOD", "END", "HAYES"], "UB3 2RJ"),
    ]
    return build_trie_from_canonical(canonical, reverse=True)


def test_hayes_cricket_default_requires_numeric_rejects(hayes_root):
    messy = (
        "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES"
    ).split()
    res = match_stage1(messy, hayes_root)  # default Params require numeric
    assert res["matched"] is False


def test_hayes_cricket_accepts_when_numeric_guard_off(hayes_root):
    messy = (
        "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES"
    ).split()
    # Relax numeric guard for non‑residential POIs and slightly raise redundant skip
    # ratio so that 'PAVILLION' is not treated as 0‑cost redundant (ratio=2.0 at 'THE').
    params = Params(require_numeric=False, skip_redundant_ratio=2.01)
    res = match_stage1(messy, hayes_root, params)
    assert res["matched"] is True
    # Prefer the PAVILLION variant (exact longer suffix); expect cost 0
    assert res["uprn"] == 10091101198
    assert res["cost"] == 0


def test_hayes_cricket_tie_on_cost_rejects_by_default(hayes_root):
    # With numeric guard off but default skip_redundant_ratio=2.0, skipping
    # 'PAVILLION' at 'THE' is 0‑cost (ratio=2.0), creating a 0‑cost tie between
    # the PAVILLION and non‑PAVILLION variants → reject.
    messy = (
        "HAYES CRICKET CLUB (BAR) HAYES CRICKET CLUB PAVILLION THE GREEN WOOD END HAYES"
    ).split()
    params = Params(require_numeric=False)
    res = match_stage1(messy, hayes_root, params)
    assert res["matched"] is False
