from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import match_stage1, Params
from matcher.trace_utils import Trace, build_alignment_table


def _build_haydn_root():
    canonical = [
        (1001, ["12", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
        (1002, ["10", "HAYDN", "PARK", "ROAD"], "W12 3AB"),
    ]
    return build_trie_from_canonical(canonical, reverse=True)


def test_fuzzy_transpose_alignment_reason():
    root = _build_haydn_root()
    addr = "12 HADYN PARK ROAD"  # transpose of AY → YA in HAYDN
    tokens = addr.split()

    tr = Trace(enabled=True)
    _ = match_stage1(tokens, root, Params(), trace=tr)
    tbl = build_alignment_table(tokens, tr.events)

    # Find the column for HADYN (messy) and ensure it shows fuzzy:transpose with canonical HAYDN
    r2l = list(reversed(tokens))
    idx = r2l.index("HADYN")
    assert tbl["action"][idx] == "✓"
    assert tbl["canonical"][idx] == "HAYDN"
    assert tbl["reason"][idx].startswith("fuzzy:")
