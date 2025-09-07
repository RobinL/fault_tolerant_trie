from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import peel_end_tokens_with_trie, match_stage1, Params
from matcher.trace_utils import Trace, build_alignment_table


def _build_love_lane_root():
    canonical_love_lane = [
        (1, ["5", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (2, ["9", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (3, ["8", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (4, ["7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (5, ["ANNEX", "7", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (6, ["6", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
        (7, ["4", "LOVE", "LANE", "KINGS", "LANGLEY"], "WD4 9HW"),
    ]
    return build_trie_from_canonical(canonical_love_lane, reverse=True)


def _table_for(addr: str, root):
    tokens = addr.split()
    tr = Trace(enabled=True)
    _ = peel_end_tokens_with_trie(tokens, root, steps=4, max_k=2, trace=tr)
    _ = match_stage1(tokens, root, Params(), trace=tr)
    tbl = build_alignment_table(tokens, tr.events)
    return tokens, tbl


def test_alignment_success_no_extra():
    root = _build_love_lane_root()
    addr = "KIMS NAILS 4 LOVE LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
    tokens, tbl = _table_for(addr, root)

    r2l = list(reversed(tokens))
    action = tbl["action"]
    canon = tbl["canonical"]
    reason = tbl["reason"]

    # Rightmost two columns (R→L) are ENGLAND, HERTFORDSHIRE → peeled
    assert r2l[0] == "ENGLAND" and action[0] == "⌫" and reason[0] == "peel"
    assert r2l[1] == "HERTFORDSHIRE" and action[1] == "⌫" and reason[1] == "peel"

    # Exact columns for LANGLEY, KINGS, LANE, LOVE, and star on 4
    idx_lang = r2l.index("LANGLEY")
    idx_kings = r2l.index("KINGS")
    idx_lane = r2l.index("LANE")
    idx_love = r2l.index("LOVE")
    idx_num = r2l.index("4")

    for idx, tok in [(idx_lang, "LANGLEY"), (idx_kings, "KINGS"), (idx_lane, "LANE"), (idx_love, "LOVE")]:
        assert action[idx] == "✓"
        assert canon[idx] == tok
        assert reason[idx] == "exact"

    assert action[idx_num] == "✓★"
    assert canon[idx_num] == "4"
    assert reason[idx_num] in ("unique leaf", "terminal")

    # Left of star (R→L), i.e., columns beyond the star, should mark post-accept
    for k in range(idx_num + 1, len(r2l)):
        assert reason[k] == "post-accept"


def test_alignment_with_extra_penalized():
    root = _build_love_lane_root()
    addr = "KIMS NAILS 4 LOVE EXTRA LANE KINGS LANGLEY HERTFORDSHIRE ENGLAND"
    tokens, tbl = _table_for(addr, root)

    r2l = list(reversed(tokens))
    action = tbl["action"]
    canon = tbl["canonical"]
    reason = tbl["reason"]

    # EXTRA column should be a penalized skip (dot with reason 'skip')
    idx_extra = r2l.index("EXTRA")
    assert action[idx_extra] == "·"
    assert reason[idx_extra] == "skip"
    assert canon[idx_extra] == "—"
