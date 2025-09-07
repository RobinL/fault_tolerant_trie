import duckdb
from matcher.get_data import (
    get_address_data_from_messy_address,
    get_random_address_data,
    OS_PARQUET,
    show_uprns,
)
from matcher.trie_builder import build_trie_from_canonical
from matcher.matcher_stage1 import (
    match_stage1,
    Params,
)
from matcher.trace_utils import (
    Trace,
    build_alignment_table,
    render_alignment_text,
    render_consumed_summary,
)


messy_address, canonical_addresses = get_random_address_data(print_output=False)

# addr = "SUES NAILS 20 Essex Close Bletchley, Milton Keynes, UK"
# pc = "MK3 7ET"

# messy_address, canonical_addresses = get_address_data_from_messy_address(
#     addr, pc, print_output=False
# )


# Build suffix trie from canonical rows for this postcode block
root = build_trie_from_canonical(canonical_addresses, reverse=True)

# Build uprn -> canonical tokens map for pretty printing
uprn_to_tokens = {}
for row in canonical_addresses:
    try:
        uprn = int(row[0])
        toks = [str(t) for t in row[1]]
        uprn_to_tokens[uprn] = toks
    except Exception:
        pass

# Default: permissive settings for exploration. Explicitly set all Params.
params = Params(
    max_cost=2,
    min_exact_hits=1,  # more permissive (was 2)
    require_numeric=False,  # more permissive: don’t require numbers
    numeric_must_be_exact=False,  # irrelevant if require_numeric=False
    skip_redundant_ratio=1.8,  # slightly lower threshold for 0-cost skip
    accept_terminal_if_exhausted=True,
    accept_unique_subtree_if_blocked=True,  # accept unique subtree when blocked
    max_uprns_to_return=50,  # show more candidates on no-match
    allow_swap_adjacent=True,  # enable adjacent token swap
    swap_cost=1,
    allow_canonical_insert=True,  # enable canonical insertions
    canonical_insert_cost=1,
    canonical_insert_allow_fuzzy=False,  # not implemented yet
    canonical_insert_max_candidates=5,  # allow a few insert candidates
    canonical_insert_disallow_numeric=False,  # allow numeric inserts if helpful
)


def run_alignment(
    addr: str, *, title: str | None = None, params_override: Params | None = None
) -> None:
    if title:
        print(f"\n=== {title} ===\n")
    tokens = addr.split()
    trace = Trace(enabled=True)
    res = match_stage1(tokens, root, params_override or params, trace=trace)
    # Show params for transparency
    print("Params:", params_override or params)

    # Main result: show messy vs canonical clearly at the top
    messy_line = " ".join(tokens)
    if res.get("matched") and res.get("uprn") is not None:
        canon_tokens = uprn_to_tokens.get(int(res["uprn"]))
        canon_line = (
            " ".join(canon_tokens) if canon_tokens else "(canonical tokens unavailable)"
        )
    else:
        canon_line = "(no match)"
    print("Messy:     ", messy_line)
    print("Canonical: ", canon_line)
    tbl = build_alignment_table(tokens, trace.events)
    print(render_alignment_text(tbl))
    print("\nResult summary:")
    print(
        f"  matched={res.get('matched')} uprn={res.get('uprn')} cost={res.get('cost')}"
    )
    print(
        render_consumed_summary(
            res.get("consumed_path", []),
            res.get("consumed_path_counts", []),
            res.get("final_node_count"),
        )
    )
    if "candidate_uprns" in res:
        cands = res.get("candidate_uprns") or []
        print(f"  Candidate UPRNs (≤{res.get('limit_used')}): {cands}")
        if not res.get("matched") and cands:
            print("\nCandidate UPRN details (from OS AddressBase):")
            show_uprns(cands)


# Use the messy tokens from FHRS row
_uid, messy_tokens, _pc = messy_address
params_swap = Params(allow_swap_adjacent=True, accept_unique_subtree_if_blocked=True)

addr_text = " ".join(messy_tokens)
run_alignment(addr_text, params_override=params_swap)
