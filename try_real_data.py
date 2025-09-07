import duckdb
from matcher.get_data import (
    get_address_data_from_messy_address,
    get_random_address_data,
    OS_PARQUET,
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

# Default: enable token swap, canonical insertions, and unique-subtree acceptance
params = Params(
    allow_swap_adjacent=True,
    allow_canonical_insert=True,
    accept_unique_subtree_if_blocked=True,
)


def run_alignment(
    addr: str, *, title: str | None = None, params_override: Params | None = None
) -> None:
    if title:
        print(f"\n=== {title} ===\n")
    tokens = addr.split()
    trace = Trace(enabled=True)
    res = match_stage1(tokens, root, params_override or params, trace=trace)
    print(res)

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
        print(
            f"  Candidate UPRNs (≤{res.get('limit_used')}): {res.get('candidate_uprns')}"
        )


# Use the messy tokens from FHRS row
_uid, messy_tokens, _pc = messy_address
params_swap = Params(allow_swap_adjacent=True)

addr_text = " ".join(messy_tokens)
run_alignment(addr_text, params_override=params_swap)
