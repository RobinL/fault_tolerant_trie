from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# Simple alias for trace events
Event = Dict[str, Any]


@dataclass
class Trace:
    enabled: bool = True
    tokens_l2r: Optional[List[str]] = None
    events: List[Event] = field(default_factory=list)

    def set_tokens(self, tokens_l2r: Sequence[str]) -> None:
        if self.enabled:
            self.tokens_l2r = list(tokens_l2r)

    def add(self, event: Event) -> None:
        if self.enabled:
            self.events.append(event)


ICONS = {
    "exact": "✓",
    "star": "✓★",
    "peel": "⌫",
    "dot": "·",
    "stop": "×",
}

DASH = "—"


def build_alignment_table(tokens_l2r: Sequence[str], events: List[Event]) -> Dict[str, List[str]]:
    """
    Build a column-aligned table showing the alignment between messy tokens and
    canonical consumption actions. Orientation is R→L for display.

    Returns a dict with four equal-length lists:
      - "messy_r2l": tokens printed R→L
      - "action":    icon per column (⌫, ✓, ✓★, ·)
      - "canonical": canonical token aligned under exact matches; DASH otherwise
      - "reason":    short reason labels
    """
    n = len(tokens_l2r)

    # Initialize columns
    messy_r2l = list(reversed(list(tokens_l2r)))
    action = [ICONS["dot"]] * n
    canonical = [DASH] * n
    reason = [""] * n
    condition = [""] * n

    def col_from_m_index(m_index: int) -> int:
        # Column 0 is the rightmost messy token (L2R index n-1)
        return (n - 1) - m_index

    def set_if_empty(idx: int, text: str) -> None:
        if 0 <= idx < n and not condition[idx]:
            condition[idx] = str(text)

    # First, mark peeled tail (rightmost k columns in R→L)
    for ev in events:
        if ev.get("action") == "PEEL_TAIL":
            k = int(ev.get("k", 0))
            for j in range(k):  # j=0 => rightmost
                idx = j  # R→L index
                action[idx] = ICONS["peel"]
                reason[idx] = "peel"
            # Attach decision summary to the rightmost column
            base_a = ev.get("base_anchor")
            base_c = ev.get("base_score")
            best_a = ev.get("best_anchor")
            best_c = ev.get("best_score")
            if k > 0 and base_a is not None and best_a is not None:
                set_if_empty(0, f"{base_a}:{base_c}→{best_a}:{best_c} (k={k})")

    # Next, mark exact / skip events
    for ev in events:
        a = ev.get("action")
        if a == "EXACT_DESCEND":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["exact"]
            canonical[j] = str(ev.get("canon", DASH))
            reason[j] = "exact"
            cc = ev.get("child_count")
            ac = ev.get("anchor_count")
            if cc is not None:
                set_if_empty(j, f"child_count={cc}" + (f", anchor_count={ac}" if ac is not None else ""))
        elif a == "FUZZY_CONSUME":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["exact"]
            canonical[j] = str(ev.get("canon", DASH))
            edit_type = ev.get("edit_type")
            reason[j] = f"fuzzy:{edit_type}" if edit_type else "fuzzy"
            if edit_type:
                set_if_empty(j, f"edit={edit_type}")
        elif a == "SKIP_REDUNDANT":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "redundant"
            ac = ev.get("anchor_count")
            cc = ev.get("child_count")
            ratio = ev.get("ratio")
            thr = ev.get("threshold")
            if ratio is not None and thr is not None and ac is not None and cc is not None:
                cmp = "≥" if float(ratio) >= float(thr) else "<"
                set_if_empty(j, f"ratio={ac}/{max(1,int(cc))}={float(ratio):.2f} {cmp} {float(thr)}")
        elif a == "SKIP_PENALIZED":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "skip"
            ac = ev.get("anchor_count")
            cc = ev.get("child_count")
            ratio = ev.get("ratio")
            thr = ev.get("threshold")
            if ratio is not None and thr is not None and ac is not None and cc is not None:
                cmp = "≥" if float(ratio) >= float(thr) else "<"
                set_if_empty(j, f"ratio={ac}/{max(1,int(cc))}={float(ratio):.2f} {cmp} {float(thr)}")
        elif a and str(a).startswith("STOP_"):
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["stop"]
            reason_map = {
                "STOP_NO_CHILD": "no-child",
                "STOP_INCOMPLETE": "incomplete",
                "STOP_AMBIGUOUS": "ambiguous",
                "STOP_GUARD_NUMERIC": "guard:numeric",
                "STOP_GUARD_MIN_EXACT": "guard:exact",
                "STOP_BUDGET": "budget",
                "STOP_TIE": "tie",
                "STOP_UNKNOWN": "stop",
            }
            reason[j] = reason_map.get(a, "stop")
            if a == "STOP_NO_CHILD":
                tok = ev.get("messy")
                ac = ev.get("anchor_count")
                set_if_empty(j, f"next '{tok}' not child (anchor_count={ac})")
            elif a == "STOP_GUARD_MIN_EXACT":
                hits = ev.get("exact_hits"); need = ev.get("min_exact_hits")
                set_if_empty(j, f"hits={hits} < min_exact={need}")
            elif a == "STOP_GUARD_NUMERIC":
                mex = ev.get("numeric_must_be_exact")
                saw = ev.get("saw_num_exact" if mex else "saw_num_any")
                set_if_empty(j, f"need numeric{' exact' if mex else ''}; seen={bool(saw)}")
            elif a == "STOP_AMBIGUOUS":
                cnt = ev.get("node_count")
                set_if_empty(j, f"count={cnt} (>1)")
            elif a == "STOP_INCOMPLETE":
                cnt = ev.get("node_count")
                set_if_empty(j, f"incomplete (count={cnt})")

    # Finally, mark acceptance star at the consumed token column
    for ev in events:
        if ev.get("action") in ("ACCEPT_UNIQUE", "ACCEPT_TERMINAL"):
            j = col_from_m_index(int(ev["at_m_index"]))
            action[j] = ICONS["star"]
            # If we know which canonical label was accepted, show it in the star column
            if ev.get("accepted_label"):
                canonical[j] = str(ev.get("accepted_label"))
            reason[j] = "unique leaf" if ev["action"] == "ACCEPT_UNIQUE" else "terminal"
            if ev.get("action") == "ACCEPT_UNIQUE":
                cnt = ev.get("node_count"); nxt = ev.get("next_child_exists")
                set_if_empty(j, f"unique (count={cnt}, next_child={bool(nxt)})")
            else:
                hits = ev.get("exact_hits"); saw = ev.get("saw_num_exact")
                set_if_empty(j, f"terminal (hits={hits}, num_exact={bool(saw)})")

            # Mark post-accept tokens to the right in the R→L display (override reasons)
            for k in range(j + 1, n):
                reason[k] = "post-accept"
                action[k] = ICONS["dot"]
            break  # only one accept per path

    return {
        "messy_r2l": messy_r2l,
        "action": action,
        "canonical": canonical,
        "reason": reason,
        "condition": condition,
    }


def render_alignment_text(table: Dict[str, List[str]]) -> str:
    rows = [
        ("Messy (R→L):", table["messy_r2l"]),
        ("Action:", table["action"]),
        ("Canonical:", table["canonical"]),
        ("Reason:", table["reason"]),
    ]
    if "condition" in table:
        rows.append(("Condition:", table["condition"]))

    cols = len(table["messy_r2l"])
    widths = [0] * cols
    for _, values in rows:
        for i, v in enumerate(values):
            widths[i] = max(widths[i], len(str(v)))

    lines: List[str] = []
    for label, values in rows:
        parts = [label.ljust(13)]
        for i, v in enumerate(values):
            parts.append(str(v).rjust(widths[i] + 2))
        lines.append("".join(parts))
    return "\n".join(lines)
