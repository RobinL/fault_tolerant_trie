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

    def col_from_m_index(m_index: int) -> int:
        # Column 0 is the rightmost messy token (L2R index n-1)
        return (n - 1) - m_index

    # First, mark peeled tail (rightmost k columns in R→L)
    for ev in events:
        if ev.get("action") == "PEEL_TAIL":
            k = int(ev.get("k", 0))
            for j in range(k):  # j=0 => rightmost
                idx = j  # R→L index
                action[idx] = ICONS["peel"]
                reason[idx] = "peel"

    # Next, mark exact / skip events
    for ev in events:
        a = ev.get("action")
        if a == "EXACT_DESCEND":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["exact"]
            canonical[j] = str(ev.get("canon", DASH))
            reason[j] = "exact"
        elif a == "SKIP_REDUNDANT":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "redundant"
        elif a == "SKIP_PENALIZED":
            j = col_from_m_index(int(ev["m_index"]))
            action[j] = ICONS["dot"]
            reason[j] = "skip"

    # Finally, mark acceptance star at the consumed token column
    for ev in events:
        if ev.get("action") in ("ACCEPT_UNIQUE", "ACCEPT_TERMINAL"):
            j = col_from_m_index(int(ev["at_m_index"]))
            action[j] = ICONS["star"]
            reason[j] = "unique leaf" if ev["action"] == "ACCEPT_UNIQUE" else "terminal"

            # Mark post-accept leftover tokens to the left in R→L
            for k in range(j + 1, n):
                if reason[k] == "":
                    reason[k] = "post-accept"
            break  # only one accept per path

    return {
        "messy_r2l": messy_r2l,
        "action": action,
        "canonical": canonical,
        "reason": reason,
    }


def render_alignment_text(table: Dict[str, List[str]]) -> str:
    rows = [
        ("Messy (R→L):", table["messy_r2l"]),
        ("Action:", table["action"]),
        ("Canonical:", table["canonical"]),
        ("Reason:", table["reason"]),
    ]

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

