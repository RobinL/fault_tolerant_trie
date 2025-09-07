# basic_trie.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Union, Optional

TokenSeq = Sequence[str]
CanonRow = Union[Tuple[int, TokenSeq, str], TokenSeq]


@dataclass
class TrieNode:
    """Simple token trie node (right-to-left by default)."""

    count: int = 0  # how many paths pass through this node
    terminal: int = 0  # how many paths end at this node
    children: Dict[str, "TrieNode"] = field(default_factory=dict)
    uprn: Optional[int] = None  # UPRN for terminal nodes; None otherwise

    # --- core ops ---
    def insert(self, tokens: TokenSeq, uprn: Optional[int] = None) -> None:
        node = self
        # Bump root count so future skip logic (which may use node.count at root)
        # has meaningful totals.
        node.count += 1
        for tok in tokens:
            node = node.children.setdefault(tok, TrieNode())
            node.count += 1
        node.terminal += 1
        # Mark leaf with UPRN if provided
        if uprn is not None:
            node.uprn = int(uprn)

    def has_path(self, tokens: TokenSeq) -> bool:
        node = self
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return False
        return node.terminal > 0

    def count_for_path(self, tokens: TokenSeq) -> int:
        node = self
        for tok in tokens:
            node = node.children.get(tok)
            if node is None:
                return 0
        return node.count

    # --- convenience helpers for matcher readability ---
    def child(self, tok: str) -> Optional["TrieNode"]:
        """Return the direct child for token, or None if absent."""
        return self.children.get(tok)

    def has_child(self, tok: str) -> bool:
        """True if a direct child exists for token."""
        return tok in self.children

    def child_count(self, tok: str) -> int:
        """Count of the child node for token; 0 if it does not exist."""
        child = self.children.get(tok)
        return 0 if child is None else int(child.count)

    def iter_children(self):
        """Iterator over (token, child_node) pairs."""
        return self.children.items()


# --- construction helpers ---


def build_trie(paths: Iterable[TokenSeq], *, reverse: bool = True) -> TrieNode:
    """Build a trie from sequences of tokens. If reverse=True, builds a suffix trie."""
    root = TrieNode()
    for p in paths:
        toks = list(reversed(p)) if reverse else list(p)
        root.insert(toks, uprn=None)
    return root


def build_trie_from_canonical(
    canonical_addresses: Iterable[CanonRow], *, reverse: bool = True
) -> TrieNode:
    """
    Accepts rows like:
      (id: int, tokens: List[str], postcode_str: str)  OR just tokens: List[str]
    and builds a trie from the token lists.
    """
    root = TrieNode()
    for row in canonical_addresses:
        # Expect (uprn, tokens, postcode) or just tokens
        if (
            isinstance(row, (list, tuple))
            and len(row) == 3
            and isinstance(row[1], Sequence)
        ):
            uprn = int(row[0])
            tokens = [str(t) for t in row[1]]
            postcode_str = str(row[2]) if row[2] is not None else ""

            # Ensure postcode tokens are NOT included in the trie
            pc_parts = [p for p in postcode_str.strip().split() if p]
            if pc_parts:
                pc_set = {p.upper() for p in pc_parts}
                tokens = [t for t in tokens if t.upper() not in pc_set]

            toks = list(reversed(tokens)) if reverse else list(tokens)
            root.insert(toks, uprn=uprn)
        else:
            tokens = [str(t) for t in (row if isinstance(row, Sequence) else [row])]
            toks = list(reversed(tokens)) if reverse else list(tokens)
            root.insert(toks, uprn=None)
    return root


# --- pretty-print (ASCII) ---


def ascii_lines(node: TrieNode, prefix: str = "", sort: bool = True) -> List[str]:
    """Return lines for an ASCII tree. Terminal nodes show UPRN; no '*'."""
    lines: List[str] = []
    items = node.children.items()
    if sort:
        items = sorted(items, key=lambda kv: kv[0])
    total = len(node.children)
    for i, (tok, child) in enumerate(items):
        last = i == total - 1
        conn = "`-- " if last else "|-- "
        uprn_str = f", uprn={child.uprn}" if child.uprn is not None else ""
        lines.append(f"{prefix}{conn}{tok} (count={child.count}{uprn_str})")
        lines.extend(ascii_lines(child, prefix + ("    " if last else "|   "), sort))
    return lines


# Intentionally no printing function to keep library silent


# --- wrappers for suffix count semantics (L2R to R2L) ---

def count_tail_L2R(root: TrieNode, suffix_tokens_L2R: Sequence[str]) -> int:
    """
    Given a left-to-right suffix (e.g., ["KINGS","LANGLEY"]) for an address,
    convert to the right-to-left trie path and return the node.count reachable
    by that path. Returns 0 if the path does not exist.
    """
    path = list(reversed([t for t in suffix_tokens_L2R if t is not None]))
    return root.count_for_path(path)
