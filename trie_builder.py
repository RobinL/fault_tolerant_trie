# basic_trie.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple, Union

TokenSeq = Sequence[str]
CanonRow = Union[Tuple[int, TokenSeq, str], TokenSeq]


@dataclass
class TrieNode:
    """Simple token trie node (right-to-left by default)."""

    count: int = 0  # how many paths pass through this node
    terminal: int = 0  # how many paths end at this node
    children: Dict[str, "TrieNode"] = field(default_factory=dict)

    # --- core ops ---
    def insert(self, tokens: TokenSeq) -> None:
        node = self
        for tok in tokens:
            node = node.children.setdefault(tok, TrieNode())
            node.count += 1
        node.terminal += 1

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


# --- construction helpers ---


def build_trie(paths: Iterable[TokenSeq], *, reverse: bool = True) -> TrieNode:
    """Build a trie from sequences of tokens. If reverse=True, builds a suffix trie."""
    root = TrieNode()
    for p in paths:
        toks = list(reversed(p)) if reverse else list(p)
        root.insert(toks)
    return root


def build_trie_from_canonical(
    canonical_addresses: Iterable[CanonRow], *, reverse: bool = True
) -> TrieNode:
    """
    Accepts rows like:
      (id: int, tokens: List[str], postcode_str: str)  OR just tokens: List[str]
    and builds a trie from the token lists.
    """
    paths: List[List[str]] = []
    for row in canonical_addresses:
        if (
            isinstance(row, (list, tuple))
            and len(row) == 3
            and isinstance(row[1], Sequence)
        ):
            tokens = [str(t) for t in row[1]]
        else:
            tokens = [str(t) for t in row]  # tolerate plain token sequences
        paths.append(tokens)
    return build_trie(paths, reverse=reverse)


# --- pretty-print (ASCII) ---


def ascii_lines(node: TrieNode, prefix: str = "", sort: bool = True) -> List[str]:
    """Return lines for an ASCII tree. '*' marks terminal nodes."""
    lines: List[str] = []
    items = node.children.items()
    if sort:
        items = sorted(items, key=lambda kv: kv[0])
    total = len(node.children)
    for i, (tok, child) in enumerate(items):
        last = i == total - 1
        conn = "`-- " if last else "|-- "
        star = "*" if child.terminal else ""
        lines.append(f"{prefix}{conn}{tok}{star} (count={child.count})")
        lines.extend(ascii_lines(child, prefix + ("    " if last else "|   "), sort))
    return lines


def print_trie(node: TrieNode) -> None:
    print("(root)")
    for line in ascii_lines(node):
        print(line)
