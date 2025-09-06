from __future__ import annotations

from typing import Callable, List, Sequence, Optional

from .trie_builder import TrieNode, count_tail_L2R


def peel_end_tokens(
    tokens: Sequence[str],
    count_tail: Callable[[Sequence[str]], int],
    steps: int = 4,
    max_k: int = 2,
) -> List[str]:
    """
    Deterministic tail peeling by counts.

    Iteratively drop up to `max_k` final tokens (default 2) from the messy address when doing so
    "joins a larger subtree" in the canonical trie, as measured by the count
    for the last token (anchor) increasing.

    Rule per step:
      - Let base = count_tail([last_token]).
      - For k in {1..max_k}, check new_base = count_tail([token_at(-k-1)]).
      - If any k yields new_base > base, drop the last k tokens (choose the
        k with the largest strictly-positive increase). Otherwise stop.

    This removes tails like "... HERTFORDSHIRE ENGLAND" while keeping
    informative locality tails like "... KINGS LANGLEY" intact.
    """
    if not tokens:
        return []

    out = list(tokens)
    for _ in range(max(0, int(steps))):
        if len(out) <= 1:
            break

        base = count_tail([out[-1]])
        best_k = 0
        best_score = base

        max_try = min(int(max_k), len(out) - 1)
        for k in range(1, max_try + 1):
            new_last = out[-k - 1]
            score = count_tail([new_last])
            if score > best_score:
                best_score = score
                best_k = k

        if best_k > 0:
            out = out[: -best_k]
        else:
            break

    return out


def peel_end_tokens_with_trie(
    tokens: Sequence[str],
    root: TrieNode,
    steps: int = 4,
    max_k: int = 2,
) -> List[str]:
    """Thin wrapper wiring peel_end_tokens to the trie count helper."""

    def _count_tail(tail: Sequence[str]) -> int:
        return count_tail_L2R(root, tail)

    return peel_end_tokens(tokens, _count_tail, steps=steps, max_k=max_k)


def walk_exact(tokens_L2R: Sequence[str], root: TrieNode) -> Optional[int]:
    """
    Consume tokens right-to-left using exact child transitions only.

    Acceptance (precision-first): at any point, if the current node has
    an attached UPRN and is a unique suffix (count==1), and either:
      - all tokens are consumed, or
      - the next token cannot be consumed (no matching child),
    then return that UPRN. Otherwise return None.
    """
    node = root
    t = list(reversed([str(x) for x in tokens_L2R]))

    i = 0
    n = len(t)
    while True:
        # Check acceptance at current node before attempting to consume next token
        can_accept = (
            node.uprn is not None
            and node.count == 1
            and (i >= n or not node.has_child(t[i]))
        )
        if can_accept:
            return node.uprn

        if i >= n:
            return None

        nxt = t[i]
        child = node.child(nxt)
        if child is None:
            return None

        node = child
        i += 1
