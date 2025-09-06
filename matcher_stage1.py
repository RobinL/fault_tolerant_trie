from __future__ import annotations

from typing import Callable, Iterable, List, Sequence

from trie_builder import TrieNode, count_tail_L2R


def peel_end_tokens(
    tokens: Sequence[str],
    count_tail: Callable[[Sequence[str]], int],
    steps: int = 4,
    max_k: int = 2,
) -> List[str]:
    """
    Deterministic tail peeling by counts.

    Iteratively drop 1–2 final tokens from the messy address when doing so
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

