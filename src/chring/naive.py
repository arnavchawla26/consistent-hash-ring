"""Naive mod-N hashing, for comparison against the consistent hash ring.

This is the "obvious" approach: ``node = hash(key) % N``. It distributes
keys perfectly uniformly for a *fixed* N, which is exactly why it makes a
good baseline -- the problem it has is not distribution, it's what
happens when N changes. Because every key's assignment depends on the
current node *count*, changing N remaps almost every key, even ones whose
owning node didn't need to change at all. That's the effect this
project's simulator is built to demonstrate against consistent hashing.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from .hashing import hash_to_int


class EmptyPoolError(RuntimeError):
    """Raised when a lookup is attempted with no nodes configured."""


class NaiveModNHasher:
    """Assigns keys to nodes via ``hash(key) % len(nodes)``.

    Nodes are kept in a stable, sorted order so that ``get_node`` is a
    pure function of (key, current node set) -- matching how a naive
    mod-N deployment actually behaves (every client recomputes ``% N``
    against whatever node list it currently has).
    """

    def __init__(self, nodes: Optional[Iterable[str]] = None):
        self._nodes: List[str] = sorted(set(nodes)) if nodes else []

    def __len__(self) -> int:
        return len(self._nodes)

    def __contains__(self, node: str) -> bool:
        return node in self._nodes

    @property
    def nodes(self) -> List[str]:
        return list(self._nodes)

    def add_node(self, node: str) -> None:
        if node in self._nodes:
            raise ValueError(f"node {node!r} is already in the pool")
        self._nodes.append(node)
        self._nodes.sort()

    def remove_node(self, node: str) -> None:
        if node not in self._nodes:
            raise KeyError(f"node {node!r} is not in the pool")
        self._nodes.remove(node)

    def get_node(self, key: str) -> str:
        if not self._nodes:
            raise EmptyPoolError("cannot look up a key with no nodes configured")
        idx = hash_to_int(key) % len(self._nodes)
        return self._nodes[idx]

    def distribution(self, keys: Iterable[str]) -> Dict[str, int]:
        counts: Dict[str, int] = {node: 0 for node in self._nodes}
        for key in keys:
            owner = self.get_node(key)
            counts[owner] = counts.get(owner, 0) + 1
        return counts
