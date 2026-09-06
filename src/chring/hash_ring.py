"""Consistent hash ring with virtual nodes.

Classic consistent hashing (Karger et al., 1997): physical nodes and keys
are both hashed onto the same circular integer space (a "ring"). A key is
owned by the first node whose hash is >= the key's hash, walking clockwise
and wrapping around at the top of the ring. When a node is added or
removed, only the keys that fall in the arc it now owns (or used to own)
move -- everything else stays put.

A single point per physical node makes the arcs (and therefore the key
distribution across nodes) very uneven. The standard fix, used here, is
"virtual nodes": each physical node is hashed onto the ring many times
(``replicas`` times) under distinguishable labels (``f"{node}#{i}"``),
which averages out the arc-length variance and gives a much more uniform
key distribution without changing the core algorithm.
"""

from __future__ import annotations

import bisect
from typing import Dict, Iterable, List, Optional, Tuple

from .hashing import hash_to_int


class EmptyRingError(RuntimeError):
    """Raised when a lookup is attempted on a ring with no nodes."""


class ConsistentHashRing:
    """A consistent hash ring supporting virtual nodes.

    Parameters
    ----------
    replicas:
        Number of virtual nodes placed on the ring per physical node.
        Higher values give a more uniform key distribution at the cost of
        more memory and slower inserts/removals. 100-200 is a common
        real-world default; this project defaults to 100.
    nodes:
        Optional iterable of physical node names to add immediately.
    """

    def __init__(self, replicas: int = 100, nodes: Optional[Iterable[str]] = None):
        if replicas < 1:
            raise ValueError("replicas must be >= 1")
        self.replicas = replicas
        # Sorted parallel arrays: _ring_hashes[i] is the ring position of
        # the virtual node whose physical owner is _ring_owners[i].
        self._ring_hashes: List[int] = []
        self._ring_owners: List[str] = []
        # physical node -> list of its virtual-node ring positions, so
        # remove_node() doesn't have to do a linear owner scan.
        self._node_points: Dict[str, List[int]] = {}

        if nodes:
            for node in nodes:
                self.add_node(node)

    def __len__(self) -> int:
        return len(self._node_points)

    def __contains__(self, node: str) -> bool:
        return node in self._node_points

    @property
    def nodes(self) -> List[str]:
        """Physical nodes currently on the ring, in insertion order."""
        return list(self._node_points.keys())

    def _vnode_label(self, node: str, replica_index: int) -> str:
        return f"{node}#{replica_index}"

    def add_node(self, node: str, replicas: Optional[int] = None) -> None:
        """Add a physical node to the ring, placing its virtual nodes.

        Raises ``ValueError`` if ``node`` is already on the ring.
        """
        if node in self._node_points:
            raise ValueError(f"node {node!r} is already on the ring")
        n_replicas = self.replicas if replicas is None else replicas
        if n_replicas < 1:
            raise ValueError("replicas must be >= 1")

        points: List[int] = []
        for i in range(n_replicas):
            point = hash_to_int(self._vnode_label(node, i))
            idx = bisect.bisect_left(self._ring_hashes, point)
            # Extremely unlikely at 64 bits, but guard against a genuine
            # hash collision landing exactly on an existing point.
            if idx < len(self._ring_hashes) and self._ring_hashes[idx] == point:
                raise RuntimeError(
                    f"hash collision placing virtual node {i} for {node!r}; "
                    "retry with a different node label"
                )
            self._ring_hashes.insert(idx, point)
            self._ring_owners.insert(idx, node)
            points.append(point)

        self._node_points[node] = points

    def remove_node(self, node: str) -> None:
        """Remove a physical node (and all its virtual nodes) from the ring.

        Raises ``KeyError`` if ``node`` is not on the ring.
        """
        if node not in self._node_points:
            raise KeyError(f"node {node!r} is not on the ring")
        for point in self._node_points[node]:
            idx = bisect.bisect_left(self._ring_hashes, point)
            # There can be at most one entry at this exact point (see the
            # collision guard in add_node), so idx is unambiguous.
            del self._ring_hashes[idx]
            del self._ring_owners[idx]
        del self._node_points[node]

    def get_node(self, key: str) -> str:
        """Return the physical node responsible for ``key``."""
        if not self._ring_hashes:
            raise EmptyRingError("cannot look up a key on an empty ring")
        point = hash_to_int(key)
        idx = bisect.bisect_left(self._ring_hashes, point)
        if idx == len(self._ring_hashes):
            idx = 0  # wrap around the top of the ring
        return self._ring_owners[idx]

    def get_nodes(self, key: str, count: int) -> List[str]:
        """Return up to ``count`` distinct physical nodes for ``key``.

        Walks clockwise from the key's ring position, the way a real
        system would pick N-way replica placement for a key: the primary
        owner plus the next distinct physical nodes encountered.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        if not self._ring_hashes:
            raise EmptyRingError("cannot look up a key on an empty ring")
        point = hash_to_int(key)
        start = bisect.bisect_left(self._ring_hashes, point)
        n = len(self._ring_hashes)
        seen: List[str] = []
        seen_set = set()
        for step in range(n):
            owner = self._ring_owners[(start + step) % n]
            if owner not in seen_set:
                seen_set.add(owner)
                seen.append(owner)
                if len(seen) == count:
                    break
        return seen

    def distribution(self, keys: Iterable[str]) -> Dict[str, int]:
        """Count how many of ``keys`` land on each physical node."""
        counts: Dict[str, int] = {node: 0 for node in self._node_points}
        for key in keys:
            owner = self.get_node(key)
            counts[owner] = counts.get(owner, 0) + 1
        return counts

    def ring_points(self) -> List[Tuple[int, str]]:
        """Return a copy of the raw (hash, owner) ring, sorted by hash."""
        return list(zip(self._ring_hashes, self._ring_owners))
