"""Simulations comparing consistent hashing against naive mod-N hashing.

Two things are measured, both against the same synthetic key set so the
comparison is apples-to-apples:

* ``simulate_topology_change`` -- given a starting cluster of N nodes and
  a single add-node or remove-node event, what fraction of keys change
  which node owns them? This is *the* headline property of consistent
  hashing: only ~1/N_after of keys should move, versus almost all of them
  for naive mod-N.
* ``simulate_distribution`` -- for a fixed cluster, how evenly are keys
  spread across nodes? Reports each node's share and the coefficient of
  variation (stdev / mean) so callers can see how virtual-node count
  trades off against uniformity.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Literal

from .hash_ring import ConsistentHashRing
from .naive import NaiveModNHasher

EventType = Literal["add", "remove"]


def make_keys(count: int, prefix: str = "key") -> List[str]:
    """Deterministically generate ``count`` synthetic keys."""
    return [f"{prefix}-{i}" for i in range(count)]


def make_nodes(count: int, prefix: str = "node") -> List[str]:
    """Deterministically generate ``count`` synthetic node names."""
    return [f"{prefix}-{i}" for i in range(count)]


@dataclass
class TopologyChangeResult:
    algorithm: str
    event: EventType
    changed_node: str
    nodes_before: int
    nodes_after: int
    total_keys: int
    moved_keys: int

    @property
    def moved_fraction(self) -> float:
        if self.total_keys == 0:
            return 0.0
        return self.moved_keys / self.total_keys

    @property
    def theoretical_minimum_fraction(self) -> float:
        """Lower bound on the fraction of keys that *must* move.

        For an add, at minimum the keys that end up owned by the new node
        have to move (1 / nodes_after of the key space, in expectation).
        For a remove, at minimum the keys that were owned by the removed
        node have to move (1 / nodes_before of the key space).
        """
        if self.event == "add":
            return 1.0 / self.nodes_after if self.nodes_after else 0.0
        return 1.0 / self.nodes_before if self.nodes_before else 0.0


def simulate_topology_change(
    *,
    initial_node_count: int,
    key_count: int,
    event: EventType,
    replicas: int = 100,
    changed_node_name: str = "new-node",
) -> Dict[str, TopologyChangeResult]:
    """Compare key movement for one topology change, both algorithms.

    Returns a dict with keys ``"consistent"`` and ``"naive"``.
    """
    if initial_node_count < 1:
        raise ValueError("initial_node_count must be >= 1")
    if event == "remove" and initial_node_count < 2:
        raise ValueError("need at least 2 nodes to simulate a removal")

    nodes = make_nodes(initial_node_count)
    keys = make_keys(key_count)

    # --- consistent hashing ---
    ring = ConsistentHashRing(replicas=replicas, nodes=nodes)
    before_ring = {key: ring.get_node(key) for key in keys}
    if event == "add":
        ring.add_node(changed_node_name)
        changed = changed_node_name
        nodes_after = initial_node_count + 1
    else:
        changed = nodes[0]
        ring.remove_node(changed)
        nodes_after = initial_node_count - 1
    after_ring = {key: ring.get_node(key) for key in keys}
    moved_ring = sum(1 for key in keys if before_ring[key] != after_ring[key])

    # --- naive mod-N ---
    naive = NaiveModNHasher(nodes)
    before_naive = {key: naive.get_node(key) for key in keys}
    if event == "add":
        naive.add_node(changed_node_name)
    else:
        naive.remove_node(changed)
    after_naive = {key: naive.get_node(key) for key in keys}
    moved_naive = sum(1 for key in keys if before_naive[key] != after_naive[key])

    return {
        "consistent": TopologyChangeResult(
            algorithm="consistent",
            event=event,
            changed_node=changed,
            nodes_before=initial_node_count,
            nodes_after=nodes_after,
            total_keys=key_count,
            moved_keys=moved_ring,
        ),
        "naive": TopologyChangeResult(
            algorithm="naive-mod-n",
            event=event,
            changed_node=changed,
            nodes_before=initial_node_count,
            nodes_after=nodes_after,
            total_keys=key_count,
            moved_keys=moved_naive,
        ),
    }


@dataclass
class DistributionResult:
    replicas: int
    node_count: int
    total_keys: int
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def mean(self) -> float:
        return statistics.mean(self.counts.values()) if self.counts else 0.0

    @property
    def stdev(self) -> float:
        return statistics.pstdev(self.counts.values()) if len(self.counts) > 1 else 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """stdev / mean -- lower means more uniform. 0 is perfectly even."""
        mean = self.mean
        return self.stdev / mean if mean else 0.0

    @property
    def min_count(self) -> int:
        return min(self.counts.values()) if self.counts else 0

    @property
    def max_count(self) -> int:
        return max(self.counts.values()) if self.counts else 0


def simulate_distribution(
    *, node_count: int, key_count: int, replicas: int = 100
) -> DistributionResult:
    """Measure how evenly a ring with ``replicas`` virtual nodes per
    physical node spreads ``key_count`` keys across ``node_count`` nodes.
    """
    nodes = make_nodes(node_count)
    keys = make_keys(key_count)
    ring = ConsistentHashRing(replicas=replicas, nodes=nodes)
    counts = ring.distribution(keys)
    return DistributionResult(
        replicas=replicas,
        node_count=node_count,
        total_keys=key_count,
        counts=counts,
    )
