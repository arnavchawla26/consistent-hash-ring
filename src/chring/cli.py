"""``chring`` command-line interface.

Subcommands:

* ``chring movement``     -- compare key movement on add/remove between
                              consistent hashing and naive mod-N hashing.
* ``chring distribution`` -- show how evenly keys spread across nodes for
                              a given virtual-node replica count.
* ``chring lookup``       -- look up which node(s) own a given key on an
                              ad-hoc ring (mostly a smoke-test / demo aid).
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from .hash_ring import ConsistentHashRing
from .simulator import simulate_distribution, simulate_topology_change


def _cmd_movement(args: argparse.Namespace) -> int:
    results = simulate_topology_change(
        initial_node_count=args.nodes,
        key_count=args.keys,
        event=args.event,
        replicas=args.replicas,
    )
    consistent = results["consistent"]
    naive = results["naive"]

    print(f"Topology change: {args.event} a node "
          f"({consistent.nodes_before} -> {consistent.nodes_after} nodes), "
          f"{args.keys} keys, {args.replicas} virtual nodes/physical node\n")

    header = f"{'algorithm':<14}{'moved':>10}{'moved %':>12}{'theoretical min %':>20}"
    print(header)
    print("-" * len(header))
    for result in (consistent, naive):
        print(
            f"{result.algorithm:<14}"
            f"{result.moved_keys:>10}"
            f"{result.moved_fraction * 100:>11.2f}%"
            f"{result.theoretical_minimum_fraction * 100:>19.2f}%"
        )

    if naive.moved_keys > 0:
        ratio = consistent.moved_fraction / naive.moved_fraction if naive.moved_fraction else 0
        print(f"\nConsistent hashing moved {ratio * 100:.1f}% as many keys as naive mod-N.")
    return 0


def _cmd_distribution(args: argparse.Namespace) -> int:
    result = simulate_distribution(
        node_count=args.nodes, key_count=args.keys, replicas=args.replicas
    )
    print(
        f"Distribution across {result.node_count} nodes, {result.total_keys} keys, "
        f"{result.replicas} virtual nodes/physical node\n"
    )
    ideal = result.total_keys / result.node_count if result.node_count else 0
    header = f"{'node':<12}{'keys':>10}{'% of ideal':>14}"
    print(header)
    print("-" * len(header))
    for node in sorted(result.counts):
        count = result.counts[node]
        pct = (count / ideal * 100) if ideal else 0.0
        print(f"{node:<12}{count:>10}{pct:>13.1f}%")
    print(
        f"\nmin={result.min_count} max={result.max_count} "
        f"mean={result.mean:.1f} stdev={result.stdev:.2f} "
        f"coefficient_of_variation={result.coefficient_of_variation:.4f}"
    )
    return 0


def _cmd_lookup(args: argparse.Namespace) -> int:
    ring = ConsistentHashRing(replicas=args.replicas, nodes=args.node)
    owners = ring.get_nodes(args.key, count=min(args.replicate, len(ring)))
    print(f"key={args.key!r} -> {owners}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="chring",
        description="Consistent hashing with virtual nodes: simulator and CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_move = sub.add_parser(
        "movement", help="compare key movement vs naive mod-N on a topology change"
    )
    p_move.add_argument("--nodes", type=int, default=10, help="starting node count (default: 10)")
    p_move.add_argument("--keys", type=int, default=10_000, help="number of keys (default: 10000)")
    p_move.add_argument(
        "--event", choices=["add", "remove"], default="add", help="topology change to simulate"
    )
    p_move.add_argument(
        "--replicas", type=int, default=100, help="virtual nodes per physical node (default: 100)"
    )
    p_move.set_defaults(func=_cmd_movement)

    p_dist = sub.add_parser(
        "distribution", help="measure how evenly keys spread across nodes"
    )
    p_dist.add_argument("--nodes", type=int, default=10, help="node count (default: 10)")
    p_dist.add_argument("--keys", type=int, default=10_000, help="number of keys (default: 10000)")
    p_dist.add_argument(
        "--replicas", type=int, default=100, help="virtual nodes per physical node (default: 100)"
    )
    p_dist.set_defaults(func=_cmd_distribution)

    p_lookup = sub.add_parser("lookup", help="look up which node(s) own a key")
    p_lookup.add_argument("key", help="key to look up")
    p_lookup.add_argument(
        "--node", action="append", required=True, help="a node name; repeat for multiple nodes"
    )
    p_lookup.add_argument(
        "--replicas", type=int, default=100, help="virtual nodes per physical node (default: 100)"
    )
    p_lookup.add_argument(
        "--replicate", type=int, default=1, help="number of distinct owning nodes to return"
    )
    p_lookup.set_defaults(func=_cmd_lookup)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
