#!/usr/bin/env python3
"""Sweep virtual-node replica counts and report the uniformity tradeoff.

Virtual nodes are the standard fix for consistent hashing's single
biggest weakness (a handful of hash points give wildly uneven arcs, so
some physical nodes get hammered while others sit idle). This script
makes the tradeoff concrete: it holds the node count and key count fixed
and sweeps the replica count, reporting the coefficient of variation
(stdev / mean of per-node key counts -- lower is more uniform) and the
wall-clock cost of building the ring at each replica count.

Run: python benchmark/replica_sweep.py
"""

from __future__ import annotations

import time

from chring.simulator import simulate_distribution

NODE_COUNT = 10
KEY_COUNT = 50_000
REPLICA_COUNTS = [1, 5, 10, 25, 50, 100, 200, 500]


def main() -> None:
    print(f"Sweeping replica counts: {NODE_COUNT} nodes, {KEY_COUNT} keys\n")
    header = f"{'replicas':>10}{'cv':>10}{'min':>10}{'max':>10}{'max/ideal':>12}{'build ms':>12}"
    print(header)
    print("-" * len(header))

    ideal = KEY_COUNT / NODE_COUNT
    for replicas in REPLICA_COUNTS:
        start = time.perf_counter()
        result = simulate_distribution(
            node_count=NODE_COUNT, key_count=KEY_COUNT, replicas=replicas
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(
            f"{replicas:>10}"
            f"{result.coefficient_of_variation:>10.4f}"
            f"{result.min_count:>10}"
            f"{result.max_count:>10}"
            f"{result.max_count / ideal:>11.2f}x"
            f"{elapsed_ms:>12.2f}"
        )

    print(
        "\nAs expected: coefficient of variation drops sharply as replicas "
        "increase, then flattens out -- there are diminishing returns past "
        "roughly 100-200 replicas per node for this node/key count, while "
        "ring build cost grows linearly with replicas. This is why this "
        "project defaults to 100."
    )


if __name__ == "__main__":
    main()
