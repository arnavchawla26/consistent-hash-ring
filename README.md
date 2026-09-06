# consistent-hash-ring

A from-scratch implementation of consistent hashing with virtual nodes,
plus a simulator that measures how much less data moves compared to
naive `hash(key) % N` sharding when a cluster's node count changes.

## Why this exists

Say you're sharding keys across N cache/database nodes with the obvious
approach, `node = hash(key) % N`. It distributes keys perfectly evenly
for a fixed N -- but the moment N changes (a node is added or removed),
almost every key's `% N` result changes too, even keys whose "natural"
owner didn't need to move. In a real cluster that means a wave of cache
misses or a full data reshuffle every time you scale.

Consistent hashing (Karger et al., 1997) fixes this by hashing both nodes
and keys onto the same circular space and giving each key to the next
node clockwise. Adding or removing a node only touches the keys in the
arc that node now owns (or used to own) -- everything else stays put.
The catch with a single hash point per node is uneven arc lengths (some
nodes get much more of the ring than others), which is why real systems
use **virtual nodes**: each physical node is hashed onto the ring many
times under different labels, averaging out the unevenness.

This project implements both algorithms from scratch and a simulator
that puts a number on the difference.

## What's in here

- `chring.hash_ring.ConsistentHashRing` -- the ring itself: add/remove
  nodes, look up the owner (or the first N distinct owners) of a key,
  inspect the key distribution.
- `chring.naive.NaiveModNHasher` -- the `hash(key) % N` baseline, for
  comparison.
- `chring.simulator` -- given a starting cluster and a single
  add-node/remove-node event, computes exactly how many keys change
  owner under each algorithm (`simulate_topology_change`), and how evenly
  a ring spreads keys for a given virtual-node replica count
  (`simulate_distribution`).
- `chring` CLI -- run the simulator and look up keys from the command
  line without writing any code.

## Example: key movement on scale-out

```
$ chring movement --nodes 10 --keys 10000 --event add
Topology change: add a node (10 -> 11 nodes), 10000 keys, 100 virtual nodes/physical node

algorithm          moved     moved %   theoretical min %
--------------------------------------------------------
consistent            822        8.22%              9.09%
naive-mod-n          9050       90.50%               9.09%

Consistent hashing moved 9.1% as many keys as naive mod-N.
```

Adding an 11th node to a 10-node cluster should move at minimum ~1/11 of
keys (the ones that now belong to the new node). Consistent hashing gets
within a hair of that theoretical floor; naive mod-N reshuffles 90% of
everything.

## Example: virtual-node replica count vs. distribution uniformity

```
$ chring distribution --nodes 8 --keys 20000 --replicas 100

Distribution across 8 nodes, 20000 keys, 100 virtual nodes/physical node

node              keys    % of ideal
------------------------------------
node-0            2567        102.7%
node-1            2438         97.5%
...

min=2174 max=2780 mean=2500.0 stdev=171.89 coefficient_of_variation=0.0688
```

`benchmark/replica_sweep.py` sweeps the replica count from 1 to 500 and
shows the coefficient of variation (stdev/mean of per-node key counts)
dropping sharply and then flattening out past ~100-200 replicas per node
-- the basis for this project's default of 100.

## Tech stack

Pure Python 3.9+, standard library only (`hashlib`, `bisect`,
`statistics`, `argparse`). No runtime dependencies. `pytest` for tests,
`pyflakes` for linting (dev-only).

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# CLI
chring movement --nodes 10 --keys 10000 --event add
chring movement --nodes 10 --keys 10000 --event remove
chring distribution --nodes 8 --keys 20000 --replicas 100
chring lookup my-key --node a --node b --node c --replicate 2

# tests
pytest

# lint
pyflakes src tests

# replica-count-vs-uniformity sweep
python benchmark/replica_sweep.py
```

## Library usage

```python
from chring import ConsistentHashRing

ring = ConsistentHashRing(replicas=100, nodes=["cache-1", "cache-2", "cache-3"])
ring.get_node("user:12345")          # -> which node owns this key
ring.get_nodes("user:12345", count=2)  # -> primary + one replica, for N-way placement

ring.add_node("cache-4")             # only ~1/4 of keys move
ring.remove_node("cache-2")          # only cache-2's former keys move
```

## Design notes / what I'd add next

- **Hashing**: MD5 truncated to 64 bits, purely for speed and even
  distribution -- not used for anything security-sensitive here.
- **Ring storage**: a sorted `(hash, owner)` array with `bisect` for
  O(log n) lookups and O(n) insert/remove (via `list.insert`/`del`,
  since Python lists don't have log-n insert). Fine for the node/replica
  counts this project simulates (thousands of virtual nodes); a
  production system with very frequent topology changes and huge
  replica counts would want a balanced tree or skip list instead.
- **`get_nodes(key, count)`** exists because real systems don't just
  want a single owner per key -- they replicate each key onto the next
  few distinct physical nodes walking clockwise from the key's ring
  position, so a node failure doesn't lose data. This project doesn't
  implement actual replication/failover, just the placement logic.
- Not implemented (out of scope for a from-scratch demo): weighted nodes
  (giving a bigger node proportionally more virtual points), an actual
  network-facing service, or persistence -- this is a library +
  simulator, not a running cluster.

## Current status

v1, complete and tested. Ring (add/remove/lookup/get_nodes/distribution),
naive mod-N baseline, topology-change and distribution simulators, and a
CLI are all implemented and covered by 47 tests (unit tests per module
plus statistical-property tests on the ring's core claims: bounded key
movement on topology change, distribution uniformity improving with
replica count). `benchmark/replica_sweep.py` provides the empirical
replica-count-vs-uniformity data referenced above.
