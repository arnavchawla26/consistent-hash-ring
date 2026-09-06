"""consistent-hash-ring: consistent hashing with virtual nodes.

A from-scratch implementation of consistent hashing (as used to place
shards/keys across nodes in a distributed cache or data store), plus a
naive mod-N hasher for comparison, and a simulator that measures how many
keys actually move when the cluster topology changes.
"""

from .hash_ring import ConsistentHashRing
from .naive import NaiveModNHasher

__version__ = "0.1.0"

__all__ = ["ConsistentHashRing", "NaiveModNHasher", "__version__"]
