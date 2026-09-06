"""Deterministic hash helpers shared by the ring and the naive hasher.

Uses MD5 (not for security -- purely as a fast, well-distributed,
deterministic 128-bit hash) truncated to a 64-bit unsigned integer, which
gives a big enough space that collisions between distinct ring points are
effectively impossible for the scales this project simulates.
"""

from __future__ import annotations

import hashlib

MASK_64 = (1 << 64) - 1


def hash_to_int(value: str) -> int:
    """Hash an arbitrary string to a deterministic unsigned 64-bit integer."""
    digest = hashlib.md5(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) & MASK_64
