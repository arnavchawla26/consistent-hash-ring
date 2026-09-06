import pytest

from chring.hash_ring import ConsistentHashRing, EmptyRingError


def test_empty_ring_raises_on_lookup():
    ring = ConsistentHashRing()
    with pytest.raises(EmptyRingError):
        ring.get_node("anything")


def test_add_and_contains():
    ring = ConsistentHashRing(replicas=10)
    assert "a" not in ring
    ring.add_node("a")
    assert "a" in ring
    assert ring.nodes == ["a"]
    assert len(ring) == 1


def test_add_duplicate_raises():
    ring = ConsistentHashRing(replicas=10, nodes=["a"])
    with pytest.raises(ValueError):
        ring.add_node("a")


def test_remove_missing_raises():
    ring = ConsistentHashRing(replicas=10, nodes=["a"])
    with pytest.raises(KeyError):
        ring.remove_node("b")


def test_remove_restores_single_node_ownership():
    ring = ConsistentHashRing(replicas=20, nodes=["a", "b"])
    ring.remove_node("b")
    assert len(ring) == 1
    # with only "a" left, every key must map to "a"
    for i in range(200):
        assert ring.get_node(f"k{i}") == "a"


def test_single_node_ring_owns_everything():
    ring = ConsistentHashRing(replicas=50, nodes=["only"])
    for i in range(500):
        assert ring.get_node(f"key-{i}") == "only"


def test_lookup_is_deterministic_and_stable_across_instances():
    ring1 = ConsistentHashRing(replicas=30, nodes=["a", "b", "c"])
    ring2 = ConsistentHashRing(replicas=30, nodes=["a", "b", "c"])
    for i in range(200):
        key = f"key-{i}"
        assert ring1.get_node(key) == ring2.get_node(key)


def test_ring_points_sorted_and_correct_count():
    ring = ConsistentHashRing(replicas=15, nodes=["a", "b", "c"])
    points = ring.ring_points()
    assert len(points) == 45
    hashes = [h for h, _ in points]
    assert hashes == sorted(hashes)
    assert len(set(hashes)) == 45  # no collisions at this scale


def test_get_nodes_returns_distinct_owners_in_ring_order():
    ring = ConsistentHashRing(replicas=25, nodes=["a", "b", "c", "d"])
    owners = ring.get_nodes("some-key", count=3)
    assert len(owners) == 3
    assert len(set(owners)) == 3
    for o in owners:
        assert o in ring.nodes


def test_get_nodes_count_greater_than_ring_size_caps_at_node_count():
    ring = ConsistentHashRing(replicas=10, nodes=["a", "b"])
    owners = ring.get_nodes("k", count=5)
    assert set(owners) == {"a", "b"}


def test_get_nodes_rejects_nonpositive_count():
    ring = ConsistentHashRing(replicas=10, nodes=["a"])
    with pytest.raises(ValueError):
        ring.get_nodes("k", count=0)


def test_replicas_must_be_positive():
    with pytest.raises(ValueError):
        ConsistentHashRing(replicas=0)


def test_per_node_replica_override():
    ring = ConsistentHashRing(replicas=10)
    ring.add_node("a", replicas=5)
    ring.add_node("b", replicas=50)
    assert len(ring._node_points["a"]) == 5
    assert len(ring._node_points["b"]) == 50


def test_distribution_counts_all_keys_exactly_once():
    ring = ConsistentHashRing(replicas=40, nodes=["a", "b", "c"])
    keys = [f"key-{i}" for i in range(1000)]
    dist = ring.distribution(keys)
    assert sum(dist.values()) == 1000
    assert set(dist.keys()) == {"a", "b", "c"}


def test_more_replicas_improves_uniformity():
    # Coefficient of variation should shrink (roughly) as replica count
    # grows -- this is the whole point of virtual nodes. Use a large key
    # sample so the comparison isn't noise-dominated.
    keys = [f"key-{i}" for i in range(20_000)]

    def cv(replicas):
        ring = ConsistentHashRing(replicas=replicas, nodes=[f"n{i}" for i in range(8)])
        counts = list(ring.distribution(keys).values())
        mean = sum(counts) / len(counts)
        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        return (variance ** 0.5) / mean

    cv_few = cv(1)
    cv_many = cv(200)
    assert cv_many < cv_few


def test_adding_node_moves_roughly_expected_fraction_of_keys():
    # Adding the (N+1)th node to an N-node ring should move roughly
    # 1/(N+1) of keys, not (N-1)/N of them like naive mod-N would.
    keys = [f"key-{i}" for i in range(20_000)]
    ring = ConsistentHashRing(replicas=100, nodes=[f"n{i}" for i in range(9)])
    before = {k: ring.get_node(k) for k in keys}
    ring.add_node("new")
    after = {k: ring.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    fraction = moved / len(keys)
    expected = 1 / 10
    # generous tolerance -- this is a statistical property, not exact
    assert expected * 0.5 < fraction < expected * 1.5


def test_removing_node_only_moves_that_nodes_keys():
    keys = [f"key-{i}" for i in range(20_000)]
    ring = ConsistentHashRing(replicas=100, nodes=[f"n{i}" for i in range(10)])
    before = {k: ring.get_node(k) for k in keys}
    ring.remove_node("n0")
    after = {k: ring.get_node(k) for k in keys}
    for k in keys:
        if before[k] != "n0":
            # keys not owned by the removed node must be unaffected
            assert after[k] == before[k]
