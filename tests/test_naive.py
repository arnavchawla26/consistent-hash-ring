import pytest

from chring.naive import EmptyPoolError, NaiveModNHasher


def test_empty_pool_raises():
    hasher = NaiveModNHasher()
    with pytest.raises(EmptyPoolError):
        hasher.get_node("k")


def test_add_duplicate_raises():
    hasher = NaiveModNHasher(["a"])
    with pytest.raises(ValueError):
        hasher.add_node("a")


def test_remove_missing_raises():
    hasher = NaiveModNHasher(["a"])
    with pytest.raises(KeyError):
        hasher.remove_node("b")


def test_deterministic_for_fixed_topology():
    hasher1 = NaiveModNHasher(["a", "b", "c"])
    hasher2 = NaiveModNHasher(["a", "b", "c"])
    for i in range(200):
        key = f"key-{i}"
        assert hasher1.get_node(key) == hasher2.get_node(key)


def test_single_node_owns_everything():
    hasher = NaiveModNHasher(["only"])
    for i in range(200):
        assert hasher.get_node(f"k{i}") == "only"


def test_distribution_is_reasonably_uniform_for_fixed_n():
    # Naive mod-N is uniform *for a fixed node count* -- that's not the
    # thing it's bad at. Confirm that baseline property.
    hasher = NaiveModNHasher([f"n{i}" for i in range(8)])
    keys = [f"key-{i}" for i in range(20_000)]
    dist = hasher.distribution(keys)
    counts = list(dist.values())
    mean = sum(counts) / len(counts)
    for c in counts:
        assert abs(c - mean) / mean < 0.1


def test_adding_node_moves_almost_all_keys():
    keys = [f"key-{i}" for i in range(5_000)]
    hasher = NaiveModNHasher([f"n{i}" for i in range(9)])
    before = {k: hasher.get_node(k) for k in keys}
    hasher.add_node("new")
    after = {k: hasher.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    fraction = moved / len(keys)
    # this is the defining weakness of mod-N: almost everything reshuffles
    assert fraction > 0.7


def test_removing_node_moves_almost_all_keys():
    keys = [f"key-{i}" for i in range(5_000)]
    hasher = NaiveModNHasher([f"n{i}" for i in range(10)])
    before = {k: hasher.get_node(k) for k in keys}
    hasher.remove_node("n0")
    after = {k: hasher.get_node(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    fraction = moved / len(keys)
    assert fraction > 0.7
