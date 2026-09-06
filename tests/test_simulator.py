import pytest

from chring.simulator import (
    make_keys,
    make_nodes,
    simulate_distribution,
    simulate_topology_change,
)


def test_make_keys_and_nodes_deterministic_and_sized():
    assert make_keys(5) == ["key-0", "key-1", "key-2", "key-3", "key-4"]
    assert make_nodes(3) == ["node-0", "node-1", "node-2"]


def test_remove_requires_at_least_two_nodes():
    with pytest.raises(ValueError):
        simulate_topology_change(
            initial_node_count=1, key_count=10, event="remove"
        )


def test_topology_change_add_reports_both_algorithms():
    results = simulate_topology_change(
        initial_node_count=10, key_count=5000, event="add", replicas=100
    )
    assert set(results) == {"consistent", "naive"}
    consistent = results["consistent"]
    naive = results["naive"]
    assert consistent.nodes_before == 10
    assert consistent.nodes_after == 11
    assert consistent.total_keys == 5000
    assert naive.nodes_after == 11


def test_consistent_moves_far_fewer_keys_than_naive_on_add():
    results = simulate_topology_change(
        initial_node_count=10, key_count=10_000, event="add", replicas=100
    )
    assert results["consistent"].moved_fraction < results["naive"].moved_fraction / 3


def test_consistent_moves_far_fewer_keys_than_naive_on_remove():
    results = simulate_topology_change(
        initial_node_count=10, key_count=10_000, event="remove", replicas=100
    )
    assert results["consistent"].moved_fraction < results["naive"].moved_fraction / 3


def test_consistent_movement_close_to_theoretical_minimum():
    results = simulate_topology_change(
        initial_node_count=19, key_count=20_000, event="add", replicas=150
    )
    consistent = results["consistent"]
    # should be in the right ballpark of the theoretical minimum (1/20)
    assert consistent.moved_fraction < consistent.theoretical_minimum_fraction * 2.5


def test_naive_movement_is_severe():
    results = simulate_topology_change(
        initial_node_count=19, key_count=20_000, event="add", replicas=100
    )
    assert results["naive"].moved_fraction > 0.7


def test_simulate_distribution_shape():
    result = simulate_distribution(node_count=6, key_count=12_000, replicas=100)
    assert result.node_count == 6
    assert result.total_keys == 12_000
    assert sum(result.counts.values()) == 12_000
    assert result.mean == pytest.approx(2000.0)
    assert result.coefficient_of_variation < 0.15


def test_distribution_result_handles_zero_nodes_gracefully():
    from chring.simulator import DistributionResult

    result = DistributionResult(replicas=10, node_count=0, total_keys=0, counts={})
    assert result.mean == 0.0
    assert result.stdev == 0.0
    assert result.coefficient_of_variation == 0.0
    assert result.min_count == 0
    assert result.max_count == 0


def test_theoretical_minimum_fraction_add_vs_remove():
    add_results = simulate_topology_change(
        initial_node_count=9, key_count=100, event="add", replicas=50
    )
    remove_results = simulate_topology_change(
        initial_node_count=10, key_count=100, event="remove", replicas=50
    )
    assert add_results["consistent"].theoretical_minimum_fraction == pytest.approx(1 / 10)
    assert remove_results["consistent"].theoretical_minimum_fraction == pytest.approx(1 / 10)
