import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_incident_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("incident_benchmarks", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_cluster_indices_respects_gap():
    assert MODULE.cluster_indices([1, 2, 4, 9], max_gap=2) == [[1, 2, 4], [9]]


def test_exact_random_distribution_single_draw():
    distribution = MODULE.exact_random_episode_distribution(4, 1, [1, 1])
    assert distribution == {0: 0.5, 1: 0.5}


def test_exact_random_distribution_is_normalized():
    distribution = MODULE.exact_random_episode_distribution(20, 5, [2, 3, 1])
    assert abs(sum(distribution.values()) - 1.0) < 1e-12


def test_exact_random_expected_hits_matches_episode_formula():
    windows, budget = 30, 6
    sizes = [2, 4, 3]
    distribution = MODULE.exact_random_episode_distribution(windows, budget, sizes)
    expected = sum(hits * probability for hits, probability in distribution.items())
    direct = sum(
        1 - MODULE.math.comb(windows - size, budget) / MODULE.math.comb(windows, budget)
        for size in sizes
    )
    assert abs(expected - direct) < 1e-12
