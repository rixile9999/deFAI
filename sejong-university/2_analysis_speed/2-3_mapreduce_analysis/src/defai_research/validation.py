from __future__ import annotations

import numpy as np

from .aggregation import (
    aggregate_sufficient_statistics, keyed_shuffle_reduce,
    local_sufficient_statistics, map_reduce,
)


def verify_aggregation(*, full=False):
    conditions = []
    sizes = (12, 48, 192) if full else (12, 48)
    widths = (2410, 24100) if full else (257,)
    repeats = range(5) if full else range(2)
    for clients in sizes:
        for width in widths:
            for groups in (1, 4, 12):
                for repeat in repeats:
                    conditions.append((clients, width, groups, 256, repeat))
    if full:
        conditions.extend((192, width, 12, block, repeat)
                          for width in widths for block in (64, 1024) for repeat in repeats)
    max_gap = 0.0
    for clients, width, group_count, block, repeat in conditions:
        rng = np.random.default_rng(np.random.SeedSequence([clients, width, block, repeat, 731]))
        updates = list(rng.normal(size=(clients, width)))
        counts = rng.integers(1, 1000, size=clients).tolist()
        groups = [0] * (clients - group_count + 1) + list(range(1, group_count))
        actual, _, _, gap = map_reduce(updates, counts, groups, block)
        expected = np.average(np.stack(updates), axis=0, weights=counts)
        if width <= 2410:
            shuffled, _ = keyed_shuffle_reduce(updates, counts, groups, block, repeat)
            np.testing.assert_allclose(shuffled, expected, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        max_gap = max(max_gap, gap, float(np.max(np.abs(actual - expected))))
    stats_cases = []
    for clients in ((12, 48, 192) if full else (12,)):
        for features in (4, 19):
            for repeat in repeats:
                rng = np.random.default_rng(np.random.SeedSequence([clients, features, repeat, 739]))
                arrays = [rng.normal(size=(int(n), features))
                          for n in rng.integers(1, 20, clients)]
                groups = [cid % 4 for cid in range(clients)]
                result = aggregate_sufficient_statistics(
                    [local_sufficient_statistics(x) for x in arrays], groups)
                pooled = np.concatenate(arrays)
                np.testing.assert_allclose(result['mean'], pooled.mean(0), rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(result['population_variance'], pooled.var(0), rtol=1e-12, atol=1e-12)
                stats_cases.append(len(pooled))
    return {'passed': True, 'vector_conditions': len(conditions),
            'statistic_conditions': len(stats_cases), 'maximum_absolute_gap': max_gap,
            'scope': 'FP64 weighted numerical identities; no communication speed or lossy-codec accuracy claim'}
