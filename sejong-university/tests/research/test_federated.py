import unittest
import numpy as np

from defai_research.federated import (
    GraphSAGECachedDeduplicator, ImmutableTrainingSnapshot,
    aggregate_sufficient_statistics, keyed_shuffle_reduce,
    local_sufficient_statistics, map_reduce,
)
from defai_research.federated.snapshots import (
    deserialize_digest_packet, deserialize_ownership_packet,
    serialize_digest_packet, serialize_ownership_packet,
)


class AggregationTests(unittest.TestCase):
    def test_unequal_groups_use_sample_counts_not_group_means(self):
        values = [np.array([1., 4., -3.]), np.array([9., -2., 8.]), np.array([5., 1., 6.])]
        counts, groups = [2, 13, 5], [7, 2, 7]
        expected = np.array([144., -13., 128.]) / 20
        actual, regional, totals, gap = map_reduce(values, counts, groups, block_size=2)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        self.assertEqual(totals, {2: 13, 7: 7})
        for seed in (1, 17, 913):
            shuffled, _ = keyed_shuffle_reduce(values, counts, groups, 2, seed)
            np.testing.assert_allclose(shuffled, expected, atol=1e-14)
        self.assertLessEqual(gap, 1e-14)
        self.assertGreater(np.max(np.abs(sum(regional.values()) / 2 - expected)), .1)

    def test_sufficient_statistics_match_population_with_unequal_sizes(self):
        arrays = [np.arange(12).reshape(4, 3), np.array([[21, -7, 2]]), np.arange(27).reshape(9, 3)]
        merged = aggregate_sufficient_statistics([local_sufficient_statistics(x) for x in arrays], [1, 2, 1])
        pooled = np.concatenate(arrays).astype(np.float64)
        self.assertEqual(merged['count'], 14)
        np.testing.assert_allclose(merged['mean'], pooled.mean(0), atol=1e-13)
        np.testing.assert_allclose(merged['population_variance'], pooled.var(0), atol=1e-13)

    def test_invalid_counts_and_dimensions_fail_explicitly(self):
        with self.assertRaises(ValueError):
            map_reduce([np.ones(2), np.ones(3)], [1, 2], [0, 1])
        with self.assertRaises(ValueError):
            map_reduce([np.ones(2)], [0], [0])
        with self.assertRaises(ValueError):
            map_reduce([np.ones(2)], [1], [0], 0)
        with self.assertRaises(ValueError):
            map_reduce([np.array([1+2j])], [1], [0])
        with self.assertRaises(ValueError):
            ImmutableTrainingSnapshot.from_arrays([np.ones((2, 0))])


class GraphTests(unittest.TestCase):
    @staticmethod
    def snapshot():
        rng = np.random.default_rng(319)
        x, y = [], []
        for cid in range(12):
            base = rng.normal(loc=cid / 12, size=(4, 6)).astype(np.float32)
            x.append(np.concatenate([base, base[:1], base[:1]]))
            y.append(np.array([0, 1, 0, 1, 0, 1], dtype=np.int64))
        return ImmutableTrainingSnapshot.from_arrays(x, y)

    def test_exact_identity_label_guard_and_immutable_storage(self):
        snapshot = self.snapshot()
        with self.assertRaises(ValueError):
            snapshot.x[0].setflags(write=True)
        policy = GraphSAGECachedDeduplicator(seed=42, epochs=4)
        masks, audit = policy.apply(snapshot, 1)
        self.assertTrue(all(not mask[4] and mask[5] for mask in masks))
        before = {h for client in snapshot.fingerprints for h in client}
        after = {h for mask, client in zip(masks, snapshot.fingerprints)
                 for h, keep in zip(client, mask) if keep}
        self.assertEqual(before, after)
        self.assertEqual(audit['actual_graphsage_total_fits'], 1)
        self.assertEqual(policy.history[0]['embedding_epochs'], 4)
        self.assertGreater(policy.history[0]['learned_parameter_count'], 0)
        self.assertEqual(len(policy.history[0]['loss_curve']), 4)
        self.assertTrue(all(np.isfinite(policy.history[0]['loss_curve'])))

    def test_unchanged_cache_reused_changed_snapshot_refits(self):
        snapshot = self.snapshot()
        policy = GraphSAGECachedDeduplicator(seed=42, epochs=3)
        first, a = policy.apply(snapshot, 1)
        second, b = policy.apply(snapshot, 2)
        self.assertFalse(a['mask_cache_hit'])
        self.assertTrue(b['mask_cache_hit'])
        self.assertEqual(b['actual_graphsage_total_fits'], 1)
        self.assertEqual(b['new_exact_digest_rows_sent'], 0)
        self.assertTrue(all(np.array_equal(x, y) for x, y in zip(first, second)))
        changed = [x.copy() for x in snapshot.x]
        changed[0][0, 0] += 10
        fresh = ImmutableTrainingSnapshot.from_arrays(changed, snapshot.y)
        _, c = policy.apply(fresh, 3)
        self.assertFalse(c['mask_cache_hit'])
        self.assertTrue(c['training_summary_drift_since_graph_fit'])
        self.assertEqual(c['actual_graphsage_total_fits'], 2)
        with self.assertRaises(ValueError):
            policy.apply(fresh, 3)

    def test_static_graph_stays_frozen_when_data_changes(self):
        snapshot = self.snapshot()
        policy = GraphSAGECachedDeduplicator(seed=42, dynamic=False, epochs=3)
        policy.apply(snapshot, 1)
        changed = [x.copy() for x in snapshot.x]
        changed[0] += 5
        fresh = ImmutableTrainingSnapshot.from_arrays(changed, snapshot.y)
        _, audit = policy.apply(fresh, 2)
        self.assertEqual(audit['actual_graphsage_total_fits'], 1)
        self.assertFalse(audit['mask_cache_hit'])

    def test_control_frames_reject_truncation(self):
        version = b'v' * 32
        digest = b'h' * 32
        frame = serialize_digest_packet(2, 7, version, [(3, digest)])
        self.assertEqual(deserialize_digest_packet(frame), (2, 7, version, [(3, digest)]))
        own = serialize_ownership_packet(2, 7, version, [(3, False)])
        self.assertEqual(deserialize_ownership_packet(own), (2, 7, version, [(3, False)]))
        with self.assertRaises(ValueError):
            deserialize_digest_packet(frame[:-1])
        with self.assertRaises(ValueError):
            deserialize_ownership_packet(own[:-1])


if __name__ == '__main__':
    unittest.main()
