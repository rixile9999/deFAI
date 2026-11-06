import importlib
import math
import unittest

import numpy as np

from defai_research.metrics import (
    adaptation_metrics, categorical_joint, classification_metrics, communication_metrics,
    conditional_mi, convergence_time, entropy_counts, evaluate16,
    gaussian_privacy_accountant, information_metrics, knowledge_integration,
    metric_definitions, multimodal_alignment, recovery_metrics, resource_usage,
    transmission_rate, update_distortion,
)


class MetricPurposeTests(unittest.TestCase):
    def test_registry_is_complete_and_missing_stays_missing(self):
        self.assertEqual([m['id'] for m in metric_definitions()], list(range(1, 17)))
        self.assertEqual(len({m['field'] for m in metric_definitions()}), 16)
        self.assertTrue(all(m['status'] == 'not_measured' for m in evaluate16()))
        observed = {m['field']: {'controlled_measurement': True} for m in metric_definitions()}
        self.assertTrue(all(m['status'] == 'measured' for m in evaluate16(observations=observed)))

    def test_precision_recall_and_classwise_f1_known_answer(self):
        y = np.array([0, 0, 1, 1])
        p = np.eye(2)[[0, 1, 1, 1]]
        r = classification_metrics(y, p)
        self.assertAlmostEqual(r['macro_precision'], 5/6)
        self.assertAlmostEqual(r['macro_recall'], 3/4)
        self.assertAlmostEqual(r['macro_f1'], (2/3+4/5)/2)
        self.assertEqual(r['confusion_matrix'], [[1, 1], [0, 2]])
        self.assertEqual([m['predicted'] for m in r['per_class']], [1, 3])

    def test_six_class_api_distinguishes_perfect_and_collapsed_predictions(self):
        y = np.tile(np.arange(6), 2)
        perfect = classification_metrics(y, np.eye(6)[y])
        collapsed = classification_metrics(y, np.eye(6)[np.zeros(len(y), dtype=int)])
        for key in ('macro_precision', 'macro_recall', 'macro_f1'):
            self.assertEqual(perfect[key], 1.)
            self.assertLess(collapsed[key], perfect[key])
        self.assertEqual(len(collapsed['per_class']), 6)
        self.assertEqual(collapsed['per_class'][5]['predicted'], 0)
        rows = evaluate16(labels=y, probabilities=np.eye(6)[y])
        self.assertEqual([rows[i]['value'] for i in range(3)], [1., 1., 1.])

    def test_invalid_probability_domain_is_rejected(self):
        for p in (np.array([[.8, .8]]), np.array([[-.1, 1.1]]), np.array([[np.nan, 1.]])):
            with self.assertRaises(ValueError):
                classification_metrics(np.array([0]), p)

    def test_entropy_and_information_are_not_reconstruction_loss(self):
        y = np.array([0, 0, 1, 1])
        perfect = information_metrics(y, np.eye(2)[y])
        constant = information_metrics(y, np.eye(2)[np.zeros(4, dtype=int)])
        reversed_labels = information_metrics(y, np.eye(2)[1-y])
        self.assertEqual(entropy_counts([1, 1]), 1.)
        self.assertEqual(entropy_counts([0, 0]), 0.)
        self.assertAlmostEqual(perfect['mutual_information_bits'], 1.)
        self.assertAlmostEqual(constant['mutual_information_bits'], 0.)
        self.assertAlmostEqual(reversed_labels['mutual_information_bits'], 1.)

    def test_conditional_information_known_xor_answer(self):
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        y = a ^ b
        joint = categorical_joint(y, a, b, sizes=(2, 2, 2))
        self.assertAlmostEqual(conditional_mi(joint, (0,), (2,), (1,)), 1.)
        self.assertAlmostEqual(conditional_mi(joint.sum(1), (0,), (1,)), 0.)
        with self.assertRaises(ValueError):
            conditional_mi(joint, (0,), (1,), (1, 2))

    def test_convergence_requires_sustained_predeclared_quality(self):
        reached = convergence_time([.8, .91, .92, .93], [1, 2, 4, 7], threshold=.9)
        missed = convergence_time([.95, .89, .95, .89], [1, 2, 4, 7], threshold=.9)
        self.assertEqual(reached['completion_round'], 4)
        self.assertEqual(reached['elapsed_seconds'], 7.)
        self.assertFalse(missed['reached'])
        self.assertIsNone(missed['elapsed_seconds'])

    def test_communication_counts_full_supplied_frames(self):
        r = communication_metrics([1040, 540], reference_bytes=2000)
        self.assertEqual(r['total_framed_bytes'], 1580)
        self.assertAlmostEqual(r['savings_fraction'], .21)
        with self.assertRaises(ValueError):
            communication_metrics([10.5])

    def test_alignment_distinguishes_disagreement_and_missingness(self):
        same = np.array([[[1., 0.]], [[1., 0.]]])
        opposite = np.array([[[1., 0.]], [[0., 1.]]])
        self.assertEqual(multimodal_alignment(same)['predictive_jsd_bits'], 0.)
        self.assertEqual(multimodal_alignment(opposite)['predictive_jsd_bits'], 1.)
        missing = multimodal_alignment(same, availability=[[1], [0]])
        self.assertIsNone(missing['predictive_jsd_bits'])
        self.assertEqual(missing['coverage_by_modality'], [1., 0.])

    def test_adaptation_tracks_loss_area_and_censored_recovery(self):
        result = adaptation_metrics([.8]*9, [.8,.8,.4,.6,.79,.8,.81,.82,.8], shock_start=3, return_start=7)
        self.assertAlmostEqual(result['positive_deficit_quality_rounds'], .61)
        self.assertIsNone(result['recovery_observations_after_shock'])
        self.assertEqual(result['recovery_observations_after_return'], 3)

    def test_integration_can_detect_negative_transfer(self):
        positive = knowledge_integration(.9, [.6, .7], ablated_score=.8)
        negative = knowledge_integration(.5, [.6, .7])
        self.assertAlmostEqual(positive['global_minus_isolated'], .25)
        self.assertLess(negative['global_minus_isolated'], 0)

    def test_resource_units_stay_separate(self):
        r = resource_usage([1., 2.], [2**20, 3*2**20])
        self.assertEqual(r['total_process_cpu_seconds'], 3.)
        self.assertEqual(r['sampled_peak_rss_MiB'], 3.)
        self.assertNotIn('mixed_unit_score', r)

    def test_privacy_composes_transcript_and_accounts_zero_noise(self):
        weak = gaussian_privacy_accountant(20, 2.)
        strong = gaussian_privacy_accountant(20, 32.)
        longer = gaussian_privacy_accountant(40, 32.)
        self.assertLess(strong['epsilon'], weak['epsilon'])
        self.assertGreater(longer['epsilon'], strong['epsilon'])
        self.assertIsNone(gaussian_privacy_accountant(20, 0.)['epsilon'])
        with self.assertRaises(ValueError):
            gaussian_privacy_accountant(20, 1., delta=1.)

    def test_actual_received_distortion_and_recovery(self):
        reference = np.array([1., 2.])
        damage = np.array([0., 0.])
        self.assertEqual(update_distortion(reference, damage)['relative_l2'], 1.)
        recovery = recovery_metrics(reference, damage, reference, reference_utility=.9, damaged_utility=.3, restored_utility=.9)
        self.assertEqual(recovery['vector_recovery_fraction'], 1.)
        self.assertEqual(recovery['utility_recovery_fraction'], 1.)
        self.assertIsNone(update_distortion(damage, damage)['relative_l2'])
        tied = recovery_metrics(reference, damage, reference, reference_utility=.9, damaged_utility=.9, restored_utility=.9)
        self.assertIsNone(tied['utility_recovery_fraction'])
        self.assertEqual(tied['utility_status'], 'nonpositive_or_tiny_headroom')

    def test_transfer_rate_requires_actual_timing_scope(self):
        self.assertEqual(transmission_rate(1000, 2., scope='training-free loopback echo')['application_bits_per_second'], 4000.)
        with self.assertRaises(ValueError):
            transmission_rate(1000, 2., scope='')

    def test_retained_sensor_feature_extractor_and_metrics(self):
        module = importlib.import_module('defai_research.metrics.studies.purpose_multimodal_study')
        features = module.sensor_features(np.zeros((2, 128, 3)))
        self.assertEqual(features.shape, (2, 36))
        score = module.score(np.arange(6), np.eye(6))
        self.assertEqual([score[k] for k in ('macro_precision', 'macro_recall', 'macro_f1')], [1., 1., 1.])


if __name__ == '__main__':
    unittest.main()
