import tempfile
import unittest
import hashlib
from pathlib import Path

import numpy as np
import torch

from defai_research.compression import (TransformerCodecConfig, TransformerBlockAutoencoder,
                               fit_transformer_codec, export_inference_artifact,
                               TransformerCompactCodec)
from defai_research.compression.lossless import self_test
from defai_research.compression.legacy import LegacyLosslessCodec
from defai_research.compression.runner import run_benchmark


class CompressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        blocks = np.random.default_rng(902).normal(size=(24, 256)).astype(np.float32)
        mask = np.arange(len(blocks)) >= 16
        fitted = fit_transformer_codec(blocks, cls.root/'trained',
            config=TransformerCodecConfig(epochs=2, batch_size=8, patience=0),
            validation_mask=mask, manifest={'test_data_used': False, 'scope': 'synthetic codec unit fixture'})
        cls.source = Path(fitted['artifact_path'])
        cls.artifact = Path(export_inference_artifact(cls.source)['artifact_path'])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_actual_attention_architecture(self):
        model = TransformerBlockAutoencoder(TransformerCodecConfig())
        self.assertIsInstance(model.encoder, torch.nn.TransformerEncoder)
        self.assertIsInstance(model.decoder, torch.nn.TransformerDecoder)
        self.assertIsInstance(model.latent_attention, torch.nn.MultiheadAttention)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 2360)

    def test_lossless_wire_guards_and_exact_arrays(self):
        result = self_test()
        self.assertEqual(result['status'], 'passed')
        self.assertTrue(result['flower_npy_byte_exact'])

    def test_legacy_reference_is_byte_exact_with_independent_receiver(self):
        sender, receiver = LegacyLosslessCodec(), LegacyLosslessCodec()
        arrays = [np.arange(257, dtype=np.float32), np.array([-0., 1.], np.float64)]
        channel = ('edge', 'up', 0, 0)
        for rnd in (1, 2):
            sent = sender.transmit(arrays, channel=channel, round_id=rnd)
            decoded = receiver.decode_frame(sent.frame, expected_channel=channel, expected_round=rnd)
            for original, received in zip(arrays, decoded.arrays):
                self.assertEqual(original.dtype, received.dtype)
                self.assertEqual(original.tobytes(), received.tobytes())

    def test_deployment_is_required_and_independent_receiver_matches(self):
        codec = TransformerCompactCodec(self.artifact)
        values = np.random.default_rng(41).normal(size=513).astype(np.float32)
        channel = ('edge', 'up', 0, 0)
        with self.assertRaises(RuntimeError):
            codec.transmit([values], channel=channel, round_id=1)
        packets = codec.deployment_packets()
        self.assertEqual(len(packets), 16)
        self.assertEqual(codec.residual_topk, 8)
        self.assertEqual(codec.residual_bits, 8)
        self.assertTrue(codec.block_gain)
        packet = next(p for p in packets if p['destination']==['group', 0])
        receiver = TransformerCompactCodec.receiver_from_deployment(packet['payload'])
        self.assertEqual(set(receiver._endpoint_artifacts), {('group', 0)})
        sent = codec.transmit([values], channel=channel, round_id=1)
        received, metadata = receiver.decode_frame(sent.frame, channel=channel, round_id=1)
        np.testing.assert_array_equal(received[0], sent.decoded_arrays[0])
        self.assertEqual(received[0].shape, values.shape)
        self.assertTrue(np.isfinite(received[0]).all())
        self.assertEqual(metadata['residual_bits'], 8)
        with self.assertRaises(ValueError):
            receiver.decode_frame(sent.frame, channel=channel, round_id=1)

    def test_corruption_and_wrong_channel_rejected_before_state_advance(self):
        codec = TransformerCompactCodec(self.artifact)
        packet = next(p for p in codec.deployment_packets() if p['destination']==['group', 0])
        receiver = TransformerCompactCodec.receiver_from_deployment(packet['payload'])
        channel = ('edge', 'up', 0, 0)
        sent = codec.transmit([np.ones(257, np.float32)], channel=channel, round_id=1)
        corrupted = bytearray(sent.frame)
        corrupted[-1] ^= 1
        with self.assertRaises(ValueError):
            receiver.decode_frame(bytes(corrupted), channel=channel, round_id=1)
        with self.assertRaises(ValueError):
            receiver.decode_frame(sent.frame, channel=('edge', 'up', 1, 0), round_id=1)
        received, _ = receiver.decode_frame(sent.frame, channel=channel, round_id=1)
        np.testing.assert_array_equal(received[0], sent.decoded_arrays[0])

    def test_train_only_calibration_guard(self):
        with self.assertRaises(ValueError):
            fit_transformer_codec(np.zeros((4,256),np.float32), self.root/'forbidden',
                                  validation_mask=[False,False,True,True],
                                  manifest={'test_data_used': True})

    def test_modified_calibration_manifest_is_rejected_before_training(self):
        path = self.root/'mutable_calibration.npz'
        blocks = np.zeros((4,256), np.float32)
        np.savez_compressed(path, blocks=blocks)
        manifest = dict(status='completed', npz_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        blocks[0,0] = 1.
        np.savez_compressed(path, blocks=blocks)
        target = self.root/'checksum_rejected'
        with self.assertRaisesRegex(ValueError, 'manifest checksum'):
            fit_transformer_codec(path, target, validation_mask=[False,False,True,True], manifest=manifest)
        self.assertFalse(target.exists())

    def test_received_transformer_updates_drive_two_hop_flower(self):
        result = run_benchmark(output=self.root/'integrated', dataset='digits', seed=9301,
            rounds=4, drift_round=3, codec='transformer', artifact=self.artifact,
            graph_epochs=3, allow_test=False)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['received_updates_used_by_flower'])
        self.assertTrue(result['separate_receiver_objects'])
        self.assertEqual(result['sum_lost_unique_records'], 0)
        self.assertGreater(result['sum_transformer_frames'], 0)
        self.assertGreaterEqual(result['graphsage_fits'], 2)
        self.assertGreater(result['sum_codec_deployment_bytes'], 0)
        self.assertFalse(result['test_evaluated'])


if __name__ == '__main__':
    unittest.main()
