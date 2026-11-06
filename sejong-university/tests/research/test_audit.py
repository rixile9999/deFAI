import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from defai_research.audit import verify_benchmark


@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Transformer extra required by benchmark integration")
class BenchmarkAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from defai_research.compression.runner import run_benchmark
        from defai_research import aggregation
        cls.fixture = tempfile.TemporaryDirectory(prefix="defai-audit-fixture-")
        cls.original = Path(cls.fixture.name) / "run"
        hashes = {"aggregation.py": hashlib.sha256(Path(aggregation.__file__).read_bytes()).hexdigest()}
        with patch("defai_research.compression.runner._source_hashes", return_value=hashes):
            run_benchmark(output=cls.original, codec="compact", dataset="digits", seed=48001,
                          rounds=2, drift_round=2, graph_epochs=2)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    def setUp(self):
        self.copy = tempfile.TemporaryDirectory(prefix="defai-audit-case-")
        self.run = Path(self.copy.name) / "run"
        shutil.copytree(self.original, self.run)

    def tearDown(self):
        self.copy.cleanup()

    def test_completed_artifacts_pass_and_remain_read_only(self):
        before = {p.relative_to(self.run): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in self.run.rglob("*") if p.is_file()}
        result = verify_benchmark(self.run)
        self.assertTrue(result["passed"], result["failures"])
        self.assertGreater(result["check_count"], 250)
        after = {p.relative_to(self.run): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in self.run.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_tampered_model_ledger_is_rejected(self):
        path = self.run / "model_wire_audit.json"
        value = json.loads(path.read_text())
        value[0]["frames"][0]["frame_bytes"] += 17
        path.write_text(json.dumps(value), encoding="utf-8")
        result = verify_benchmark(self.run)
        self.assertFalse(result["passed"])
        self.assertIn("round 1: frame accounting", result["failures"])

    def test_undeclared_test_artifact_is_rejected(self):
        for filename in ("protocol.json", "final.json"):
            path = self.run / filename
            value = json.loads(path.read_text())
            value["test_evaluated"] = False
            path.write_text(json.dumps(value), encoding="utf-8")
        result = verify_benchmark(self.run)
        self.assertFalse(result["passed"])
        self.assertIn("test artifacts excluded", result["failures"])

    def test_missing_required_ledger_is_an_error(self):
        (self.run / "wire_audit.json").unlink()
        with self.assertRaises(FileNotFoundError):
            verify_benchmark(self.run)


if __name__ == "__main__":
    unittest.main()
