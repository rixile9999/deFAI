import unittest
from pathlib import Path

from defai_research import aggregation
from defai_research.federated import graph
from defai_research.metrics import api
from defai_research.source_files import source_files, resolve_source


class SourceLayoutTests(unittest.TestCase):
    def test_inventory_includes_all_three_module_sources(self):
        files = source_files()
        for name, module in (("aggregation.py", aggregation), ("federated/graph.py", graph), ("metrics/api.py", api)):
            self.assertEqual(files[name], Path(module.__file__).resolve())
        self.assertIn("compression/runner.py", files)
        self.assertIn("metrics/studies/purpose_system_study.py", files)

    def test_manifest_paths_cannot_escape_package_sources(self):
        for name in ("../README.md", "/aggregation.py", "federated/../aggregation.py", "federated\\graph.py"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                resolve_source(name)
        with self.assertRaises(FileNotFoundError):
            resolve_source("nonexistent.py")


if __name__ == "__main__":
    unittest.main()
