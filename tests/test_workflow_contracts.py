"""Input failures are explicit, and reports do not depend on an unpublished template."""
import importlib.util
import json

import numpy as np
import pandas as pd
import pytest

from defai.cli import main
from defai.research.resources import catalog, source_path


@pytest.mark.parametrize("row", ((11001, 0), (11001, float("nan")), (11001.5, 2)))
def test_generation_cli_rejects_invalid_rows_without_silently_filtering(tmp_path, row):
    path = tmp_path / "input.csv"
    pd.DataFrame([row], columns=["child_sigungu", "T_days"]).to_csv(path, index=False)
    assert main(["generation", "--input", str(path), "--output", str(tmp_path / "out.json")]) == 2
    assert not (tmp_path / "out.json").exists()


def test_policy_cli_complete_panels_and_invalid_schema(tmp_path):
    panels = tmp_path / "panels"
    panels.mkdir()
    rng = np.random.default_rng(42)
    for i in range(6):
        pd.DataFrame({"record_format": "daily_counts_v1", "date": pd.date_range("2024-01-01", periods=100),
                      "cases": rng.poisson(5, 100), "시도": f"province{i}", "시군구": f"district{i}",
                      "treated": int(i >= 3)}).to_csv(panels / f"{i}.csv", index=False)
    output = tmp_path / "analysis.json"
    assert main(["policy", "--input", str(panels), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["stage"] == "research_only"
    (panels / "bad.csv").write_text("wrong,column\n1,2\n")
    assert main(["policy", "--input", str(panels), "--output", str(tmp_path / "bad.json")]) == 2


def test_report_base_works_without_design_and_never_duplicates_results(tmp_path, monkeypatch):
    path = next(p for p in catalog()["resources"] if p.endswith("fl_comparison/report_results.py"))
    source = source_path(path)
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("hba_report_contract_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    html, backup = tmp_path / "result.html", tmp_path / "base.html"
    base = module.report_base(html, backup)
    assert "<main>" in base and "</html>" in base
    html.write_text(base.replace("<main>", "<main>old computed section"), encoding="utf-8")
    assert module.report_base(html, backup) == base
