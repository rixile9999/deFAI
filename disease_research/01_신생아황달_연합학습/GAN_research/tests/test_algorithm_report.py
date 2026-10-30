"""The reported winner must remain validation-selected even if test disagrees."""
import json
from hashlib import sha256
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import summarize_algorithm_study as report


def test_report_keeps_validation_winner_and_discloses_all_methods(tmp_path, monkeypatch):
    root = tmp_path
    output = root / "results/algorithm_study"; output.mkdir(parents=True)
    (root / "data").mkdir(); (root / "data/real_data.npz").write_bytes(b"fixture hash only")
    legacy = root / "results/skin_bias"; legacy.mkdir()
    families = ["fedavg_adam", "fedprox", "fedavg_sgd", "scaffold", "fedbn", "fedavg_trainbn"]
    selections, runs, locked, old = [], [], [], []
    blood = np.linspace(3, 21, 112)
    for context_index, (condition, bias) in enumerate(report.CONTEXTS):
        old.append({"condition": condition, "method": "gain_consistency" if bias else "baseline", "mae_mgdl": .5})
        locked.append({"condition": condition, "bias": bias, "family": "fedavg_adam"})
        for family_index, family in enumerate(families):
            config = {"id": family, "family": family, "lr": .01}
            selections.append({"condition": condition, "bias": bias, "family": family, "config": config})
            for seed in [17, 29, 43]:
                name = f"{condition}_bias{int(bias)}_{family}_seed{seed}"
                # FedProx wins test by construction; FedAvg wins validation.
                error = .1 if family == "fedprox" else .5
                frame = pd.DataFrame({"patient": [f"{i:04d}" for i in range(112)], "client": np.arange(112)%3,
                    "blood": blood, "treatment": (blood >= 12).astype(int), "prediction": blood+error,
                    "probability": np.where(blood >= 12, .8, .2), "group": np.arange(112)%3,
                    "brightness": np.linspace(.2, .8, 112)})
                for gain in [1., .55, .75, .9]:
                    changed = frame.copy(); changed.prediction += (1-gain)*.1
                    changed.to_csv(output / f"{name}_gain{gain:.2f}_test.csv", index=False)
                runs.append({"name": name, "condition": condition, "bias": bias, "config": config, "seed": seed,
                             "validation_mae": 1.8+family_index*.1+context_index*.01, "best_round": 20,
                             "seconds": 1., "scaffold_mean_control_max_error": 0.})
    (legacy / "summary.json").write_text(json.dumps({"summary": old}))
    (output / "complete.json").write_text(json.dumps({"trainings": len(runs), "selected_models": len(runs)}))
    (output / "protocol.json").write_text(json.dumps({"client_weights": [174, 173, 172],
        "data_sha256": sha256((root / "data/real_data.npz").read_bytes()).hexdigest()}))
    (output / "selection.json").write_text(json.dumps(selections))
    (output / "training_summary.json").write_text(json.dumps(runs))
    (output / "validation_winners_locked.json").write_text(json.dumps({"per_context": locked, "overall": locked[0]}))
    monkeypatch.setattr(report, "ROOT", root); monkeypatch.setattr(report, "OUTPUT", output)
    report.main()
    result = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert result["validation_selected_overall"]["family"] == "fedavg_adam"
    assert len(result["summary"]) == 24
    assert len(result["subgroups"]) == 72
    contrast = next(c for c in result["contrasts"] if c["family"] == "fedprox")
    assert contrast["differences"]["mae_mgdl"]["difference"] < -.39
    assert (output / "report_section.html").is_file()
