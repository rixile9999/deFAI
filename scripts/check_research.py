"""Offline component checks across the four integrated research domains."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1] / "disease_research"
    neonatal = next(root.glob("01*/GAN_research"))
    policy = next(root.glob("02*/fcausal_HIERARCHICAL/fcausal_hierarchical"))
    generation = next(root.glob("03*/abm_fl_simulation"))
    hba1c = next(root.glob("04*/hba1c_fl_research/experiments/fl_comparison"))
    jobs = [
        (neonatal, ["-m", "pytest", "tests/test_federated_algorithms.py", "tests/test_skin_bias.py", "tests/test_algorithm_report.py", "-q"]),
        (policy, ["-m", "pytest", "tests/test_policy_stats.py", "tests/test_region_expansion.py", "-q"]),
        (generation, ["fl_pipeline/paillier.py"]),
        (generation, ["fl_pipeline/federated_burr.py"]),
        (hba1c, ["test_learning.py"]),
    ]
    for directory, arguments in jobs:
        subprocess.run([sys.executable, *arguments], cwd=directory, check=True)


if __name__ == "__main__":
    main()
