"""Inspect all district workbooks with the same definition used by all tiers."""
import argparse
from pathlib import Path

from core.policy_stats import add_analysis_arguments, config_from_args, print_analysis
from run_policy_validation import load_records, reference_analysis, run_strategies


def verify_hierarchical_data():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/districts"))
    add_analysis_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    records = load_records(args.data_dir, config)
    reference = reference_analysis(records, config)
    actual = run_strategies(records, config, (None, None), False)
    if abs(reference["effect"]-actual["effect"]) >= 1e-12:
        raise AssertionError("Pooled and hierarchical results differ")
    print_analysis(actual)
    print(f"Verified {len(records)} district files; pooled/hierarchical agreement < 1e-12")


if __name__ == "__main__":
    verify_hierarchical_data()
