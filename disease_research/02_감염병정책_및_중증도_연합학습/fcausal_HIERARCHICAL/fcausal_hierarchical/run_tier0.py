"""Tier 0: decrypt province moments and compute a defined policy contrast."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import flwr as fl

from core.policy_stats import (AnalysisConfig, add_analysis_arguments, analyze, config_from_args,
                               print_analysis, province_summary)
from core.policy_transport import decode, load_context, validate_metadata


class CentralStrategy(fl.server.strategy.FedAvg):
    def __init__(self, context=None, config=None, key_id="plaintext", output=None, **kwargs):
        kwargs.update(initial_parameters=fl.common.ndarrays_to_parameters([]),
                      fraction_evaluate=0.0, min_evaluate_clients=0, accept_failures=False)
        super().__init__(**kwargs)
        self.context = context
        self.config = config or AnalysisConfig()
        self.key_id = key_id
        self.output = Path(output) if output else None
        self.result = None

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.min_fit_clients:
            raise ValueError("Incomplete province cohort")
        provinces = []
        for _, response in results:
            info = response.metrics
            validate_metadata(info, self.config, self.context is not None, self.key_id)
            value = decode(fl.common.parameters_to_ndarrays(response.parameters), self.context)
            vector = value.decrypt() if self.context is not None else value
            provinces.append(province_summary(vector, info["region"], info["treated"],
                                              self.config, info["n_districts"]))
        self.result = analyze(sorted(provinces, key=lambda p: p["region"]))
        self.result["config"] = asdict(self.config)
        self.result["encrypted"] = self.context is not None
        print_analysis(self.result)
        if self.output:
            self.output.parent.mkdir(parents=True, exist_ok=True)
            self.output.write_text(json.dumps(self.result, ensure_ascii=False, indent=2, allow_nan=False),
                                   encoding="utf-8")
        return None, {}


def main():
    parser = argparse.ArgumentParser(description="Tier 0 - Research policy analysis")
    parser.add_argument("--address", default="127.0.0.1:8080")
    parser.add_argument("--min-provinces", type=int, default=2, help="Exact planned province cohort")
    parser.add_argument("--plaintext", action="store_true")
    parser.add_argument("--key-dir", default="keys")
    parser.add_argument("--output", default="results/latest_policy_analysis.json")
    add_analysis_arguments(parser)
    args = parser.parse_args()
    ctx, key_id = (None, "plaintext") if args.plaintext else load_context(args.key_dir, secret=True)
    strategy = CentralStrategy(ctx, config_from_args(args), key_id, args.output,
                               min_fit_clients=args.min_provinces,
                               min_available_clients=args.min_provinces)
    fl.server.start_server(server_address=args.address,
                           config=fl.server.ServerConfig(num_rounds=1, round_timeout=120),
                           strategy=strategy, grpc_max_message_length=1024*1024*1024)
    if strategy.result is None:
        raise RuntimeError("Analysis did not complete")


if __name__ == "__main__":
    main()
