"""Tier 1: sum additive statistics without a secret key, then forward once."""
import argparse

import flwr as fl
import numpy as np

from core.policy_stats import AnalysisConfig, add_analysis_arguments, config_from_args, validate_vector
from core.policy_transport import (decode, load_context, metadata, normalize_region,
                                   validate_metadata)


class ProvinceStrategy(fl.server.strategy.FedAvg):
    def __init__(self, context, region_name, config=None, treated=False,
                 key_id="plaintext", **kwargs):
        kwargs.update(initial_parameters=fl.common.ndarrays_to_parameters([]),
                      fraction_evaluate=0.0, min_evaluate_clients=0, accept_failures=False)
        super().__init__(**kwargs)
        if context is not None and context.has_secret_key():
            raise ValueError("Tier 1 must not hold a secret key")
        self.context = context
        self.region_name = normalize_region(region_name)
        self.config = config or AnalysisConfig()
        self.treated = int(treated)
        self.key_id = key_id
        self.aggregated_stats = None
        self.n_districts = 0

    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.min_fit_clients:
            raise ValueError("Incomplete district cohort; refusing to change the estimand silently")
        total, seen = None, set()
        for _, response in results:
            info = response.metrics
            validate_metadata(info, self.config, self.context is not None, self.key_id)
            if normalize_region(info["province"]) != self.region_name or info["treated"] != self.treated:
                raise ValueError("Mixed province/treatment assignment")
            if info["district"] in seen:
                raise ValueError("Duplicate district received")
            seen.add(info["district"])
            value = decode(fl.common.parameters_to_ndarrays(response.parameters), self.context)
            if self.context is None:
                validate_vector(value, self.config, expected_districts=1)
            total = value if total is None else total + value
        self.aggregated_stats = total
        self.n_districts = len(results)
        print(f"Province {self.region_name}: summed {self.n_districts} districts; "
              f"encrypted={self.context is not None}; secret_key=False", flush=True)
        return None, {}


class ProvinceClient(fl.client.NumPyClient):
    def __init__(self, stats, region_name, n_districts, is_treated,
                 config=None, encrypted=True, key_id="plaintext"):
        self.stats = stats
        self.region_name = normalize_region(region_name)
        self.n_districts = n_districts
        self.treated = int(is_treated)
        self.config = config or AnalysisConfig()
        self.encrypted = encrypted
        self.key_id = key_id

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        array = np.frombuffer(self.stats.serialize(), dtype=np.uint8) if self.encrypted else self.stats
        info = {**metadata(self.config, self.encrypted, self.key_id), "region": self.region_name,
                "n_districts": self.n_districts, "treated": self.treated}
        return [array], self.n_districts, info

    def evaluate(self, parameters, config):
        return 0.0, self.n_districts, {}


def main():
    parser = argparse.ArgumentParser(description="Tier 1 - Blind additive aggregation")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--treated", action="store_true")
    parser.add_argument("--min-districts", type=int, default=2, help="Exact planned district cohort")
    parser.add_argument("--target", default="127.0.0.1:8080")
    parser.add_argument("--plaintext", action="store_true")
    parser.add_argument("--key-dir", default="keys")
    add_analysis_arguments(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    ctx, key_id = (None, "plaintext") if args.plaintext else load_context(args.key_dir)
    strategy = ProvinceStrategy(ctx, args.region, config, args.treated, key_id,
                                min_fit_clients=args.min_districts,
                                min_available_clients=args.min_districts)
    fl.server.start_server(server_address=f"127.0.0.1:{args.port}",
                           config=fl.server.ServerConfig(num_rounds=1, round_timeout=120),
                           strategy=strategy, grpc_max_message_length=1024*1024*1024)
    if strategy.aggregated_stats is None:
        raise RuntimeError("No district statistics aggregated")
    client = ProvinceClient(strategy.aggregated_stats, args.region, strategy.n_districts,
                            args.treated, config, ctx is not None, key_id)
    fl.client.start_client(server_address=args.target, client=client.to_client(),
                           grpc_max_message_length=1024*1024*1024)


if __name__ == "__main__":
    main()
