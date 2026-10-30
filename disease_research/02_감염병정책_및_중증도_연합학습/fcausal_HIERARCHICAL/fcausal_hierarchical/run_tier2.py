"""Tier 2: build additive moments on a complete calendar, then encrypt."""
import argparse
from pathlib import Path

import flwr as fl

from core.policy_stats import (AnalysisConfig, add_analysis_arguments, config_from_args,
                               daily_counts, district_identity, district_vector,
                               district_treatment, read_district_file)
from core.policy_transport import encode, load_context, metadata


class DistrictClient(fl.client.NumPyClient):
    def __init__(self, filepath, context=None, use_smpc=True, config=None, treated=None, key_id="plaintext"):
        self.context = context
        if use_smpc and context is None:
            raise ValueError("Encrypted mode requires a public context")
        self.use_smpc = use_smpc
        self.config = config or AnalysisConfig()
        self.key_id = key_id
        frame = read_district_file(filepath)
        self.province, self.district = district_identity(frame)
        assignment = district_treatment(frame, self.province)
        if treated is not None and "treated" in frame and int(treated) != assignment:
            raise ValueError("CLI treatment disagrees with generated dataset assignment")
        self.treated = assignment if treated is None else int(treated)
        self.counts = daily_counts(frame, self.config)
        self.stats = district_vector(self.counts, self.config)

    def get_parameters(self, config):
        return []

    def fit(self, parameters, config):
        info = {**metadata(self.config, self.use_smpc, self.key_id),
                "province": self.province, "district": self.district, "treated": self.treated}
        print(f"District {self.district}: {len(self.counts)} days (including zeros), additive moments", flush=True)
        return encode(self.stats, self.context if self.use_smpc else None), len(self.counts), info

    def evaluate(self, parameters, config):
        return 0.0, len(self.counts), {}


def main():
    parser = argparse.ArgumentParser(description="Tier 2 - District additive statistics")
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--plaintext", action="store_true")
    parser.add_argument("--key-dir", default="keys")
    parser.add_argument("--treated", type=int, choices=[0, 1], default=None)
    add_analysis_arguments(parser)
    args = parser.parse_args()
    ctx, key_id = (None, "plaintext") if args.plaintext else load_context(args.key_dir)
    client = DistrictClient(Path(args.data), ctx, not args.plaintext,
                            config_from_args(args), args.treated, key_id)
    fl.client.start_client(server_address=args.target, client=client.to_client(),
                           grpc_max_message_length=1024*1024*1024)


if __name__ == "__main__":
    main()
