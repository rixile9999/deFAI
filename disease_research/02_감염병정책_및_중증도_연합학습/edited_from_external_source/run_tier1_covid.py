import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*DEPRECATED.*")

import argparse
import logging
import time
logging.getLogger("flwr").setLevel(logging.ERROR)

import flwr as fl
import numpy as np
import tenseal as ts
from pathlib import Path


# ── City Strategy (Server mode - collect from districts) ───────
class CityStrategy(fl.server.strategy.FedAvg):
    """
    CAgg: Cluster-Aware Aggregation at city level.
    Equal weight per district regardless of size.
    This is the intra-cluster aggregation step.
    """
    def __init__(self, context, city_name, **kwargs):
        super().__init__(**kwargs)
        self.context       = context
        self.city_name     = city_name
        self.agg_weights   = None   # aggregated weights array
        self.total_n       = 0      # total training samples
        self.n_districts   = 0
        self.district_info = []

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        print(f"\n{'='*60}")
        print(f"[TIER 1] {self.city_name}: CAgg from "
              f"{len(results)} districts")
        print(f"{'='*60}")

        weights_list = []
        n_list       = []

        for _, fit_res in results:
            params     = fl.common.parameters_to_ndarrays(fit_res.parameters)
            is_enc     = fit_res.metrics.get("encrypted", False)
            district   = fit_res.metrics.get("district", "Unknown")
            n_train    = fit_res.num_examples
            local_f1   = fit_res.metrics.get("local_f1", 0)
            local_auc  = fit_res.metrics.get("local_auc", 0)
            threshold  = fit_res.metrics.get("threshold", 0.5)

            print(f"  District: {district} | n={n_train} | "
                  f"F1={local_f1:.4f} | AUC={local_auc:.4f} | "
                  f"thr={threshold:.3f}")

            if is_enc and self.context:
                # Decrypt weights
                enc_bytes = params[0].tobytes()
                vec       = ts.ckks_vector_from(self.context, enc_bytes)
                weights   = np.array(vec.decrypt(), dtype=np.float32)
            else:
                weights = params[0].astype(np.float32)

            weights_list.append(weights)
            n_list.append(n_train)

            self.district_info.append({
                "district" : district,
                "n_train"  : n_train,
                "local_f1" : local_f1,
                "local_auc": local_auc,
                "threshold": threshold,
            })

        # ── CAgg: Equal weight per district (intra-cluster) ───
        # This is what makes it cluster-aware:
        # regardless of district size, each gets equal weight
        # within the city cluster
        n_d           = len(weights_list)
        agg           = np.mean(weights_list, axis=0)  # equal weight
        self.total_n  = sum(n_list)
        self.n_districts = n_d
        self.agg_weights = agg

        print(f"\n  CAgg complete: {n_d} districts -> equal weight avg")
        print(f"  Total training samples in {self.city_name}: {self.total_n:,}")
        print(f"{'='*60}")

        return None, {}


# ── City Client (Client mode - forward to Tier 0) ─────────────
class CityClient(fl.client.NumPyClient):
    """
    Forwards city-aggregated weights to Tier 0 (Central).
    Sends total_n as the weight for Tier 0 aggregation.
    """
    def __init__(self, agg_weights, city_name, total_n, context, use_enc):
        self.agg_weights = agg_weights
        self.city_name   = city_name
        self.total_n     = total_n
        self.context     = context
        self.use_enc     = use_enc

    def fit(self, parameters, config):
        print(f"\n[TIER 1] {self.city_name}: Forwarding to Central "
              f"(n={self.total_n:,})")

        metadata = {
            "city"      : self.city_name,
            "total_n"   : self.total_n,
            "encrypted" : False,
        }

        if self.use_enc and self.context:
            arr       = self.agg_weights.astype(np.float64).tolist()
            enc_vec   = ts.ckks_vector(self.context, arr)
            enc_bytes = enc_vec.serialize()
            enc_array = np.frombuffer(enc_bytes, dtype=np.uint8)
            metadata["encrypted"] = True
            print(f"  Weights encrypted -> sending to Tier 0")
            return [enc_array], self.total_n, metadata
        else:
            print(f"  Sending aggregated weights (plaintext)")
            return [self.agg_weights], self.total_n, metadata

    def evaluate(self, parameters, config):
        return 0.0, self.total_n, {}


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Tier 1 — City Server (COVID 18-district)"
    )
    parser.add_argument("--port",          type=int, required=True,
                        help="Listen port for districts")
    parser.add_argument("--city",          type=str, required=True,
                        help="City name (Seoul, Gyeonggi, etc.)")
    parser.add_argument("--min-districts", type=int, default=2,
                        help="Min districts to wait for")
    parser.add_argument("--plaintext",     action="store_true",
                        help="No encryption (test mode)")
    args = parser.parse_args()

    use_enc = not args.plaintext
    ctx     = None

    if use_enc:
        pk_path = Path("keys/public.ctx")
        if not pk_path.exists():
            print("Error: public.ctx not found.")
            print("Use --plaintext for testing.")
            return
        with open(pk_path, "rb") as f:
            ctx = ts.context_from(f.read())

    print("\n" + "="*70)
    print(f"[TIER 1] CITY SERVER: {args.city.upper()}")
    print("="*70)
    print(f"  Port          : {args.port}")
    print(f"  Min districts : {args.min_districts}")
    print(f"  Encryption    : {'ENABLED' if use_enc else 'DISABLED'}")
    print(f"  CAgg alpha    : Equal weight per district (intra-cluster)")
    print(f"  Waiting for {args.min_districts}+ districts...")
    print("="*70)

    # Phase 1: Server — collect from districts
    strategy = CityStrategy(
        context=ctx,
        city_name=args.city,
        min_fit_clients=args.min_districts,
        min_available_clients=args.min_districts,
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{args.port}",
        config=fl.server.ServerConfig(num_rounds=1),
        strategy=strategy,
        grpc_max_message_length=1024*1024*1024,
    )

    # Phase 2: Client — forward to Tier 0
    if strategy.agg_weights is not None:
        print(f"\n[TIER 1] {args.city}: Forwarding to Central (Tier 0)...")
        time.sleep(2)

        fl.client.start_numpy_client(
            server_address="127.0.0.1:8080",
            client=CityClient(
                strategy.agg_weights,
                args.city,
                strategy.total_n,
                ctx,
                use_enc,
            ),
            grpc_max_message_length=1024*1024*1024,
        )

        print(f"\n[TIER 1] {args.city}: Mission complete!")
        print("="*70)
        print(f"  Districts aggregated : {strategy.n_districts}")
        print(f"  Total samples        : {strategy.total_n:,}")
        print(f"  CAgg method          : Equal weight per district")
        print("="*70)
    else:
        print(f"\n[TIER 1] {args.city}: No weights received from districts.")
        input("Press Enter to close...")


if __name__ == "__main__":
    main()
