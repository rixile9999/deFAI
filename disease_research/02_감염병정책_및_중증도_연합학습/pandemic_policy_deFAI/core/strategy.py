import flwr as fl
import numpy as np
import tenseal as ts
from .serialization import unpack_payload


class FCausalStrategy(fl.server.strategy.FedAvg):
    def __init__(self, context=None, **kwargs):
        super().__init__(**kwargs)
        self.context = context

    def aggregate_fit(self, server_round, results, failures):
        if not results: return None, {}

        sum_xtx, sum_xty = None, None
        is_encrypted = False

        print(f"\n[Round {server_round}] Aggregating results from {len(results)} clients...")

        for _, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            is_encrypted = fit_res.metrics.get("encrypted", False)

            if is_encrypted:
                # [Phase 3] 암호문 덧셈 (Homomorphic Addition)
                xtx_vec = ts.ckks_vector_from(self.context, params[0].tobytes())
                xty_vec = ts.ckks_vector_from(self.context, params[1].tobytes())

                if sum_xtx is None:
                    sum_xtx, sum_xty = xtx_vec, xty_vec
                else:
                    sum_xtx += xtx_vec
                    sum_xty += xty_vec
            else:
                # [Phase 2] 평문 덧셈
                payload = unpack_payload(params)
                if sum_xtx is None:
                    sum_xtx, sum_xty = payload['xtx'], payload['xty']
                else:
                    sum_xtx += payload['xtx']
                    sum_xty += payload['xty']

        # 복호화 및 결과 도출
        if is_encrypted:
            print("   🔓 [Server] Decrypting aggregated statistics...")
            xtx_final = np.array(sum_xtx.decrypt())
            xty_final = np.array(sum_xty.decrypt())

            # Shape 복원 (4x4 Matrix)
            dim = int(np.sqrt(len(xtx_final)))
            xtx_final = xtx_final.reshape(dim, dim)
        else:
            xtx_final, xty_final = sum_xtx, sum_xty

        # Ridge Regression Solver
        try:
            beta = np.linalg.solve(xtx_final + 1e-5 * np.eye(xtx_final.shape[0]), xty_final)
            self._print_results(beta)
        except Exception as e:
            print(f"   ⚠️ Solver Error: {e}")

        return None, {}

    def _print_results(self, beta):
        names = ['Intercept', 'Post_Time', 'Treated_Region', 'DiD_Effect']
        print(f"   📊 Global Coefficients:")
        for n, v in zip(names, beta):
            print(f"      {n:15s}: {v:+.4f}")

        # DiD 계수(beta[3])가 음수면 정책 효과 있음
        if beta[3] < -0.1:
            print("   ✅ SUCCESS: Significant Policy Effect Detected (Cases Reduced)")
        else:
            print(f"   ℹ️ Result: DiD Effect is {beta[3]:.4f} (Neutral/Positive)")