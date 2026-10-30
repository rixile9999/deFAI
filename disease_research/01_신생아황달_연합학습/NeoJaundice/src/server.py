import flwr as fl
import sys
from pathlib import Path
from collections import OrderedDict
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model import MultiModalNet
from src.federated import average_states


class NeonatalFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        if failures or len(results) != self.min_fit_clients:
            raise ValueError("Incomplete client cohort")
        reference = MultiModalNet(pretrained=False).state_dict()
        states, weights = [], []
        for _, result in results:
            arrays = fl.common.parameters_to_ndarrays(result.parameters)
            if len(arrays) != len(reference):
                raise ValueError("Model payload length mismatch")
            states.append(OrderedDict((name, torch.as_tensor(array, dtype=value.dtype))
                                      for (name, value), array in zip(reference.items(), arrays)))
            weights.append(result.num_examples)
        averaged = average_states(states, weights)
        return fl.common.ndarrays_to_parameters([v.numpy() for v in averaged.values()]), {}

def weighted_average(metrics):
    # 각 클라이언트의 메트릭을 데이터 수에 비례하여 가중 평균
    accuracies = [num * m["accuracy"] for num, m in metrics]
    maes = [num * m["mae"] for num, m in metrics]
    examples = [num for num, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "mae": sum(maes) / sum(examples),
    }

if __name__ == "__main__":
    torch.manual_seed(42)
    initial = MultiModalNet()
    strategy = NeonatalFedAvg(
        initial_parameters=fl.common.ndarrays_to_parameters([v.numpy() for v in initial.state_dict().values()]),
        accept_failures=False,
        fraction_fit=1.0,           # 모든 클라이언트 학습 참여
        min_fit_clients=3,
        min_available_clients=3,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )
