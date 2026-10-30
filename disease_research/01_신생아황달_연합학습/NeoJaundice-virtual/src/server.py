import flwr as fl

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
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,           # 모든 클라이언트 학습 참여
        min_fit_clients=2,          # 최소 2개 연결 시 시작
        min_available_clients=2,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    print("Starting Flower Server...")
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )