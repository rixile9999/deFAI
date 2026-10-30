"""Small, entirely synthetic execution of all public workflow families."""
from pathlib import Path
import time


def run_demo(output):
    import numpy as np
    import torch
    from scipy.stats import burr12
    from defai.io import write_json
    from defai.mobile import create_app
    from defai.research.generation import build_hierarchy, federated_fit, pooled_reference
    from defai.research.hba1c import initial_parameters, local_fit
    from defai.research.neonatal import MultiModalNet, average_states
    from defai.research.policy import AnalysisConfig, analyze, district_vector, province_summary
    from defai.simulation import run_simulation

    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Demo output already exists: {output}")
    output.mkdir(parents=True)
    start = time.perf_counter()
    old_threads = torch.get_num_threads()
    result = {"research_only": True, "data": "synthetic_only", "clinical_validation": False}
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model = MultiModalNet(pretrained=False)
            model.train()
            image, metadata = torch.rand(4, 3, 64, 64), torch.rand(4, 4)
            before = {k: v.detach().clone() for k, v in model.state_dict().items()}
            regression, classification = model(image, metadata)
            loss = ((regression-.4)**2).mean() + torch.nn.functional.binary_cross_entropy_with_logits(classification, torch.zeros_like(classification))
            loss.backward()
            with torch.no_grad():
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.add_(parameter.grad, alpha=-.001)
            aggregate = average_states([before, model.state_dict()], [1, 1])
            result["neonatal"] = {"local_loss": float(loss.detach()), "parameters": len(aggregate),
                                  "finite": all(torch.isfinite(v).all().item() for v in aggregate.values())}
            initial = initial_parameters(42, hidden=8, device="cpu")
            x, y, mask = torch.rand(3, 6, 20), torch.rand(3, 6), torch.ones(3, 6)
            states, _, losses = local_fit(x, y, mask, [p[None].expand(3, *p.shape).clone() for p in initial], 3, .001)
            result["hba1c"] = {"clients": 3, "initial_mean_squared_loss": float(losses.mean()),
                               "finite": all(torch.isfinite(p).all().item() for p in states)}
        config = AnalysisConfig()
        rng = np.random.default_rng(42)
        provinces = []
        for region in range(6):
            vectors = []
            treated = int(region >= 3)
            for _ in range(2):
                counts = rng.poisson(8, len(config.calendar))
                if treated:
                    counts[config.masks[-1]] //= 2
                vectors.append(district_vector(counts, config))
            provinces.append(province_summary(np.sum(vectors, axis=0), str(region), treated, config, 2))
        result["policy"] = analyze(provinces)
        samples = burr12.rvs(1.7, 1.5, scale=5.6, size=3000, random_state=rng)
        _, nodes = build_hierarchy({11001+i: p for i, p in enumerate(np.array_split(samples, 3))})
        parameters, nll, rounds, fit = federated_fit(nodes)
        _, pooled_nll = pooled_reference(samples)
        result["generation"] = {"parameters": [float(v) for v in parameters], "rounds": rounds,
                                "optimizer_success": bool(fit.success),
                                "nll_difference_per_observation": abs(float(nll-pooled_nll))/len(samples)}
        app = create_app(output / "mobile", expected_clients=2)
        client = app.test_client()
        payload = client.get("/get_parameters").get_json()
        first = client.post("/submit_parameters", json={**payload, "client_id": "first"})
        second = client.post("/submit_parameters", json={**payload, "client_id": "second"})
        if first.status_code != 200 or second.status_code != 200:
            raise RuntimeError("Mobile coordinator smoke failed")
        result["mobile"] = second.get_json()
        common = run_simulation(output / "common", clients=2, overrides={
            "dataset": "synthetic", "num-server-rounds": 1, "fraction-train": 1., "fraction-evaluate": 1.})
        result["common"] = {"history": common["history"], "backend": common["backend"]}
        result["elapsed_seconds"] = time.perf_counter()-start
        if not result["neonatal"]["finite"] or not result["hba1c"]["finite"] or not fit.success:
            raise RuntimeError("A synthetic workflow failed numerical validation")
        write_json(output / "demo.json", result, overwrite=False)
        return result
    finally:
        torch.set_num_threads(old_threads)
