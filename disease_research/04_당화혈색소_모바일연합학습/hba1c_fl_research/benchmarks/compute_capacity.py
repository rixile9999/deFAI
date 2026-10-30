"""Measure isolated per-client training kernels; not a clinical/FL benchmark."""
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil

N, SAMPLES, EPOCHS = 200_000, 8, 5
rng = np.random.default_rng(20261016)
x = rng.standard_normal((N, SAMPLES, 20), dtype=np.float32)
y = (0.2 * x[:, :, 0] + 0.1 * x[:, :, 1]).astype(np.float32)


def local_linear(features, targets, epochs=EPOCHS):
    weights = np.zeros((len(features), features.shape[2]), dtype=np.float32)
    bias = np.zeros(len(features), dtype=np.float32)
    for _ in range(epochs):
        errors = np.einsum('cnd,cd->cn', features, weights) + bias[:, None] - targets
        weights -= (0.02 / features.shape[1]) * np.einsum('cnd,cn->cd', features, errors)
        bias -= 0.02 * errors.mean(axis=1)
    return weights, bias


# Verify independent local updates agree with fitting clients one at a time.
test_w, test_b = local_linear(x[:3], y[:3])
for index in range(3):
    one_w, one_b = local_linear(x[index:index + 1], y[index:index + 1])
    np.testing.assert_allclose(test_w[index], one_w[0], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(test_b[index], one_b[0], rtol=1e-5, atol=1e-7)

results = {
    'scope': 'Random synthetic arrays, independent local full-batch SGD, then averaging. '
             'No clinical simulation, behavior generation, full evaluation, persistence, '
             'Flower runtime, cryptography, or transport.',
    'population': N, 'records_per_client': SAMPLES, 'local_epochs': EPOCHS,
    'dtype': 'float32', 'python': platform.python_version(), 'numpy': np.__version__,
    'data_array_bytes': x.nbytes + y.nbytes,
    'independent_update_equivalence': 'passed for 3 clients', 'measurements': [],
}
for dimensions in (2, 20):
    for clients in (200, 2000, 10000, 200000):
        durations = []
        for repeat in range(4):
            start = time.perf_counter()
            selected = rng.choice(N, clients, replace=False)
            weights, bias = local_linear(x[selected, :, :dimensions], y[selected])
            averaged = np.r_[weights.mean(axis=0), bias.mean()]
            assert np.isfinite(averaged).all()
            elapsed = time.perf_counter() - start
            if repeat:
                durations.append(elapsed)
        results['measurements'].append({
            'engine': 'numpy_cpu', 'parameters': dimensions + 1,
            'clients_per_round': clients, 'median_round_seconds': float(np.median(durations)),
            'measured_round_seconds': durations,
        })

try:
    import torch
    results['torch'] = torch.__version__
    results['cuda_available'] = torch.cuda.is_available()
    if torch.cuda.is_available():
        results['gpu'] = torch.cuda.get_device_name(0)
        torch.manual_seed(20261016)
        base = [torch.randn(20, 64, device='cuda') * 0.01,
                torch.zeros(64, device='cuda'),
                torch.randn(64, 1, device='cuda') * 0.01,
                torch.zeros(1, device='cuda')]

        def mlp_round(clients):
            selected = rng.choice(N, clients, replace=False)
            summed = [torch.zeros_like(p) for p in base]
            for offset in range(0, clients, 512):
                ids = selected[offset:offset + 512]
                xx = torch.from_numpy(x[ids]).to('cuda')
                yy = torch.from_numpy(y[ids]).to('cuda')
                params = [p.unsqueeze(0).expand(len(ids), *p.shape).clone().requires_grad_()
                          for p in base]
                for _ in range(EPOCHS):
                    w1, b1, w2, b2 = params
                    hidden = torch.relu(torch.bmm(xx, w1) + b1[:, None, :])
                    prediction = (torch.bmm(hidden, w2) + b2[:, None, :]).squeeze(-1)
                    # Sum of per-client mean losses keeps each client's update independent.
                    loss = ((prediction - yy) ** 2).mean(dim=1).sum()
                    gradients = torch.autograd.grad(loss, params)
                    params = [(p - 0.01 * g).detach().requires_grad_()
                              for p, g in zip(params, gradients)]
                for total, p in zip(summed, params):
                    total.add_(p.detach().sum(dim=0))
            return [p / clients for p in summed]

        for clients in (200, 2000, 10000):
            durations = []
            for repeat in range(4):
                torch.cuda.synchronize()
                start = time.perf_counter()
                aggregated = mlp_round(clients)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if repeat:
                    durations.append(elapsed)
            results['measurements'].append({
                'engine': 'torch_cuda_batched', 'architecture': '20-64-1 ReLU',
                'parameters': 1409, 'clients_per_round': clients,
                'median_round_seconds': float(np.median(durations)),
                'measured_round_seconds': durations,
                'includes_cpu_to_gpu_transfer': True,
            })
        results['cuda_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
except ImportError as error:
    results['torch_unavailable'] = str(error)

memory = psutil.Process().memory_info()
results['process_memory_bytes'] = memory.rss
results['process_peak_working_set_bytes'] = getattr(memory, 'peak_wset', None)
path = Path(__file__).with_name('compute_capacity_results.json')
path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
