"""Run existing experiments on CUDA with an explicit, recorded memory budget."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import runpy
import sys
import time

import torch

import learning
from cohort import ROOT


def check_cuda():
    """Check local update agreement and time representative batches, not full runs."""
    generator = torch.Generator().manual_seed(20261022)
    x = torch.randn(32, 5, 20, generator=generator)
    y = torch.randn(32, 5, generator=generator)
    mask = torch.ones(32, 5)
    mask[:8, 1:] = 0
    checks = []
    for method in ('fedavg', 'fedprox', 'scaffold', 'fedavg_bn', 'fedbn'):
        bn = method in ('fedavg_bn', 'fedbn')
        base = learning.initial_parameters(20261022, bn=bn, device='cpu')
        def update(device):
            copies = [p.to(device)[None].expand(32, *p.shape).clone() for p in base]
            state = [torch.zeros(32, 64, device=device), torch.ones(32, 64, device=device)] if bn else None
            control = [torch.full_like(p, .001) for p in copies] if method == 'scaffold' else None
            return learning.local_fit(x.to(device), y.to(device), mask.to(device), copies,
                                      5, .001, .001 if method == 'fedprox' else 0., control, state)
        cpu = update('cpu')
        gpu = update('cuda')
        difference = 0.
        for left, right in zip(cpu[0], gpu[0]):
            torch.testing.assert_close(left, right.cpu(), rtol=1e-4, atol=2e-6)
            difference = max(difference, float((left-right.cpu()).abs().max()))
        if bn:
            for left, right in zip(cpu[1], gpu[1]):
                torch.testing.assert_close(left, right.cpu(), rtol=1e-4, atol=2e-6)
        checks.append({'method': method, 'passed': True, 'max_parameter_absolute_difference': difference})
    # One independent parameter copy for each of 2,000 selected people.
    bx = torch.randn(2000, 5, 20, device='cuda')
    by = torch.randn(2000, 5, device='cuda')
    bm = torch.ones(2000, 5, device='cuda')
    base = learning.initial_parameters(20261022)
    copies = [p[None].expand(2000, *p.shape).clone() for p in base]
    for _ in range(3):
        learning.local_fit(bx, by, bm, copies, 5, .001)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(20):
        learning.local_fit(bx, by, bm, copies, 5, .001)
    torch.cuda.synchronize()
    return {'checks': checks, 'benchmark': {
        'selected_people': 2000, 'local_samples_each': 5, 'local_steps': 5,
        'repetitions': 20, 'mean_batch_seconds': (time.perf_counter()-start)/20,
        'peak_tensor_memory_bytes': torch.cuda.max_memory_allocated(),
        'scope': 'local_fit only; excludes cohort generation, transfers, aggregation, evaluation and persistent client state'}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fraction', type=float, default=.60)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error('--fraction must be in (0, 1]')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    torch.cuda.set_per_process_memory_fraction(args.fraction)
    free, total = torch.cuda.mem_get_info()
    metadata = {'timestamp_local': datetime.now().isoformat(), 'device': torch.cuda.get_device_name(),
                'torch_version': torch.__version__, 'memory_fraction': args.fraction,
                'memory_budget_bytes': int(total*args.fraction), 'free_bytes_at_start': free,
                'research_stage': 'simulation; no clinical validation'}
    if args.check:
        metadata.update(check_cuda())
    folder = ROOT/'benchmarks/gpu_execution'
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/f'gpu_{datetime.now():%Y%m%d_%H%M%S}.json'
    path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print(f'GPU record: {path}', flush=True)
    command = args.command
    if command and command[0] == '--':
        command = command[1:]
    if command:
        script = Path(command[0]).resolve(strict=True)
        sys.argv = [str(script), *command[1:]]
        runpy.run_path(str(script), run_name='__main__')
    elif not args.check:
        parser.error('provide --check or -- script.py [arguments]')


if __name__ == '__main__':
    main()
