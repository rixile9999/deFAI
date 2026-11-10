import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--host-output', required=True, help='New directory for the original deFAI run')
    p.add_argument('--view-output', required=True, help='New HTML file outside the host repository')
    p.add_argument('--strategy', choices=['fedavg', 'fedprox', 'qfedavg'], default='fedavg')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--clients', type=int, default=4)
    args = p.parse_args()
    if Path(args.view_output).exists():
        p.error('--view-output must not exist')
    from defai.simulation import run_simulation
    from defai_addons.api import render_histories
    run_simulation(args.host_output, clients=args.clients, device='cpu',
                   overrides={'dataset': 'synthetic', 'strategy': args.strategy,
                              'num-server-rounds': args.rounds})
    render_histories([Path(args.host_output)/'history.json'], args.view_output)
    print(args.view_output)


if __name__ == '__main__':
    main()
