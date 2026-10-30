"""Compatibility entry point for the packaged round-safe mobile coordinator."""
import argparse
from pathlib import Path
from defai.cli import main as defai_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path.cwd() / 'saved_models')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8001)
    parser.add_argument('--clients', type=int, default=1)
    args = parser.parse_args()
    return defai_main(['mobile', '--output', str(args.output), '--host', args.host,
                       '--port', str(args.port), '--clients', str(args.clients)])


if __name__ == '__main__':
    raise SystemExit(main())
