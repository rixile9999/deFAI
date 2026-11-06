import argparse
import json
from .integration.history import read_history, render_histories, write_new


def main(argv=None):
    parser = argparse.ArgumentParser(prog='defai-addons')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('history', help='Normalize a host history.json without changing it')
    p.add_argument('input'); p.add_argument('--out', required=True)
    p = sub.add_parser('visualize', help='Render completed host runs into standalone HTML')
    p.add_argument('inputs', nargs='+'); p.add_argument('--out', required=True)
    p = sub.add_parser('dashboard', help='Start the native hierarchy evaluation module')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=5055)
    p.add_argument('--store', default='addon-runs')
    p = sub.add_parser('compare', help='Run independent comparison simulations or supplied site arrays')
    p.add_argument('--config', required=True); p.add_argument('--data'); p.add_argument('--out', required=True)
    p = sub.add_parser('multimodal-predict', help='Predict using a saved trained checkpoint and NPZ tensors')
    p.add_argument('--checkpoint', required=True); p.add_argument('--data', required=True); p.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from pathlib import Path
    if getattr(args, 'out', None) and Path(args.out).exists():
        parser.error('Output already exists; choose a new path')
    if args.command == 'history':
        result = read_history(args.input)
    elif args.command == 'visualize':
        render_histories(args.inputs, args.out)
        print(args.out)
        return
    elif args.command == 'dashboard':
        from .api import create_dashboard
        create_dashboard(args.store).run(host=args.host, port=args.port, debug=False)
        return
    elif args.command == 'compare':
        from .api import compare, compare_arrays
        config = json.loads(Path(args.config).read_text(encoding='utf-8-sig'))
        if args.data:
            import numpy as np
            with np.load(args.data, allow_pickle=False) as data:
                count = int(data['n_sites'].item())
                sites = [{k: data[f'site_{i}_{k}'] for k in ('X', 'y', 'X_test', 'y_test')} for i in range(count)]
                result = compare_arrays(sites, data['X_test'], data['y_test'], config)
        else:
            result = compare(config)
    else:
        import numpy as np
        from .api import load_multimodal_checkpoint, predict_multimodal
        with np.load(args.data, allow_pickle=False) as data:
            result = predict_multimodal(load_multimodal_checkpoint(args.checkpoint), data)
        result = {k: v.tolist() for k, v in result.items()}
    write_new(args.out, json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    print(args.out)


if __name__ == '__main__':
    main()
