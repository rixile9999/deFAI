from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing as mp
from pathlib import Path
import socket
import struct
from types import SimpleNamespace

import numpy as np

RPC_HEADER = struct.Struct('!IQ')
MAX_BYTES = 128 * 1024 * 1024


def _read_exact(sock, count):
    pieces = []
    while count:
        part = sock.recv(min(count, 1024 * 1024))
        if not part:
            raise EOFError('Incomplete verification RPC')
        pieces.append(part)
        count -= len(part)
    return b''.join(pieces)


def _send(sock, metadata, payload=b''):
    encoded = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    if len(encoded) > 1024 * 1024 or len(payload) > MAX_BYTES:
        raise ValueError('Verification RPC exceeds declared bounds')
    frame = RPC_HEADER.pack(len(encoded), len(payload)) + encoded + payload
    sock.sendall(frame)
    return len(frame)


def _receive(sock):
    metadata_size, payload_size = RPC_HEADER.unpack(_read_exact(sock, RPC_HEADER.size))
    if metadata_size > 1024 * 1024 or payload_size > MAX_BYTES:
        raise ValueError('Invalid verification RPC extent')
    metadata = json.loads(_read_exact(sock, metadata_size))
    payload = _read_exact(sock, payload_size)
    return metadata, payload, RPC_HEADER.size + metadata_size + payload_size


def _array_bytes(arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **{f'array_{i}': a for i, a in enumerate(arrays)})
    return buffer.getvalue()


def _worker(ready, codec_name):
    import os
    import torch
    from threadpoolctl import threadpool_limits
    from defai_research.compression import CompactLosslessCodec, TransformerCompactCodec
    torch.set_num_threads(1)
    up, down = CompactLosslessCodec(), CompactLosslessCodec()
    endpoints = {}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        listener.settimeout(30)
        ready.send(dict(port=listener.getsockname()[1], pid=os.getpid()))
        ready.close()
        connection, _ = listener.accept()
        with connection, threadpool_limits(limits=1):
            connection.settimeout(30)
            while True:
                metadata, payload, _ = _receive(connection)
                command = metadata['command']
                try:
                    if command == 'stop':
                        _send(connection, {'ok': True})
                        return
                    if command == 'deploy':
                        restored, _, destination = TransformerCompactCodec._parse_deployment(payload)
                        if ('central', 0) not in endpoints:
                            endpoints[('central', 0)] = TransformerCompactCodec(restored)
                        if destination[0] == 'group':
                            endpoints[destination] = TransformerCompactCodec.receiver_from_deployment(payload)
                        _send(connection, dict(ok=True, deployment_bytes=len(payload), endpoint=list(destination)))
                        continue
                    channel, rnd = tuple(metadata['channel']), metadata['round']
                    if command == 'down':
                        arrays = down.decode_frame(payload, expected_channel=channel, expected_round=rnd).arrays
                        audit = {}
                    elif command == 'up' and codec_name == 'compact':
                        arrays = up.decode_frame(payload, expected_channel=channel, expected_round=rnd).arrays
                        audit = {}
                    elif command == 'up' and codec_name == 'transformer':
                        endpoint = ('group', channel[3]) if channel[0] == 'edge' else ('central', 0)
                        arrays, audit = endpoints[endpoint].decode_frame(payload, channel=channel, round_id=rnd)
                    else:
                        raise ValueError('Unknown verification RPC command')
                    _send(connection, dict(ok=True, decoder_audit=audit), _array_bytes(arrays))
                except Exception as exc:
                    _send(connection, dict(ok=False, error=f'{type(exc).__name__}: {exc}'))


class _Remote:
    def __init__(self, rpc, direction, transformer=False):
        self.rpc, self.direction, self.transformer = rpc, direction, transformer

    def decode_frame(self, frame, *, channel=None, round_id=None, expected_channel=None, expected_round=None):
        channel = channel if channel is not None else expected_channel
        rnd = round_id if round_id is not None else expected_round
        metadata, payload = self.rpc.request(self.direction, frame, channel=list(channel), round=rnd)
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            arrays = [source[f'array_{i}'].copy() for i in range(len(source.files))]
        if self.transformer:
            return arrays, metadata['decoder_audit']
        return SimpleNamespace(arrays=arrays)


class _RPC:
    def __init__(self, codec_name):
        context = mp.get_context('spawn')
        parent, child = context.Pipe(duplex=False)
        self.process = context.Process(target=_worker, args=(child, codec_name), daemon=True)
        self.process.start()
        child.close()
        if not parent.poll(30):
            self.process.terminate()
            self.process.join(5)
            raise TimeoutError('Decoder worker did not become ready')
        self.worker = parent.recv()
        parent.close()
        self.connection = socket.create_connection(('127.0.0.1', self.worker['port']), timeout=30)
        self.counts = dict(request_bytes=0, response_bytes=0, encoded_model_frame_bytes=0,
                           deployment_payload_bytes=0, returned_array_payload_bytes=0,
                           model_frames=0, deployment_packets=0)

    def request(self, command, payload=b'', **metadata):
        self.counts['request_bytes'] += _send(self.connection, dict(command=command, **metadata), payload)
        result, received, count = _receive(self.connection)
        self.counts['response_bytes'] += count
        if not result.get('ok'):
            raise RuntimeError(result.get('error', 'Decoder worker failed'))
        if command == 'deploy':
            self.counts['deployment_payload_bytes'] += len(payload)
            self.counts['deployment_packets'] += 1
        elif command in {'up', 'down'}:
            self.counts['encoded_model_frame_bytes'] += len(payload)
            self.counts['returned_array_payload_bytes'] += len(received)
            self.counts['model_frames'] += 1
        return result, received

    def close(self):
        try:
            self.request('stop')
        finally:
            self.connection.close()
            self.process.join(10)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
        if self.process.exitcode != 0:
            raise RuntimeError('Decoder worker failed to stop cleanly')


def _compare(local, remote):
    checks = {}
    for name in ('selected_model.npz', 'final_model.npz', 'retained_masks.npz', 'validation_predictions.npz'):
        with np.load(local/name, allow_pickle=False) as a, np.load(remote/name, allow_pickle=False) as b:
            checks[name] = a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files)
    left, right = [json.loads((folder/'final.json').read_text(encoding='utf-8')) for folder in (local, remote)]
    for key in ('sum_total_bytes', 'sum_up_bytes', 'sum_down_bytes', 'sum_graph_control_both_hops_bytes',
                'sum_codec_deployment_bytes', 'sum_examples_processed', 'sum_optimizer_steps',
                'selected_round', 'selected_val_macro_f1', 'post_drift_dedup_recall'):
        checks[key] = left[key] == right[key]
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=8)
    parser.add_argument('--seed', type=int, default=19001)
    parser.add_argument('--datasets', choices=('digits', 'breast_cancer'), nargs='+', default=['digits', 'breast_cancer'])
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('--rounds must be at least 2')
    if not args.artifact.is_file():
        parser.error('--artifact must name an existing trained inference artifact')
    args.output.mkdir(parents=True, exist_ok=False)
    import defai_research.compression.runner as study
    from defai_research.compression import TransformerCompactCodec
    original, runs = study.WireSession, []
    for dataset in args.datasets:
        for codec in ('compact', 'transformer'):
            stem = args.output/f'{dataset}_{codec}'
            local, remote = stem/'local', stem/'tcp'
            kwargs = dict(dataset=dataset, seed=args.seed, rounds=args.rounds,
                          drift_round=args.rounds//2+1, codec=codec,
                          artifact=args.artifact if codec=='transformer' else None, allow_test=False)
            study.run_benchmark(output=local, **kwargs)
            created = []

            class SocketWire(original):
                def __init__(self, codec, artifact=None):
                    super().__init__(codec, artifact)
                    self.rpc = _RPC(codec)
                    created.append(self)
                    self.down_receiver = _Remote(self.rpc, 'down')
                    if codec == 'compact':
                        self.up_receiver = _Remote(self.rpc, 'up')
                    else:
                        deployment = TransformerCompactCodec(artifact).deployment_packets()
                        for packet, recorded in zip(deployment, self.deployment):
                            if hashlib.sha256(packet['payload']).hexdigest() != recorded['payload_sha256']:
                                raise AssertionError('Verification deployment differs from accounted packet')
                            self.rpc.request('deploy', packet['payload'])
                        self.receivers = {endpoint: _Remote(self.rpc, 'up', transformer=True)
                                          for endpoint in self.receivers}

            study.WireSession = SocketWire
            try:
                result = study.run_benchmark(output=remote, **kwargs)
            finally:
                study.WireSession = original
                for instance in created:
                    instance.rpc.close()
            checks = _compare(local, remote)
            rpc = created[0].rpc
            if rpc.counts['encoded_model_frame_bytes'] != result['sum_model_payload_bytes']:
                raise AssertionError('Actual socket model payload differs from scientific frame accounting')
            if rpc.counts['deployment_payload_bytes'] != result['sum_codec_deployment_bytes']:
                raise AssertionError('Actual socket deployment differs from scientific deployment accounting')
            row = dict(dataset=dataset, codec=codec, passed=all(checks.values()), checks=checks,
                       worker=rpc.worker, rpc_counts=rpc.counts,
                       decoded_socket_responses_used_by_flower=True,
                       worker_receives_no_artifact_path_or_raw_updates=True)
            runs.append(row)
            print(f'{dataset}/{codec}: {"PASS" if row["passed"] else "FAIL"}', flush=True)
    report = dict(passed=all(row['passed'] for row in runs), runs=runs,
                  scope='Localhost correctness and bit-exact comparison; no WAN/throughput claim',
                  verification_return_arrays_and_rpc_overhead_excluded_from_scientific_traffic=True)
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
