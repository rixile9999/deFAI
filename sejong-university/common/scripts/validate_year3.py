import argparse
import gc
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from defai_addons.multimodal import data as MD
from defai_addons.multimodal.models import MultimodalTransformer
from defai_addons.multimodal.train import loader
from defai_addons.multimodal.federated import FedATC
from defai_addons.multimodal.transformer import MultimodalTransformerEncoder
from defai_addons.multimodal.encoders import SinusoidalPE


def sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_tokens(y):
    return torch.stack((3 + y % 3, 6 + y // 3, torch.full_like(y, 2)), 1)


def generation(seed, device, epochs):
    seed_all(seed)
    train = MD.generate(1800, seed=100 + seed)
    test = MD.generate(600, seed=10000 + seed)
    model = MultimodalTransformer(d=32, sequence_vocab=8, sequence_max_len=8, dropout=0.1).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.002)
    history = []
    start = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        losses = []
        for b in loader(train, 64, shuffle=True, seed=seed * 100 + epoch):
            b = MD.to_device(b, device)
            tgt = target_tokens(b['y'])
            inp = torch.cat((torch.ones(len(tgt), 1, dtype=torch.long, device=device), tgt[:, :-1]), 1)
            out = model(b, inp)
            loss = F.cross_entropy(out['sequence_logits'].reshape(-1, 8), tgt.reshape(-1))
            loss = loss + F.cross_entropy(out['logits'], b['y']) + out['aux']
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
        if (epoch + 1) % 5 == 0:
            print('generation seed=%d epoch=%d loss=%.4f' % (seed, epoch + 1, history[-1]), flush=True)
    model.eval()
    n, correct, exact, token_correct, eos_ok = 0, 0, 0, 0, 0
    with torch.no_grad():
        for b in loader(test, 64, shuffle=False):
            b = MD.to_device(b, device)
            out = model(b)
            tgt = target_tokens(b['y'])
            gen = model.generate(b, 1, 2, max_new_tokens=3)
            padded = torch.zeros_like(tgt)
            padded[:, :gen.size(1)] = gen
            n += len(tgt)
            correct += int(out['logits'].argmax(1).eq(b['y']).sum())
            exact += int(padded.eq(tgt).all(1).sum())
            token_correct += int(padded[:, :2].eq(tgt[:, :2]).sum())
            eos_ok += int(padded[:, 2].eq(2).sum())
    majority = int(np.bincount(train['y'], minlength=6).argmax())
    sync(device)
    return dict(seed=seed, n_train=1800, n_test=n, epochs=epochs,
                target='shape_token, frequency_band_token, EOS; not natural language',
                sequence_exact_match=exact/n, semantic_token_accuracy=token_correct/(2*n),
                eos_at_third_rate=eos_ok/n, classification_accuracy=correct/n,
                train_majority_baseline=float(np.mean(test['y'] == majority)),
                train_loss=history, elapsed_s=time.perf_counter()-start)


class RecallModel(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.emb = nn.Embedding(4, 32)
        self.pe = SinusoidalPE(32, length)
        self.enc = MultimodalTransformerEncoder(32, n_layers=1, n_heads=4, dropout=0.0)
        self.head = nn.Linear(32, 4)

    def forward(self, ids):
        h = self.pe(self.emb(ids))
        return self.head(self.enc(h, torch.zeros_like(ids, dtype=torch.bool))[:, -1])


def long_context(length, device, curriculum=False):
    seed_all(31 + length)
    model = RecallModel(length).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.003)
    rng = torch.Generator().manual_seed(80 + length)
    start = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    train_lengths = [128, length] if curriculum and length > 128 else [length]
    for train_length in train_lengths:
        for _ in range(350):
            ids = torch.randint(0, 4, (32, train_length), generator=rng).to(device)
            loss = F.cross_entropy(model(ids), ids[:, 0])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    good, perturbed_good, count = 0, 0, 0
    test_rng = torch.Generator().manual_seed((190000 if curriculum else 90000) + length)
    with torch.no_grad():
        for _ in range(16):
            ids = torch.randint(0, 4, (32, length), generator=test_rng).to(device)
            pred = model(ids).argmax(1)
            good += int(pred.eq(ids[:, 0]).sum())
            original = ids[:, 0].clone()
            ids[:, 0] = (ids[:, 0] + 1) % 4
            perturbed_good += int(model(ids).argmax(1).eq(original).sum())
            count += len(ids)
    sync(device)
    return dict(tokens=length, updates=350*len(train_lengths), training_lengths=train_lengths,
                test_samples=count, first_token_recall_accuracy=good/count,
                first_token_changed_accuracy_on_original_labels=perturbed_good/count, chance=0.25,
                elapsed_s=time.perf_counter()-start,
                peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
                scope='same_encoder_component_synthetic_recall_not_real_multimodal_long_context')


def scale(n_clients, compressed, device):
    seed_all(7)
    clients = MD.make_clients(n_clients, 32, seed=7)
    test = MD.generate(96, seed=17007)
    sync(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    fl = FedATC(lambda: MultimodalTransformer(d=16), clients, device=device, local_epochs=1,
                batch_size=32, hierarchical=True, update_compression='topk' if compressed else None,
                analysis_compression=False, seed=7, log=None)
    result = fl.run(2, test)
    sync(device)
    record = dict(clients=n_clients, samples_per_client=32, rounds=2, compressed=compressed,
                  elapsed_s=time.perf_counter()-start, rss_mb=psutil.Process(os.getpid()).memory_info().rss/2**20,
                  peak_gpu_mb=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
                  server_in_bytes=result['bytes']['server_in'], client_group_bytes=result['bytes']['client_group'],
                  accuracy=result['global_full']['accuracy'], compression_ratio=result['update_compression_ratio'],
                  scope='sequential_clients_single_host_flower_simulation_not_physical_multinode')
    del fl
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/year3_validation')
    ap.add_argument('--part', choices=['all', 'generation', 'context', 'scale'], default='all')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--context-curriculum', action='store_true', help='128 to 512 token curriculum; fresh held-out data')
    args = ap.parse_args()
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(device=device, torch=torch.__version__, python=platform.python_version(),
                  gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None,
                  generation=[], context=[], scale=[])
    def save():
        suffix = '_curriculum' if args.context_curriculum else ''
        (out / (args.part + suffix + '.json')).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for part in ['generation', 'context', 'scale']:
        if args.part not in ['all', part]:
            continue
        if part == 'generation':
            for seed in range(3):
                r = generation(seed, device, args.epochs)
                report[part].append(r)
                print(json.dumps({part: r}, ensure_ascii=False), flush=True)
                save()
        elif part == 'context':
            for length in [128, 512]:
                r = long_context(length, device, args.context_curriculum)
                report[part].append(r)
                print(json.dumps({part: r}), flush=True)
                save()
        else:
            for n in [9, 30, 90]:
                for compressed in [False, True]:
                    r = scale(n, compressed, device)
                    report[part].append(r)
                    print(json.dumps({part: r}), flush=True)
                    save()


if __name__ == '__main__':
    main()
