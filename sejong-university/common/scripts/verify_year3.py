import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F

from defai_addons.compare.runner import run_comparison
from defai_addons.multimodal import data as MD
from defai_addons.multimodal.models import MultimodalTransformer
from defai_addons.multimodal.train import loader
from run_comparison import to_markdown


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/year3_verification')
    args = parser.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    comparison = run_comparison(dict(dataset='digits', n_sites=5, max_train=1000, epochs=2,
                                    hidden=[16], use_pca=False, encrypt_updates=True))
    (outdir / 'comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')
    (outdir / 'comparison.md').write_text(to_markdown(comparison), encoding='utf-8')

    torch.manual_seed(11)
    data = MD.generate(32, seed=11)
    batch = next(iter(loader(data, 32, shuffle=False)))
    labels = torch.as_tensor(data['y'])
    targets = torch.stack((3 + labels % 3, 6 + labels // 3, torch.full_like(labels, 2)), 1)
    inputs = torch.cat((torch.ones(32, 1, dtype=torch.long), targets[:, :-1]), 1)
    model = MultimodalTransformer(d=16, sequence_vocab=8, sequence_max_len=8, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    losses = []
    model.train()
    for _ in range(20):
        pred = model(batch, inputs)['sequence_logits']
        loss = F.cross_entropy(pred.reshape(-1, 8), targets.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    with torch.no_grad():
        outputs = model(batch, inputs)
        generated = model.generate(batch, bos_id=1, eos_id=2, max_new_tokens=3)
    report = dict(scope='synthetic_training_smoke_test_not_generalization_or_scale_evaluation',
                  samples=32, optimization_steps=20, initial_loss=losses[0], final_loss=losses[-1],
                  representation_shape=list(outputs['representation'].shape),
                  probability_shape=list(outputs['probabilities'].shape),
                  sequence_logits_shape=list(outputs['sequence_logits'].shape),
                  example_targets=targets[:5].tolist(), example_generated=generated[:5].tolist(),
                  finite=bool(torch.isfinite(outputs['sequence_logits']).all()))
    (outdir / 'generation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'comparison_environments': len(comparison['results']), 'generation': report}, ensure_ascii=False))


if __name__ == '__main__':
    main()
