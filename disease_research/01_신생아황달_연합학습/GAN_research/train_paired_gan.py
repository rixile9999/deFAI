"""Local image-conditioned residual GAN learned from same-patient photo pairs.

This produces bounded derivatives of training images, NOT independent patients.
It retains observed texture and constrains learned acquisition/color variations.
"""
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
import torch
from runtime import DEVICE, device_name, require_device
from torch import nn
from torch.nn import functional as F

from gan_models import Discriminator, augment
from train_gan import image_grid

ROOT = Path(__file__).resolve().parent


class PairedGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.style = nn.Linear(16+6, 8)
        self.residual = nn.Sequential(nn.Conv2d(11, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(.2),
                                      nn.Conv2d(32, 32, 3, padding=1), nn.GroupNorm(8, 32), nn.LeakyReLU(.2),
                                      nn.Conv2d(32, 16, 3, padding=1), nn.LeakyReLU(.2), nn.Conv2d(16, 3, 3, padding=1))

    def forward(self, source, noise, condition):
        style = self.style(torch.cat([noise, condition], 1))[:, :, None, None].expand(-1, -1, 64, 64)
        return (source+.15*self.residual(torch.cat([source, style], 1)).tanh()).clamp(-1, 1)


def main(steps=2000):
    require_device()
    torch.set_num_threads(2); torch.backends.cudnn.benchmark = True
    data = np.load(ROOT / "data/real_data.npz")
    output = ROOT / "results/paired_gan"; output.mkdir(parents=True, exist_ok=True)
    all_images, all_sources, all_clients, audits = [], [], [], []
    for client in range(3):
        torch.manual_seed(391+client); torch.cuda.manual_seed_all(391+client)
        rng = np.random.default_rng(391+client)
        indices = np.flatnonzero((data["split"] == "train") & (data["client"] == client))
        x = torch.tensor(data["images"][indices], dtype=torch.float32, device=DEVICE).permute(0, 3, 1, 2)/127.5-1
        raw = torch.tensor(np.column_stack([data["meta"][indices], data["blood"][indices]/30,
                                           data["treatment"][indices]]), dtype=torch.float32, device=DEVICE)
        condition = (raw-raw.mean(0))/raw.std(0).clamp_min(.1)
        # A peer is another photo of the SAME training patient and client.
        peer_sets = [np.flatnonzero(data["patient"][indices] == patient) for patient in data["patient"][indices]]
        peer_sets = [np.array([j for j in group if j != i], dtype=int) if len(group) > 1 else group
                     for i, group in enumerate(peer_sets)]
        generator, discriminator = PairedGenerator().to(DEVICE), Discriminator().to(DEVICE)
        ema = deepcopy(generator).eval()
        opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, .9))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, .9))
        start = time.monotonic(); history = []
        for step in range(1, steps+1):
            sampled = rng.integers(0, len(indices), 32)
            peers = np.array([rng.choice(peer_sets[i]) for i in sampled])
            source, real, c = x[sampled], x[peers], condition[sampled]
            target = raw[sampled, 4]
            for p in discriminator.parameters():
                p.requires_grad_(True)
            with torch.no_grad():
                fake = generator(source, torch.randn(32, 16, device=DEVICE), c)
            rs, aux = discriminator(augment(real), c); fs, _ = discriminator(augment(fake), c)
            loss_d = F.relu(1-rs).mean()+F.relu(1+fs).mean()+5*F.mse_loss(aux, target)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
            for p in discriminator.parameters():
                p.requires_grad_(False)
            fake = generator(source, torch.randn(32, 16, device=DEVICE), c)
            other = generator(source, torch.randn(32, 16, device=DEVICE), c)
            fs, aux = discriminator(augment(fake), c)
            preserve = (fake-source).abs().mean()
            peer_color = F.l1_loss(fake.mean((2, 3)), real.mean((2, 3)))
            diversity = F.relu(.01-(fake-other).abs().mean())
            loss_g = -fs.mean()+5*F.mse_loss(aux, target)+5*preserve+10*peer_color+.5*diversity
            if not torch.isfinite(loss_g+loss_d):
                raise RuntimeError("Nonfinite paired GAN loss")
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), generator.parameters()):
                    ep.lerp_(p, .005)
            if step == 1 or step % 250 == 0:
                row = {"step": step, "d_loss": float(loss_d), "g_loss": float(loss_g),
                       "preserve_l1": float(preserve), "peer_color_l1": float(peer_color),
                       "elapsed_s": time.monotonic()-start}
                history.append(row); print(f"client={client} {row}", flush=True)
        generated = []
        with torch.no_grad():
            for offset in range(0, len(indices), 32):
                batch = x[offset:offset+32]; c = condition[offset:offset+32]
                im = ema(batch, torch.randn(len(batch), 16, device=DEVICE), c)
                generated.append(((im+1)*127.5).byte().cpu().permute(0, 2, 3, 1).numpy())
            image_grid(ema(x[:32], torch.randn(32, 16, device=DEVICE), condition[:32]),
                       raw[:32, 4]*30, output / f"client_{client}_final.jpg")
        all_images.append(np.concatenate(generated)); all_sources.append(indices); all_clients.append(np.full(len(indices), client))
        torch.save({"generator": ema.state_dict(), "condition_mean": raw.mean(0).cpu(),
                    "condition_std": raw.std(0).clamp_min(.1).cpu(), "source_indices": indices.tolist()},
                   output / f"client_{client}_generator.pt")
        (output / f"client_{client}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        audits.append({"client": client, "steps": steps, "train_images": len(indices), "elapsed_s": time.monotonic()-start,
                       "peer_identity_check": "same training patient only"})
        del generator, ema, discriminator, opt_g, opt_d, x
        torch.cuda.empty_cache()
    np.savez(ROOT / "data/gan_paired_synthetic.npz", images=np.concatenate(all_images),
             source_index=np.concatenate(all_sources), client=np.concatenate(all_clients))
    (output / "training_audit.json").write_text(json.dumps({"device": device_name(), "clients": audits,
        "max_rgb_residual_fraction": .075, "interpretation": "training-image derivatives, not independent patients"}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
