"""Train separate client-local conditional GANs on RTX 4080, then generate."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch
from runtime import DEVICE, device_name, require_device
from torch.nn import functional as F

from gan_models import Generator, Discriminator, augment

ROOT = Path(__file__).resolve().parent


def image_grid(images, labels, path):
    array = ((images.detach().float().clamp(-1, 1)+1)*127.5).byte().cpu().permute(0, 2, 3, 1).numpy()
    sheet = Image.new("RGB", (8*80, int(np.ceil(len(array)/8))*82), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (im, label) in enumerate(zip(array, labels)):
        x, y = i % 8*80, i//8*82
        sheet.paste(Image.fromarray(im), (x, y))
        draw.text((x, y+64), f"T={float(label):.1f}", fill="black")
    sheet.save(path)


def train(steps=3000, seed=91, batch_size=32):
    require_device()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    data = np.load(ROOT / "data/real_data.npz")
    output = ROOT / "results/gan"
    output.mkdir(parents=True, exist_ok=True)
    audit = {"device": device_name(), "torch": torch.__version__, "steps_per_client": steps,
             "seed": seed, "batch_size": batch_size, "selection": "fixed final EMA; no test-based selection",
             "clients": []}
    all_images, all_sources, all_clients = [], [], []
    for client in range(3):
        start = time.monotonic()
        torch.manual_seed(seed+client); torch.cuda.manual_seed_all(seed+client)
        indices = np.flatnonzero((data["split"] == "train") & (data["client"] == client))
        x = torch.tensor(data["images"][indices], device=DEVICE, dtype=torch.float32).permute(0, 3, 1, 2)/127.5-1
        raw_condition = np.column_stack([data["meta"][indices], data["blood"][indices]/30, data["treatment"][indices]])
        raw_condition = torch.tensor(raw_condition, dtype=torch.float32, device=DEVICE)
        mean, std = raw_condition.mean(0), raw_condition.std(0).clamp_min(.1)
        conditions = (raw_condition-mean)/std
        generator, discriminator = Generator().to(DEVICE), Discriminator().to(DEVICE)
        ema = deepcopy(generator).eval()
        opt_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.0, .9))
        opt_d = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.0, .9))
        fixed = torch.linspace(0, len(indices)-1, 32).long().to(DEVICE)
        fixed_z = torch.randn(32, 96, device=DEVICE)
        history = []
        for step in range(1, steps+1):
            sample = torch.randint(len(indices), (batch_size,), device=DEVICE)
            c, target = conditions[sample], raw_condition[sample, 4]
            real = x[sample]
            for p in discriminator.parameters():
                p.requires_grad_(True)
            with torch.no_grad():
                fake = generator(torch.randn(batch_size, 96, device=DEVICE), c)
            real_score, real_aux = discriminator(augment(real), c)
            fake_score, _ = discriminator(augment(fake), c)
            loss_d = F.relu(1-real_score).mean()+F.relu(1+fake_score).mean()+5*F.mse_loss(real_aux, target)
            opt_d.zero_grad(set_to_none=True); loss_d.backward(); opt_d.step()
            for p in discriminator.parameters():
                p.requires_grad_(False)
            fake = generator(torch.randn(batch_size, 96, device=DEVICE), c)
            fake_score, fake_aux = discriminator(augment(fake), c)
            loss_g = -fake_score.mean()+5*F.mse_loss(fake_aux, target)
            opt_g.zero_grad(set_to_none=True); loss_g.backward(); opt_g.step()
            if not torch.isfinite(loss_g+loss_d):
                raise RuntimeError("Nonfinite GAN loss")
            with torch.no_grad():
                for ep, p in zip(ema.parameters(), generator.parameters()):
                    ep.lerp_(p, .005)
                for eb, b in zip(ema.buffers(), generator.buffers()):
                    eb.copy_(b)
            if step == 1 or step % 250 == 0:
                row = {"step": step, "d_loss": float(loss_d), "g_loss": float(loss_g),
                       "real_aux_mae_mgdl": float((real_aux-target).abs().mean()*30),
                       "elapsed_s": time.monotonic()-start}
                history.append(row)
                print(f"client={client} {row}", flush=True)
            if step % 500 == 0 or step == steps:
                with torch.no_grad():
                    preview = ema(fixed_z, conditions[fixed])
                image_grid(preview, raw_condition[fixed, 4]*30, output / f"client_{client}_step_{step}.jpg")
                torch.save({"generator": ema.state_dict(), "condition_mean": mean.cpu(), "condition_std": std.cpu(),
                            "client": client, "step": step, "source_indices": indices.tolist()},
                           output / f"client_{client}_generator.pt")
                (output / f"client_{client}_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        generated = []
        with torch.no_grad():
            for offset in range(0, len(indices), batch_size):
                cond = conditions[offset:offset+batch_size]
                im = ema(torch.randn(len(cond), 96, device=DEVICE), cond)
                generated.append(((im.clamp(-1, 1)+1)*127.5).byte().cpu().permute(0, 2, 3, 1).numpy())
        all_images.append(np.concatenate(generated)); all_sources.append(indices)
        all_clients.append(np.full(len(indices), client))
        audit["clients"].append({"client": client, "real_training_images": len(indices),
                                 "generated_images": len(indices), "elapsed_s": time.monotonic()-start,
                                 "train_patients": len(np.unique(data["patient"][indices]))})
        (output / "training_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
        del x, generator, ema, discriminator, opt_g, opt_d
        torch.cuda.empty_cache()
    np.savez(ROOT / "data/gan_synthetic.npz", images=np.concatenate(all_images),
             source_index=np.concatenate(all_sources), client=np.concatenate(all_clients))
    print("GAN training and local synthetic generation complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=91)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    train(args.steps, args.seed, args.batch_size)
