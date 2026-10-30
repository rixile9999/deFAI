"""Descriptive held-out-validation image checks; no clinical validity claim."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
from runtime import DEVICE, device_name, require_device
from torchvision import models

ROOT = Path(__file__).resolve().parent


def features(model, array):
    output = []
    for offset in range(0, len(array), 128):
        image = torch.tensor(array[offset:offset+128], dtype=torch.float32, device=DEVICE).permute(0, 3, 1, 2)/255
        image = (image-torch.tensor([.485, .456, .406], device=DEVICE)[None, :, None, None]) / \
                torch.tensor([.229, .224, .225], device=DEVICE)[None, :, None, None]
        with torch.no_grad():
            output.append(model(image).flatten(1).cpu().numpy())
    return np.concatenate(output).astype(np.float64)


def kid(x, y):
    # Unbiased polynomial-kernel MMD on frozen ImageNet ResNet18 features.
    # This is NOT standard Inception FID/KID and is not a medical realism measure.
    xx = (x @ x.T/x.shape[1]+1)**3
    yy = (y @ y.T/y.shape[1]+1)**3
    xy = (x @ y.T/x.shape[1]+1)**3
    return float((xx.sum()-np.trace(xx))/(len(x)*(len(x)-1)) +
                 (yy.sum()-np.trace(yy))/(len(y)*(len(y)-1)) - 2*xy.mean())


def main():
    torch.set_num_threads(2)
    real = np.load(ROOT / "data/real_data.npz"); synthetic = np.load(ROOT / "data/gan_synthetic.npz")
    train = real["images"][real["split"] == "train"]
    val = real["images"][real["split"] == "val"]
    source = synthetic["source_index"]
    rng = np.random.default_rng(91)
    old = np.zeros_like(synthetic["images"], dtype=np.int16)
    old[..., 0] = 255; old[..., 1] = 200
    old[..., 2] = (200-(100*real["blood"][source]/20).astype(int))[:, None, None]
    old = np.clip(old+rng.integers(-20, 20, old.shape, dtype=np.int16), 0, 255).astype(np.uint8)
    backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model = torch.nn.Sequential(*list(backbone.children())[:-1]).to(DEVICE).eval()
    fv = features(model, val)
    result = {"stage": "research_only", "reference": "real validation images; test images not used",
              "metric": "unbiased polynomial MMD on ImageNet ResNet18 features, not Inception FID/KID",
              "n_reference": len(val), "distances": {}, "exact_crop_copies": 0}
    image_sets = [("real_train", train), ("old_procedural", old), ("local_conditional_gan", synthetic["images"])]
    paired_path = ROOT / "data/gan_paired_synthetic.npz"
    if paired_path.exists():
        image_sets.append(("paired_residual_gan", np.load(paired_path)["images"]))
    for name, array in image_sets:
        result["distances"][name] = kid(features(model, array), fv)
    real_hashes = {image.tobytes() for image in train}
    result["exact_crop_copies"] = sum(image.tobytes() in real_hashes for image in synthetic["images"])
    if paired_path.exists():
        paired = np.load(paired_path)
        result["paired_exact_crop_copies"] = sum(image.tobytes() in real_hashes for image in paired["images"])
        result["paired_source_mean_absolute_RGB_change"] = float(np.abs(
            paired["images"].astype(float)-real["images"][paired["source_index"]].astype(float)).mean())
        result["paired_interpretation"] = "Small derivatives of source training photos; not new patients or privacy-safe releases"
    # Coarse nearest-neighbor distances are descriptive, not privacy certification.
    def downsample(array):
        tensor = torch.tensor(array, dtype=torch.float32).permute(0, 3, 1, 2)/255
        return torch.nn.functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten(1)
    ft, fs, fval = downsample(train), downsample(synthetic["images"]), downsample(val)
    for name, values in [("generated_to_train", fs), ("real_validation_to_train", fval)]:
        distance = torch.cdist(values, ft)/np.sqrt(ft.shape[1])
        result[name+"_coarse_nn_rmse"] = float(distance.min(1).values.mean())
    display = [("real_source_training", real["images"][source]), *image_sets[1:]]
    sheet = Image.new("RGB", (640, len(display)*88), "white"); draw = ImageDraw.Draw(sheet)
    for row, (name, array) in enumerate(display):
        for i in range(8):
            sheet.paste(Image.fromarray(array[i]), (i*80, row*88+20))
        draw.text((0, row*88), name, fill="black")
    output = ROOT / "results"
    sheet.save(output / "generation_comparison.jpg")
    (output / "generation_quality.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
