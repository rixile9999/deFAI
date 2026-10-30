"""Brightness-nuisance suppression, not a biological melanin measurement.

The transform scales all encoded RGB channels equally without clipping. Thus
channel ratios survive, while measured brightness changes. Real melanin is
wavelength dependent: this is deliberately only a restricted nuisance model.
"""
import torch


def normalize(rgb):
    mean = rgb.new_tensor([.485, .456, .406])[None, :, None, None]
    std = rgb.new_tensor([.229, .224, .225])[None, :, None, None]
    return (rgb-mean)/std


def denormalize(image):
    mean = image.new_tensor([.485, .456, .406])[None, :, None, None]
    std = image.new_tensor([.229, .224, .225])[None, :, None, None]
    return (image*std+mean).clamp(0, 1)


def suppress_brightness(image, method):
    if method != "chromaticity":
        return image
    rgb = denormalize(image)
    # Each pixel sums to one; no learned statistics or test information.
    chromaticity = rgb/rgb.sum(1, keepdim=True).clamp_min(1e-8)
    return normalize(chromaticity)


def attenuate(image, gain):
    rgb = denormalize(image)
    gain = torch.as_tensor(gain, dtype=rgb.dtype, device=rgb.device)
    if gain.ndim == 1:
        gain = gain[:, None, None, None]
    if bool(((gain <= 0) | (gain > 1)).any()):
        raise ValueError("Only positive attenuation <= 1 avoids clipping")
    return normalize(rgb*gain)
