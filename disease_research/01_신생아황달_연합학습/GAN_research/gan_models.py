"""Conditional skin-patch GAN: resize-convolution G, projection/auxiliary D."""
import math

import torch
from torch import nn
from torch.nn.utils.parametrizations import spectral_norm


class Generator(nn.Module):
    def __init__(self, latent=96, conditions=6):
        super().__init__()
        self.input = nn.Linear(latent+conditions, 256*4*4)
        layers = []
        for cin, cout in [(256, 128), (128, 64), (64, 32), (32, 16)]:
            layers += [nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(cin, cout, 3, padding=1),
                       nn.BatchNorm2d(cout), nn.ReLU(True)]
        self.image = nn.Sequential(*layers, nn.Conv2d(16, 3, 3, padding=1), nn.Tanh())

    def forward(self, z, condition):
        return self.image(self.input(torch.cat([z, condition], 1)).reshape(-1, 256, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, conditions=6):
        super().__init__()
        layers = []
        for cin, cout in [(3, 32), (32, 64), (64, 128), (128, 256)]:
            layers += [spectral_norm(nn.Conv2d(cin, cout, 4, 2, 1)), nn.LeakyReLU(.2)]
        self.features = nn.Sequential(*layers)
        self.score = spectral_norm(nn.Linear(256, 1))
        self.projection = spectral_norm(nn.Linear(conditions, 256))
        self.bilirubin = nn.Linear(256, 1)

    def forward(self, image, condition):
        features = self.features(image).mean((2, 3))
        score = self.score(features).squeeze(1)+(features*self.projection(condition)).sum(1)/math.sqrt(256)
        return score, self.bilirubin(features).squeeze(1)


def augment(image):
    """Differentiable geometry/cutout; no random hue shifts on a color-based task."""
    if torch.rand((), device=image.device) < .5:
        image = image.flip(-1)
    shifts = torch.randint(-2, 3, (2,)).tolist()
    image = torch.roll(image, shifts, (2, 3))
    b, _, h, w = image.shape
    x = torch.randint(0, w, (b, 1, 1, 1), device=image.device)
    y = torch.randint(0, h, (b, 1, 1, 1), device=image.device)
    xx = torch.arange(w, device=image.device)[None, None, None, :]
    yy = torch.arange(h, device=image.device)[None, None, :, None]
    mask = ((xx-x).abs() > 4) | ((yy-y).abs() > 4)
    return image*mask
