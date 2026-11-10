import torch
import torch.nn as nn
import torch.nn.functional as F


def logit_kd(zs, zt, T):
    return F.kl_div(F.log_softmax(zs / T, 1), F.log_softmax(zt / T, 1), reduction="batchmean", log_target=True) * T * T


def ensemble_logits(teacher_logits, weights=None, T=1.0):
    probs = torch.stack([F.softmax(z / T, 1) for z in teacher_logits])
    w = torch.ones(len(teacher_logits)) if weights is None else torch.as_tensor(weights, dtype=probs.dtype)
    w = (w / w.sum()).view(-1, 1, 1)
    return torch.log((probs * w).sum(0).clamp_min(1e-8)) * T


def layer_norm(f):
    return F.layer_norm(f, f.shape[1:])


def feature_mse(fs, ft):
    return F.mse_loss(layer_norm(fs), layer_norm(ft))


def depth_weights(n, mode="linear"):
    if mode == "uniform":
        w = torch.ones(n)
    elif mode == "linear":
        w = torch.arange(1, n + 1, dtype=torch.float32)
    else:
        w = 2.0 ** torch.arange(n, dtype=torch.float32)
    return w / w.sum()


def attention_raw(f):
    if f.dim() == 2:
        return f.pow(2)
    return f.pow(2).mean(1).flatten(1)


def attention_map(f):
    return F.normalize(attention_raw(f), dim=1)


def attention_logp(f, tau):
    a = attention_raw(f)
    z = (a - a.mean(1, keepdim=True)) / (a.std(1, keepdim=True) + 1e-6)
    return F.log_softmax(z / tau, dim=1)


def at_loss(fs, ft):
    return sum((attention_map(s) - attention_map(t)).pow(2).mean() for s, t in zip(fs, ft))


def at_gain(fs, ft, tau):
    total = 0.0
    for s, t in zip(fs, ft):
        lp_t, lp_s = attention_logp(t, tau), attention_logp(s, tau)
        total = total + (lp_t.exp() * (lp_t - lp_s)).sum(1).mean()
    return total


def gan_d_loss(d_real, d_fake):
    return 0.5 * ((d_real - 1).pow(2).mean() + d_fake.pow(2).mean())


def gan_g_loss(d_fake):
    return 0.5 * (d_fake - 1).pow(2).mean()


class Adapter(nn.Module):
    def __init__(self, s_shape, t_shape):
        super().__init__()
        if len(t_shape) == 4:
            self.net = nn.Sequential(nn.Conv2d(s_shape[1], t_shape[1], 1, bias=False), nn.BatchNorm2d(t_shape[1]))
        else:
            self.net = nn.Linear(s_shape[1], t_shape[1])

    def forward(self, x):
        return self.net(x)


class Generator(nn.Module):
    def __init__(self, s_shape, t_shape, noise=4):
        super().__init__()
        self.noise = noise
        if len(t_shape) == 4:
            self.net = nn.Sequential(nn.Conv2d(s_shape[1] + noise, t_shape[1], 1, bias=False),
                                     nn.BatchNorm2d(t_shape[1]), nn.ReLU())
        else:
            self.net = nn.Sequential(nn.Linear(s_shape[1] + noise, t_shape[1]), nn.ReLU())

    def forward(self, f):
        if f.dim() == 4:
            z = torch.randn(f.shape[0], self.noise, f.shape[2], f.shape[3])
        else:
            z = torch.randn(f.shape[0], self.noise)
        return self.net(torch.cat([f, z], 1))


class Discriminator(nn.Module):
    def __init__(self, t_shape, width=64):
        super().__init__()
        sn = nn.utils.parametrizations.spectral_norm
        if len(t_shape) == 4:
            layers = [sn(nn.Conv2d(t_shape[1], width, 3, padding=1)), nn.LeakyReLU(0.2)]
            c, s = width, t_shape[2]
            while s > 4:
                layers += [sn(nn.Conv2d(c, width, 3, stride=2, padding=1)), nn.LeakyReLU(0.2)]
                s = (s + 1) // 2
            self.body = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.head = sn(nn.Linear(width, 1))
        else:
            self.body = nn.Sequential(sn(nn.Linear(t_shape[1], width)), nn.LeakyReLU(0.2))
            self.head = sn(nn.Linear(width, 1))

    def forward(self, f):
        return self.head(self.body(f)).squeeze(1)
