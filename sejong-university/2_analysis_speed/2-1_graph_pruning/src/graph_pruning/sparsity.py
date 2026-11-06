import torch.nn as nn


def bn_l1(model, lam):
    return lam * sum(m.weight.abs().sum() for m in model.modules()
                     if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.weight is not None)


def bn_l1_loss_factory(lam):
    def factory(model, msg, context):
        return lambda m, xb, yb, out: bn_l1(m, lam)

    return factory


def bn_scale_stats(model, threshold=1e-2):
    vals = [m.weight.detach().abs() for m in model.modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)) and m.weight is not None]
    if not vals:
        return {"bn_channels": 0, "bn_l1": 0.0, "bn_near_zero": 0.0}
    import torch

    v = torch.cat(vals)
    return {"bn_channels": int(v.numel()), "bn_l1": float(v.sum()), "bn_near_zero": float((v < threshold).float().mean())}


def weight_sparsity(model):
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += int((p == 0).sum())
    return zeros / max(total, 1)
