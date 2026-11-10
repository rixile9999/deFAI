import torch
import torch.nn as nn


def stage_points(model, input_shape):
    order, shapes = [], {}
    handles = []
    for name, module in model.named_modules():
        if not name:
            continue

        def hook(m, inp, out, name=name):
            if isinstance(out, torch.Tensor):
                order.append(name)
                shapes[name] = tuple(out.shape)

        handles.append(module.register_forward_hook(hook))
    was = model.training
    model.eval()
    with torch.no_grad():
        model(torch.zeros(input_shape))
    model.train(was)
    for h in handles:
        h.remove()
    last_linear = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    head = last_linear[-1] if last_linear else None
    stages, seen = [], {}
    for name in order:
        shp = shapes[name]
        if len(shp) == 4:
            seen[shp[2:]] = name
    for key in sorted(seen, key=lambda s: -s[0] * s[1]):
        stages.append(seen[key])
    flat = [n for n in order if len(shapes[n]) == 2 and n != head and shapes[n][1] > 10]
    if flat:
        stages.append(flat[-1])
    return stages, {n: shapes[n] for n in stages}, order


class FeatureTap:
    def __init__(self, model, names):
        self.model = model
        self.names = list(names)
        self.store = {}
        modules = dict(model.named_modules())
        self.handles = [modules[n].register_forward_hook(self._hook(n)) for n in self.names]

    def _hook(self, name):
        def fn(m, inp, out):
            self.store[name] = out

        return fn

    def __call__(self, x):
        self.store = {}
        logits = self.model(x)
        return logits, [self.store[n] for n in self.names]

    def close(self):
        for h in self.handles:
            h.remove()
