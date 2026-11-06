import math
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fl_core.task import batches
from layer_kd import losses as L
from layer_kd.features import FeatureTap, stage_points

METHODS = ("student", "logit", "ensemble", "wfm", "at", "gan", "combined")
PARTS = {"student": set(), "logit": {"kd"}, "ensemble": {"kd"}, "wfm": {"kd", "fm"}, "at": {"kd", "at"},
         "gan": {"kd", "gan"}, "combined": {"kd", "fm", "at", "gan"}}


@dataclass
class KDConfig:
    method: str = "combined"
    T: float = 3.0
    alpha: float = 0.7
    clip: float = 2.0
    beta_fm: float = 1.0
    depth_mode: str = "linear"
    layer_lr: bool = True
    beta_at: float = 50.0
    tau_a: float = 1.0
    lambda_gain: float = 0.5
    lambda_adv: float = 0.1
    lambda_rec: float = 0.5
    gan_stages: tuple = (-1,)
    d_lr: float = 2e-4
    teacher_weights: tuple = ()
    parts: set = field(default_factory=set, repr=False)

    def __post_init__(self):
        if self.method not in METHODS:
            raise ValueError(f"unknown KD method {self.method!r}; choose from {', '.join(METHODS)}")
        self.parts = set(PARTS[self.method])

    def to_dict(self):
        d = asdict(self)
        d.pop("parts")
        return d


class KDTrainer:
    def __init__(self, cfg, teachers, student, input_shape):
        self.cfg = cfg
        self.teachers = [t.eval() for t in (teachers if isinstance(teachers, (list, tuple)) else [teachers]) if t is not None]
        for t in self.teachers:
            for p in t.parameters():
                p.requires_grad_(False)
        self.student = student
        s_names, s_shapes, self.exec_order = stage_points(student, input_shape)
        self.s_tap = FeatureTap(student, s_names)
        self.t_taps = []
        t_shapes = None
        for t in self.teachers:
            t_names, t_sh, _ = stage_points(t, input_shape)
            if len(t_names) != len(s_names):
                raise ValueError(f"teacher has {len(t_names)} matching stages, student {len(s_names)}")
            self.t_taps.append(FeatureTap(t, t_names))
            t_shapes = t_shapes or [t_sh[n] for n in t_names]
        s_list = [s_shapes[n] for n in s_names]
        self.stage_names = s_names
        self.layer_w = L.depth_weights(len(s_names), cfg.depth_mode)
        self.aux = nn.ModuleDict()
        self.disc = nn.ModuleDict()
        if t_shapes is not None:
            for k, (ss, ts) in enumerate(zip(s_list, t_shapes)):
                if "fm" in cfg.parts:
                    self.aux[f"fm{k}"] = L.Adapter(ss, ts)
            gan_idx = [i % len(s_list) for i in cfg.gan_stages]
            if "gan" in cfg.parts:
                for k in gan_idx:
                    self.aux[f"g{k}"] = L.Generator(s_list[k], t_shapes[k])
                    self.disc[f"d{k}"] = L.Discriminator(t_shapes[k])
        self.gan_idx = [i % len(s_list) for i in cfg.gan_stages]
        self.opt_d = torch.optim.Adam(self.disc.parameters(), lr=cfg.d_lr, betas=(0.5, 0.999)) if len(self.disc) else None

    def param_groups(self, lr):
        aux = [{"params": list(self.aux.parameters()), "lr": lr}]
        if not (self.cfg.layer_lr and self.teachers and "fm" in self.cfg.parts):
            return [{"params": list(self.student.parameters()), "lr": lr}] + aux
        n = len(self.stage_names)
        scale = (self.layer_w * n).clamp_min(0.5).tolist()
        stage_of, k = {}, 0
        for name in self.exec_order:
            if k < n:
                stage_of[name] = k
                if name == self.stage_names[k]:
                    k += 1
        groups = {}
        for mname, m in self.student.named_modules():
            own = list(m.parameters(recurse=False))
            if own:
                groups.setdefault(round(scale[stage_of[mname]] if mname in stage_of else 1.0, 4), []).extend(own)
        return [{"params": ps, "lr": lr * sc} for sc, ps in groups.items()] + aux

    def train_mode(self):
        self.student.train()
        self.aux.train()
        self.disc.train()

    def step(self, x, y, opt):
        cfg, parts = self.cfg, self.cfg.parts
        logs = {}
        zs, fs = self.s_tap(x)
        ce = F.cross_entropy(zs, y)
        total = ce
        logs["ce"] = ce.item()
        if self.teachers and parts:
            with torch.no_grad():
                outs = [tap(x) for tap in self.t_taps]
            zts = [o[0] for o in outs]
            ft = outs[0][1]
            zt = zts[0] if len(zts) == 1 else L.ensemble_logits(zts, cfg.teacher_weights or None)
            if "kd" in parts:
                kd = L.logit_kd(zs, zt, cfg.T)
                total = (1 - cfg.alpha) * ce + cfg.alpha * kd
                logs["kd"] = kd.item()
            if "fm" in parts:
                fm = sum(self.layer_w[k] * L.feature_mse(self.aux[f"fm{k}"](fs[k]), ft[k]) for k in range(len(fs)))
                total = total + cfg.beta_fm * fm * len(fs)
                logs["fm"] = fm.item()
            if "at" in parts:
                pairs = [(s, t) for s, t in zip(fs, ft) if s.dim() == 4]
                if pairs:
                    a = 0.5 * cfg.beta_at * L.at_loss([p[0] for p in pairs], [p[1] for p in pairs])
                    g = L.at_gain([p[0] for p in pairs], [p[1] for p in pairs], cfg.tau_a)
                    total = total + a + cfg.lambda_gain * g
                    logs["at_loss"], logs["at_gain"] = a.item(), g.item()
            if "gan" in parts:
                fakes = {k: self.aux[f"g{k}"](fs[k]) for k in self.gan_idx}
                for p in self.disc.parameters():
                    p.requires_grad_(True)
                self.opt_d.zero_grad(set_to_none=True)
                d_loss = sum(L.gan_d_loss(self.disc[f"d{k}"](ft[k]), self.disc[f"d{k}"](f.detach()))
                             for k, f in fakes.items())
                d_loss.backward()
                self.opt_d.step()
                for p in self.disc.parameters():
                    p.requires_grad_(False)
                adv = sum(L.gan_g_loss(self.disc[f"d{k}"](f)) for k, f in fakes.items())
                rec = sum(L.feature_mse(f, ft[k]) for k, f in fakes.items())
                total = total + cfg.lambda_adv * adv + cfg.lambda_rec * rec
                logs.update(adv=adv.item(), rec=rec.item(), d_loss=d_loss.item())
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(list(self.student.parameters()) + list(self.aux.parameters()), cfg.clip)
        opt.step()
        logs["loss"] = total.item()
        if not math.isfinite(logs["loss"]):
            raise FloatingPointError(f"non-finite KD loss {logs}")
        return logs, zs.detach()

    def fit(self, x, y, epochs=1.0, batch_size=64, lr=0.05, seed=0, momentum=0.9):
        opt = torch.optim.SGD(self.param_groups(lr), lr=lr, momentum=momentum)
        steps_total = max(1, int(round(epochs * len(y) / batch_size)))
        self.train_mode()
        step, sums, ep = 0, {}, 0
        while step < steps_total:
            for xb, yb in batches(x, y, np.arange(len(y)), batch_size, shuffle=True, seed=seed * 7 + ep):
                logs, _ = self.step(xb, yb, opt)
                for k, v in logs.items():
                    sums[k] = sums.get(k, 0.0) + v
                step += 1
                if step >= steps_total:
                    break
            ep += 1
        return {k: v / step for k, v in sums.items()}, step

    def close(self):
        self.s_tap.close()
        for t in self.t_taps:
            t.close()
