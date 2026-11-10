from __future__ import annotations
import contextlib
import copy
import inspect
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable
import numpy as np
import torch
import torch.fx as fx
import torch.nn as nn
import torch.nn.functional as F
from torch.fx.passes.shape_prop import ShapeProp
_LAYER_TYPES = (nn.Conv2d, nn.Linear)
_NORM_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d)
_PRUNABLE_TYPES = _LAYER_TYPES + _NORM_TYPES
_PASS_MODULES = (nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.GELU, nn.SiLU, nn.Hardswish, nn.Sigmoid, nn.Tanh, nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d, nn.Dropout, nn.Dropout2d, nn.Identity)
_PASS_OPS = {'relu', 'relu_', 'relu6', 'leaky_relu', 'elu', 'gelu', 'silu', 'hardswish', 'sigmoid', 'tanh', 'max_pool2d', '_max_pool2d', 'avg_pool2d', 'adaptive_avg_pool2d', 'adaptive_max_pool2d', '_adaptive_max_pool2d', 'dropout', 'dropout2d', 'clone', 'contiguous', 'detach', 'identity'}
_RESHAPE_OPS = {'flatten', 'view', 'reshape', 'mean', 'amax', 'sum'}
_REDUCE_OPS = {'mean', 'amax', 'sum'}
_ELEMWISE_OPS = {'add', 'add_', 'iadd', 'sub', 'sub_', 'isub', 'mul', 'mul_', 'imul'}
_HARMLESS_OPS = {'size', 'dim', 'numel', 'getattr', 'eq', 'ne', '_assert', 'is_contiguous'}

def default_min_keep(c0: int) -> int:
    return max(4, math.ceil(0.1 * c0))

@contextlib.contextmanager
def _eval_mode(model: nn.Module):
    flags = {m: m.training for m in model.modules()}
    model.eval()
    try:
        yield model
    finally:
        for m, f in flags.items():
            m.training = f

def default_input_shape(model: nn.Module, input_shape=None) -> tuple:
    if input_shape:
        return tuple(input_shape)
    meta = getattr(model, 'prune_meta', None) or {}
    if meta.get('input_shape'):
        return tuple(meta['input_shape'])
    if getattr(model, 'input_shape', None):
        return tuple(model.input_shape)
    conv = next((m for m in model.modules() if isinstance(m, nn.Conv2d)), None)
    return (1, conv.in_channels if conv is not None else 3, 32, 32)

@dataclass
class Group:
    key: str
    channels: int
    producers: list[str] = field(default_factory=list)
    norms: list[str] = field(default_factory=list)
    consumers: list[tuple[str, int]] = field(default_factory=list)
    frozen: bool = False
    reason: str = ''

    @property
    def coupled(self) -> bool:
        return len(self.producers) > 1

@dataclass
class ChannelGraph:
    groups: list[Group]
    input_shape: tuple
    n_nodes: int = 0

    @property
    def prunable(self) -> list[Group]:
        return [g for g in self.groups if not g.frozen]

    def group(self, key: str) -> Group:
        return next((g for g in self.groups if g.key == key))

    def describe(self) -> str:
        lines = [f'{self.n_nodes} fx nodes, {len(self.groups)} channel spaces, {len(self.prunable)} prunable groups']
        for i, g in enumerate(self.groups):
            tag = f'FROZEN({g.reason})' if g.frozen else 'COUPLED' if g.coupled else 'single'
            cons = [q if hw == 1 else f'{q}[x{hw}]' for q, hw in g.consumers]
            lines.append(f'  G{i:02d} {g.key:<22s} C={g.channels:<4d} {tag:<16s} producers={g.producers} norms={g.norms} consumers={cons}')
        return '\n'.join(lines)

class _UnionFind:
    def __init__(self):
        self.parent: list[int] = []
        self.size: list[int] = []
        self.frozen: list[str] = []

    def new(self, size: int, frozen: str='') -> int:
        self.parent.append(len(self.parent))
        self.size.append(int(size))
        self.frozen.append(frozen)
        return len(self.parent) - 1

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> int:
        ra, rb = (self.find(a), self.find(b))
        if ra != rb:
            assert self.size[ra] == self.size[rb], 'merged channel spaces must have equal width'
            self.parent[rb] = ra
            self.frozen[ra] = self.frozen[ra] or self.frozen[rb]
        return ra

    def freeze(self, i: int, reason: str) -> None:
        r = self.find(i)
        self.frozen[r] = self.frozen[r] or reason

def trace(model: nn.Module) -> fx.GraphModule:
    params = list(inspect.signature(model.forward).parameters.values())[1:]
    concrete = {p.name: p.default for p in params if p.default is not inspect.Parameter.empty}
    return fx.symbolic_trace(model, concrete_args=concrete or None)

def _shape(node) -> tuple | None:
    tm = node.meta.get('tensor_meta') if isinstance(node, fx.Node) else None
    return tuple(tm.shape) if tm is not None and hasattr(tm, 'shape') else None

def _op_name(node: fx.Node) -> str:
    t = node.target
    return t if isinstance(t, str) else getattr(t, '__name__', str(t))

def _spatial_reduction(node: fx.Node, in_shape) -> bool:
    if in_shape is None or len(in_shape) != 4:
        return False
    d = node.args[1] if len(node.args) > 1 else node.kwargs.get('dim')
    if isinstance(d, int):
        d = (d,)
    if not isinstance(d, (tuple, list)) or not all((isinstance(i, int) for i in d)):
        return False
    return {i % 4 for i in d} == {2, 3}

def build_graph(model: nn.Module, input_shape=None) -> ChannelGraph:
    input_shape = default_input_shape(model, input_shape)
    gm = trace(model)
    with _eval_mode(model):
        p = next(model.parameters())
        with torch.no_grad():
            ShapeProp(gm).propagate(torch.zeros(input_shape, dtype=p.dtype, device=p.device))
    uf = _UnionFind()
    val: dict[fx.Node, tuple[int, int] | None] = {}
    prods, norms, cons = ([], [], [])
    calls: dict[str, list[int]] = defaultdict(list)
    n_calls: dict[str, int] = defaultdict(int)

    def freeze_inputs(node, reason):
        for a in node.all_input_nodes:
            if val.get(a) is not None:
                uf.freeze(val[a][0], reason)

    def reshape(x, in_shape, out_shape):
        if x is None or in_shape is None or out_shape is None or (len(out_shape) < 2) or (out_shape[0] != in_shape[0]):
            return None
        sid, hw = x
        c = uf.size[uf.find(sid)]
        if len(in_shape) == 4 and hw == 1:
            spatial = in_shape[2] * in_shape[3]
            if len(out_shape) == 4 and out_shape[1] == c:
                return x
            if len(out_shape) == 2 and out_shape[1] == c * spatial:
                return (sid, spatial)
            if len(out_shape) == 2 and out_shape[1] == c:
                return (sid, 1)
        if len(in_shape) == 2 and len(out_shape) == 2 and (out_shape[1] == in_shape[1]):
            return x
        return None
    tensor_inputs_seen = False
    for node in gm.graph.nodes:
        shp = _shape(node)
        if node.op == 'placeholder':
            if shp is not None and (not tensor_inputs_seen) and (len(shp) >= 2):
                val[node] = (uf.new(shp[1], frozen='input'), 1)
                tensor_inputs_seen = True
            continue
        if node.op == 'output':
            freeze_inputs(node, 'output')
            continue
        if node.op == 'get_attr':
            val[node] = None
            continue
        name = _op_name(node)
        if shp is None:
            if node.op == 'call_module' or name not in _HARMLESS_OPS:
                freeze_inputs(node, f'op:{name}')
            continue
        tensor_args = [a for a in node.args if isinstance(a, fx.Node) and _shape(a) is not None]
        x0 = tensor_args[0] if tensor_args else None
        x = val.get(x0) if x0 is not None else None
        in_shape = _shape(x0) if x0 is not None else None
        if node.op == 'call_module':
            path = node.target
            mod = model.get_submodule(path)
            n_calls[path] += 1
            if isinstance(mod, _LAYER_TYPES):
                is_conv = isinstance(mod, nn.Conv2d)
                out_c = shp[1] if is_conv else shp[-1]
                if is_conv and mod.groups != 1 or (not is_conv and len(shp) != 2):
                    freeze_inputs(node, 'grouped/nd layer')
                    sid = uf.new(out_c, frozen='grouped/nd layer')
                else:
                    if x is not None:
                        ok = is_conv and x[1] == 1 and (len(in_shape) == 4) or (not is_conv and len(in_shape) == 2 and (uf.size[uf.find(x[0])] * x[1] == in_shape[1]))
                        if ok:
                            cons.append((path, x[0], x[1]))
                            calls[path].append(x[0])
                        else:
                            uf.freeze(x[0], f'layout:{path}')
                    sid = uf.new(out_c)
                    prods.append((path, sid))
                calls[path].append(sid)
                val[node] = (sid, 1)
            elif isinstance(mod, _NORM_TYPES):
                val[node] = x
                if x is not None:
                    if x[1] != 1 or mod.num_features != uf.size[uf.find(x[0])]:
                        uf.freeze(x[0], f'layout:{path}')
                    else:
                        norms.append((path, x[0]))
                    calls[path].append(x[0])
            elif isinstance(mod, _PASS_MODULES) and in_shape is not None and (shp[:2] == in_shape[:2]):
                val[node] = x
            elif isinstance(mod, nn.Flatten):
                val[node] = reshape(x, in_shape, shp)
                if val[node] is None:
                    freeze_inputs(node, f'op:{path}')
            else:
                freeze_inputs(node, f'module:{type(mod).__name__}')
                val[node] = None
        elif name in _PASS_OPS and in_shape is not None and (len(shp) == len(in_shape)) and (shp[:2] == in_shape[:2]):
            val[node] = x
        elif name in _RESHAPE_OPS:
            ok = name not in _REDUCE_OPS or _spatial_reduction(node, in_shape)
            val[node] = reshape(x, in_shape, shp) if ok else None
            if val[node] is None:
                freeze_inputs(node, f'op:{name}')
        elif name in _ELEMWISE_OPS:
            vs = [val.get(a) for a in tensor_args]
            shapes = [_shape(a) for a in tensor_args]
            if len(tensor_args) == 1 and shapes[0] == shp:
                val[node] = vs[0]
            elif len(tensor_args) == 2 and None not in vs and (shapes[0] == shapes[1] == shp) and (vs[0][1] == vs[1][1]):
                val[node] = (uf.union(vs[0][0], vs[1][0]), vs[0][1])
            else:
                freeze_inputs(node, f'op:{name}')
                val[node] = None
        else:
            freeze_inputs(node, f'op:{name}')
            val[node] = None
    for path, sids in calls.items():
        if n_calls[path] > 1:
            for s in sids:
                uf.freeze(s, f'shared:{path}')
    groups: dict[int, Group] = {}
    order: list[int] = []

    def grp(sid: int) -> Group:
        r = uf.find(sid)
        if r not in groups:
            groups[r] = Group(key='', channels=uf.size[r])
            order.append(r)
        return groups[r]
    for sid in range(len(uf.parent)):
        grp(sid)
    for path, sid in prods:
        g = grp(sid)
        if path not in g.producers:
            g.producers.append(path)
    for path, sid in norms:
        g = grp(sid)
        if path not in g.norms:
            g.norms.append(path)
    for path, sid, hw in cons:
        g = grp(sid)
        if (path, hw) not in g.consumers:
            g.consumers.append((path, hw))
    out = []
    for r in order:
        g = groups[r]
        g.frozen, g.reason = (bool(uf.frozen[r]), uf.frozen[r])
        g.key = g.producers[0] if g.producers else f'{uf.frozen[r] or 'space'}{r}'
        out.append(g)
    return ChannelGraph(groups=out, input_shape=input_shape, n_nodes=len(gm.graph.nodes))

def _out_sq(w: torch.Tensor) -> torch.Tensor:
    return w.detach().double().pow(2).flatten(1).sum(1)

def _in_sq(w: torch.Tensor, channels: int, hw: int) -> torch.Tensor:
    s = w.detach().double().pow(2).transpose(0, 1).flatten(1).sum(1)
    return s.view(channels, hw).sum(1)

def _pristine(meta: dict | None) -> bool:
    return bool(meta) and all((len(v) == meta['orig'].get(k, len(v)) for k, v in meta['orig_idx'].items()))

def anchor_importance(model: nn.Module, kind: str, rel: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    meta = getattr(model, 'prune_meta', None)
    if meta is None:
        return rel
    store = (meta.get('imp0') or {}).get(kind)
    if store is None:
        if _pristine(meta):
            meta.setdefault('imp0', {})[kind] = {k: [round(float(x), 7) for x in v] for k, v in rel.items()}
        return rel
    out = {}
    for k, v in rel.items():
        if k in store and k in meta['orig_idx']:
            i0 = torch.as_tensor(store[k], dtype=torch.float64)[torch.as_tensor(meta['orig_idx'][k], dtype=torch.long)]
            out[k] = (v.double() * i0.mean()).float()
        else:
            out[k] = v
    return out

@torch.no_grad()
def path_importance(model: nn.Module, graph: ChannelGraph, eps: float=1e-12, anchor: bool=True) -> dict[str, torch.Tensor]:
    out = {}
    for g in graph.prunable:
        p = torch.zeros(g.channels, dtype=torch.float64)
        for name in g.producers:
            s = _out_sq(model.get_submodule(name).weight)
            p += s / (s.mean() + eps)
        for name, hw in g.consumers:
            s = _in_sq(model.get_submodule(name).weight, g.channels, hw)
            p += s / (s.mean() + eps)
        gam = [model.get_submodule(n).weight.detach().abs().double() for n in g.norms if getattr(model.get_submodule(n), 'weight', None) is not None]
        if gam:
            p = p * torch.stack(gam).mean(0)
        out[g.key] = (p / (p.mean() + eps)).float()
    return anchor_importance(model, 'path', out) if anchor else out

def taylor_importance(model: nn.Module, graph: ChannelGraph, batches, eps: float=1e-12, anchor: bool=True) -> dict[str, torch.Tensor]:
    with _eval_mode(model):
        model.zero_grad(set_to_none=True)
        for xb, yb in batches:
            F.cross_entropy(model(xb), yb).backward()
        out = {}
        with torch.no_grad():
            for g in graph.prunable:
                t = torch.zeros(g.channels, dtype=torch.float64)
                for name in g.producers + g.norms:
                    m = model.get_submodule(name)
                    for prm in (m.weight, getattr(m, 'bias', None)):
                        if prm is not None and prm.grad is not None:
                            t += (prm * prm.grad).double().reshape(g.channels, -1).sum(1)
                for name, hw in g.consumers:
                    w = model.get_submodule(name).weight
                    if w.grad is not None:
                        t += (w * w.grad).double().transpose(0, 1).flatten(1).sum(1).view(g.channels, hw).sum(1)
                s = t.pow(2)
                out[g.key] = (s / (s.mean() + eps)).float()
        model.zero_grad(set_to_none=True)
    return anchor_importance(model, 'taylor', out) if anchor else out

def percentile_threshold(importance: dict[str, torch.Tensor], q: float) -> float:
    return float(np.percentile(torch.cat(list(importance.values())).numpy(), q))

def select_channels(graph: ChannelGraph, importance: dict[str, torch.Tensor], tau: float, orig: dict[str, int] | None=None, min_keep: Callable[[int], int]=default_min_keep) -> dict[str, torch.Tensor]:
    keep = {}
    for g in graph.prunable:
        s = importance[g.key]
        floor = min(g.channels, min_keep((orig or {}).get(g.key, g.channels)))
        idx = torch.nonzero(s >= tau).flatten()
        if len(idx) < floor:
            idx = torch.topk(s, floor).indices
        keep[g.key] = idx.sort().values
    return keep

def l1_keep(model: nn.Module, graph: ChannelGraph, ratio: float, include_coupled: bool=False, min_keep: Callable[[int], int]=default_min_keep) -> dict[str, torch.Tensor]:
    orig = (getattr(model, 'prune_meta', None) or {}).get('orig', {})
    keep = {}
    for g in graph.prunable:
        if g.coupled and (not include_coupled):
            continue
        s = sum((model.get_submodule(p).weight.detach().abs().flatten(1).sum(1) for p in g.producers))
        n = max(min(g.channels, min_keep(orig.get(g.key, g.channels))), math.ceil((1 - ratio) * g.channels))
        keep[g.key] = torch.topk(s, min(n, g.channels)).indices.sort().values
    return keep

def _dims(m: nn.Module) -> tuple[int, int]:
    if isinstance(m, nn.Conv2d):
        return (m.in_channels, m.out_channels)
    if isinstance(m, nn.Linear):
        return (m.in_features, m.out_features)
    return (m.num_features, m.num_features)

def _new_like(m: nn.Module, n_in: int, n_out: int) -> nn.Module:
    kw = dict(device=m.weight.device if m.weight is not None else None, dtype=m.weight.dtype if m.weight is not None else None)
    if isinstance(m, nn.Conv2d):
        new = nn.Conv2d(n_in, n_out, m.kernel_size, m.stride, m.padding, m.dilation, m.groups, m.bias is not None, m.padding_mode, **kw)
    elif isinstance(m, nn.Linear):
        new = nn.Linear(n_in, n_out, bias=m.bias is not None, **kw)
    else:
        new = type(m)(n_out, m.eps, m.momentum, m.affine, m.track_running_stats, **kw)
    new.train(m.training)
    for (_, a), (_, b) in zip(m.named_parameters(), new.named_parameters()):
        b.requires_grad_(a.requires_grad)
    return new

def _set_submodule(root: nn.Module, path: str, new: nn.Module) -> None:
    parent, _, name = path.rpartition('.')
    setattr(root.get_submodule(parent) if parent else root, name, new)

def _feat_idx(idx: torch.Tensor, hw: int) -> torch.Tensor:
    if hw == 1:
        return idx
    return (idx.view(-1, 1) * hw + torch.arange(hw).view(1, -1)).flatten()

@torch.no_grad()
def _sliced(m: nn.Module, out: torch.Tensor | None, inp: tuple[torch.Tensor, int] | None) -> nn.Module:
    if isinstance(m, _NORM_TYPES):
        new = _new_like(m, len(out), len(out))
        for k in ('weight', 'bias', 'running_mean', 'running_var'):
            src = getattr(m, k)
            if src is not None:
                getattr(new, k).copy_(src.index_select(0, out))
        if m.num_batches_tracked is not None:
            new.num_batches_tracked.copy_(m.num_batches_tracked)
        return new
    w = m.weight.detach()
    if out is not None:
        w = w.index_select(0, out)
    if inp is not None:
        w = w.index_select(1, _feat_idx(*inp))
    new = _new_like(m, w.shape[1] * (m.groups if isinstance(m, nn.Conv2d) else 1), w.shape[0])
    new.weight.copy_(w)
    if m.bias is not None:
        new.bias.copy_(m.bias.index_select(0, out) if out is not None else m.bias)
    return new

def _arch_name(model: nn.Module) -> str | None:
    return getattr(model, 'arch_name', type(model).__name__)

def _layer_dims(model: nn.Module) -> dict[str, dict]:
    return {n: dict(type=type(m).__name__, **dict(zip(('in', 'out'), _dims(m)))) for n, m in model.named_modules() if isinstance(m, _PRUNABLE_TYPES)}

def init_meta(model: nn.Module, graph: ChannelGraph | None=None, arch: str | None=None) -> dict:
    meta = getattr(model, 'prune_meta', None)
    if meta is None:
        graph = graph or build_graph(model)
        meta = dict(arch=arch or _arch_name(model), input_shape=list(graph.input_shape), orig={g.key: g.channels for g in graph.prunable}, orig_idx={g.key: list(range(g.channels)) for g in graph.prunable}, orig_layers=_layer_dims(model))
        model.prune_meta = meta
    return meta

def _index_maps(graph: ChannelGraph, keep: dict[str, torch.Tensor]):
    out_idx, in_idx = ({}, {})
    for g in graph.prunable:
        k = keep.get(g.key)
        if k is None or len(k) == g.channels:
            continue
        for p in g.producers + g.norms:
            out_idx[p] = k
        for q, hw in g.consumers:
            in_idx[q] = (k, hw)
    return (out_idx, in_idx)

def check_model(model: nn.Module, input_shape=None) -> ChannelGraph:
    graph = build_graph(model, input_shape)
    p = next(model.parameters())
    with _eval_mode(model), torch.no_grad():
        out = model(torch.zeros(graph.input_shape, dtype=p.dtype, device=p.device))
    assert torch.isfinite(out).all()
    return graph

def prune_model(model: nn.Module, keep: dict[str, torch.Tensor], graph: ChannelGraph | None=None, verify: bool=True) -> nn.Module:
    graph = graph or build_graph(model)
    meta = copy.deepcopy(init_meta(model, graph))
    new = copy.deepcopy(model)
    out_idx, in_idx = _index_maps(graph, keep)
    for path in dict.fromkeys(list(out_idx) + list(in_idx)):
        _set_submodule(new, path, _sliced(new.get_submodule(path), out_idx.get(path), in_idx.get(path)))
    for key, k in keep.items():
        if key in meta['orig_idx']:
            meta['orig_idx'][key] = [meta['orig_idx'][key][i] for i in k.tolist()]
    new.prune_meta = meta
    if verify:
        check_model(new, graph.input_shape)
    return new

def prune(model: nn.Module, tau: float, importance_fn=path_importance, min_keep: Callable[[int], int]=default_min_keep, input_shape=None):
    graph = build_graph(model, input_shape)
    meta = init_meta(model, graph)
    keep = select_channels(graph, importance_fn(model, graph), tau, meta['orig'], min_keep)
    removed = sum((g.channels - len(keep[g.key]) for g in graph.prunable))
    return (prune_model(model, keep, graph) if removed else copy.deepcopy(model), keep, removed)

def model_spec(model: nn.Module, graph: ChannelGraph | None=None) -> dict:
    graph = graph or build_graph(model)
    meta = init_meta(model, graph)
    layers = _layer_dims(model)
    for n, d in layers.items():
        o = meta['orig_layers'].get(n, d)
        d['orig_in'], d['orig_out'] = (o['in'], o['out'])
    groups = [dict(key=g.key, channels=g.channels, orig=meta['orig'].get(g.key, g.channels), coupled=g.coupled, frozen=g.frozen, reason=g.reason, producers=list(g.producers), norms=list(g.norms), consumers=[[q, hw] for q, hw in g.consumers]) for g in graph.groups]
    return dict(arch=meta['arch'], input_shape=list(graph.input_shape), layers=layers, groups=groups, meta=copy.deepcopy(meta))

def apply_spec(model: nn.Module, spec: dict) -> nn.Module:
    for name, d in spec['layers'].items():
        m = model.get_submodule(name)
        if _dims(m) != (d['in'], d['out']):
            _set_submodule(model, name, _new_like(m, d['in'], d['out']))
    model.prune_meta = copy.deepcopy(spec['meta'])
    return model

def model_from_spec(spec: dict, builder: Callable[[str], nn.Module]) -> nn.Module:
    return apply_spec(builder(spec['arch']), spec)

def save_pruned(model: nn.Module, path: str, extra: dict | None=None) -> dict:
    spec = model_spec(model)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    sd = {k: v.contiguous() for k, v in model.state_dict().items()}
    torch.save(dict(spec=spec, state_dict=sd, extra=extra or {}), path)
    with open(os.path.splitext(path)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(spec, f, indent=2)
    return spec

def load_pruned(path: str, builder: Callable[[str], nn.Module]) -> nn.Module:
    ck = torch.load(path, map_location='cpu')
    model = model_from_spec(ck['spec'], builder)
    model.load_state_dict(ck['state_dict'], strict=True)
    return model

def _orig_index_maps(graph: ChannelGraph, orig_idx: dict[str, list[int]]):
    keep = {g.key: torch.as_tensor(orig_idx[g.key], dtype=torch.long) for g in graph.prunable if g.key in orig_idx}
    return (_index_maps(graph, keep), keep)

@torch.no_grad()
def mask_channels(model: nn.Module, orig_idx: dict[str, list[int]], zero_consumers: bool=False) -> nn.Module:
    m = copy.deepcopy(model)
    graph = build_graph(m)
    _, keep = _orig_index_maps(graph, orig_idx)
    for g in graph.prunable:
        if g.key not in keep:
            continue
        drop = torch.ones(g.channels, dtype=torch.bool)
        drop[keep[g.key]] = False
        for name in g.producers + g.norms:
            mod = m.get_submodule(name)
            if mod.weight is not None:
                mod.weight[drop] = 0
                if mod.bias is not None:
                    mod.bias[drop] = 0
            elif getattr(mod, 'running_mean', None) is not None:
                mod.running_mean[drop] = 0
        if zero_consumers:
            for q, hw in g.consumers:
                m.get_submodule(q).weight[:, _feat_idx(torch.nonzero(drop).flatten(), hw)] = 0
    return m

@torch.no_grad()
def expand_pruned(pruned: nn.Module, template: nn.Module) -> nn.Module:
    full = copy.deepcopy(template)
    graph = build_graph(full)
    (out_idx, in_idx), _ = _orig_index_maps(graph, pruned.prune_meta['orig_idx'])
    psd = pruned.state_dict()
    fsd = full.state_dict()
    fsd.update({k: v for k, v in psd.items() if k in fsd and fsd[k].shape == v.shape})
    full.load_state_dict(fsd)
    for name, mod in full.named_modules():
        if not isinstance(mod, _PRUNABLE_TYPES) or (name not in out_idx and name not in in_idx):
            continue
        src = pruned.get_submodule(name)
        n_in, n_out = _dims(mod)
        rows = out_idx.get(name, torch.arange(n_out))
        if isinstance(mod, _NORM_TYPES):
            for k, fill in (('weight', 0.0), ('bias', 0.0), ('running_mean', 0.0), ('running_var', 1.0)):
                t = getattr(mod, k)
                if t is not None:
                    t.fill_(fill)
                    t[rows] = getattr(src, k)
            continue
        cols = _feat_idx(*in_idx[name]) if name in in_idx else torch.arange(mod.weight.shape[1])
        mod.weight.zero_()
        mod.weight[rows.view(-1, 1), cols.view(1, -1)] = src.weight
        if mod.bias is not None:
            mod.bias.zero_()
            mod.bias[rows] = src.bias
    full.prune_meta = copy.deepcopy(pruned.prune_meta)
    return full

def params_nonzero(model: nn.Module) -> int:
    return int(sum((int((p != 0).sum()) for p in model.parameters())))

def kept_channels(graph: ChannelGraph) -> int:
    return int(sum((g.channels for g in graph.prunable)))
