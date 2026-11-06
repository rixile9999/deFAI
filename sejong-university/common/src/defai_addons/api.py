from .integration.history import read_history, render_histories


def create_dashboard(store_dir=None):
    from .monitor.server import create_app
    return create_app(store_dir=store_dir)


def compare(config=None, progress=None):
    from .compare.runner import run_comparison
    return run_comparison(config or {}, progress=progress)


def compare_arrays(sites, X_test, y_test, config=None, progress=None):
    import numpy as np
    from .compare.envs import SiteData
    from .compare.runner import compare_sites
    if not sites:
        raise ValueError('At least one site is required')
    def pair(X, y):
        X, y = np.asarray(X, dtype=np.float32), np.asarray(y)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or not len(y):
            raise ValueError('Features must be nonempty [N,D] and labels [N]')
        if y.dtype.kind not in 'iu' or np.min(y) < 0:
            raise ValueError('Labels must be nonnegative integer class indices')
        return X.copy(), y.astype(np.int64, copy=True)
    X_test, y_test = pair(X_test, y_test)
    parts = []
    for s in sites:
        X, y = pair(s['X'], s['y'])
        Xt, yt = pair(s['X_test'], s['y_test'])
        if X.shape[1] != X_test.shape[1] or Xt.shape[1] != X_test.shape[1]:
            raise ValueError('All sites must share the same feature dimension')
        parts.append(SiteData(X, y, Xt, yt))
    return compare_sites(parts, X_test, y_test, config, progress)


def create_multimodal_model(**config):
    from .multimodal.models import MultimodalTransformer
    return MultimodalTransformer(**config)


def save_multimodal_checkpoint(model, path, model_config):
    import torch
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as f:
        torch.save({'format': 'defai-addons.multimodal.v1', 'model_config': model_config,
                    'state_dict': model.state_dict()}, f)


def load_multimodal_checkpoint(path, device='cpu'):
    import torch
    data = torch.load(path, map_location=device, weights_only=True)
    if data.get('format') != 'defai-addons.multimodal.v1':
        raise ValueError('Unsupported checkpoint format')
    model = create_multimodal_model(**data['model_config']).to(device)
    model.load_state_dict(data['state_dict'], strict=True)
    return model.eval()


def predict_multimodal(model, batch):
    import torch
    from .multimodal.data import VOCAB, SEQ_LEN, AUDIO_LEN
    device = next(model.parameters()).device
    tensors = {k: torch.as_tensor(batch[k], device=device) for k in ('image', 'text', 'audio', 'mask')}
    n = len(tensors['mask'])
    shapes = {'image': (n, 1, 28, 28), 'text': (n, SEQ_LEN), 'audio': (n, 1, AUDIO_LEN), 'mask': (n, 3)}
    if not n or any(tuple(tensors[k].shape) != shape for k, shape in shapes.items()):
        raise ValueError('Expected image[N,1,28,28], text[N,16], audio[N,1,800], mask[N,3]')
    if tensors['mask'].dtype != torch.bool or not tensors['mask'].any(dim=1).all():
        raise ValueError('mask must be boolean with at least one present modality per sample')
    if tensors['text'].dtype not in (torch.int32, torch.int64) or tensors['text'].min() < 0 or tensors['text'].max() >= VOCAB:
        raise ValueError('Text must contain integer token IDs in the model vocabulary')
    tensors['text'] = tensors['text'].long()
    for k in ('image', 'audio'):
        tensors[k] = tensors[k].float()
        if not torch.isfinite(tensors[k]).all():
            raise ValueError(k + ' contains nonfinite values')
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            result = model(tensors)
            return {k: result[k].detach().cpu() for k in ('logits', 'probabilities', 'representation')}
    finally:
        model.train(training)
