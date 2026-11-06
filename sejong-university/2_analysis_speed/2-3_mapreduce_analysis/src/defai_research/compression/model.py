from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import numpy as np
import torch
from torch import nn

ARTIFACT_HEADER = struct.Struct('<4sIQ')

def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class TransformerCodecConfig:
    block_size: int = 256
    patch_size: int = 16
    d_model: int = 8
    nhead: int = 2
    encoder_layers: int = 1
    decoder_layers: int = 1
    latent_tokens: int = 2
    latent_dim: int = 8
    feedforward_dim: int = 16
    epochs: int = 60
    batch_size: int = 64
    learning_rate: float = .001
    patience: int = 12
    seed: int = 17001
    validation_seed: int = 14003

    def validate(self):
        for name in ("block_size","patch_size","d_model","nhead","encoder_layers",
                     "decoder_layers","latent_tokens","latent_dim","feedforward_dim",
                     "epochs","batch_size","seed","validation_seed"):
            _integer(getattr(self,name),name,1 if name not in {"seed","validation_seed"} else 0)
        _integer(self.patience,"patience",0)
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.block_size != 256 or self.patch_size != 16:
            raise ValueError("The declared wire schema requires block256 and patch16")
        if self.d_model % self.nhead or self.d_model < 2:
            raise ValueError("d_model must be divisible by nhead")
        if self.latent_tokens not in (2, 4) or self.latent_dim not in (8, 16):
            raise ValueError("Declared bottlenecks are 2/4 tokens and 8/16 latent dimensions")
        if min(self.encoder_layers, self.decoder_layers, self.epochs, self.batch_size) < 1:
            raise ValueError("Layer, epoch and batch counts must be positive")


class TransformerBlockAutoencoder(nn.Module):
    def __init__(self, config: TransformerCodecConfig):
        super().__init__()
        config.validate()
        self.config = config
        tokens = config.block_size // config.patch_size
        self.input_projection = nn.Linear(config.patch_size, config.d_model)
        positions = torch.arange(tokens, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, config.d_model, 2, dtype=torch.float32)
                                * (-math.log(10000.) / config.d_model))
        positional = torch.zeros(tokens, config.d_model)
        positional[:, 0::2] = torch.sin(positions * frequencies)
        positional[:, 1::2] = torch.cos(positions * frequencies[:positional[:, 1::2].shape[1]])
        self.register_buffer("positional_encoding", positional)
        layer = nn.TransformerEncoderLayer(
            config.d_model, config.nhead, config.feedforward_dim,
            dropout=0., batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, config.encoder_layers,
                                             enable_nested_tensor=False)
        self.latent_queries = nn.Parameter(torch.randn(config.latent_tokens, config.d_model) * .02)
        self.latent_attention = nn.MultiheadAttention(config.d_model, config.nhead,
                                                     dropout=0., batch_first=True)
        self.latent_projection = nn.Linear(config.d_model, config.latent_dim)
        self.memory_projection = nn.Linear(config.latent_dim, config.d_model)
        self.decoder_queries = nn.Parameter(torch.randn(tokens, config.d_model) * .02)
        decoder_layer = nn.TransformerDecoderLayer(
            config.d_model, config.nhead, config.feedforward_dim,
            dropout=0., batch_first=True, activation="gelu")
        self.decoder = nn.TransformerDecoder(decoder_layer, config.decoder_layers)
        self.output_projection = nn.Linear(config.d_model, config.patch_size)

    def encode(self, blocks):
        patches = blocks.reshape(-1, self.config.block_size // self.config.patch_size,
                                 self.config.patch_size)
        encoded = self.encoder(self.input_projection(patches) + self.positional_encoding)
        queries = self.latent_queries.unsqueeze(0).expand(len(blocks), -1, -1)
        pooled, _ = self.latent_attention(queries, encoded, encoded, need_weights=False)
        return self.latent_projection(pooled)

    def decode(self, latent):
        memory = self.memory_projection(latent)
        target = (self.decoder_queries + self.positional_encoding).unsqueeze(0).expand(len(latent), -1, -1)
        return self.output_projection(self.decoder(target, memory)).reshape(-1, self.config.block_size)

    def forward(self, blocks):
        latent = self.encode(blocks)
        scales = latent.detach().abs().amax(dim=(1, 2), keepdim=True).clamp_min(1e-12) / 127
        quantized = torch.clamp(torch.round(latent / scales), -127, 127) * scales
        return self.decode(latent + (quantized - latent).detach())


def _normalize(blocks, valid_lengths=None):
    blocks = np.asarray(blocks, dtype=np.float32)
    if blocks.ndim != 2 or blocks.shape[1] != 256 or not np.isfinite(blocks).all():
        raise ValueError("Calibration blocks must be finite float arrays of shape [N,256]")
    lengths = (np.full(len(blocks), 256, np.int64) if valid_lengths is None
               else np.asarray(valid_lengths, dtype=np.int64))
    if lengths.shape != (len(blocks),) or np.any((lengths < 1) | (lengths > 256)):
        raise ValueError("valid_lengths must contain one value in [1,256] per block")
    valid = np.arange(256)[None, :] < lengths[:, None]
    means = (np.where(valid, blocks, 0).sum(1, dtype=np.float64) / lengths).astype(np.float32)
    centered = np.where(valid, blocks - means[:, None], 0)
    scales = np.sqrt((centered.astype(np.float64)**2).sum(1) / lengths).astype(np.float32)
    scales[scales < 1e-12] = 0
    normalized = np.where(valid, centered / np.where(scales > 0, scales, 1)[:, None], 0).astype(np.float32)
    return normalized, means, scales, valid


def _artifact_bytes(model, config, metadata):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: value.detach().cpu().numpy()
                                  for key, value in model.state_dict().items()})
    weights = buffer.getvalue()
    document = _json(dict(schema=1, config=asdict(config), metadata=metadata,
                          weights_sha256=_sha(weights), torch_version=torch.__version__,
                          parameter_count=sum(p.numel() for p in model.parameters()),
                          architecture="TransformerEncoder + learned-query MHA latent bottleneck + TransformerDecoder"))
    return ARTIFACT_HEADER.pack(b"TCA1", len(document), len(weights)) + document + weights


def _read_artifact(payload):
    if not isinstance(payload, bytes) or len(payload) < ARTIFACT_HEADER.size:
        raise ValueError("A complete immutable artifact buffer is required")
    magic, json_size, weights_size = ARTIFACT_HEADER.unpack_from(payload)
    if magic != b"TCA1" or len(payload) != ARTIFACT_HEADER.size + json_size + weights_size:
        raise ValueError("Transformer artifact length or magic mismatch")
    end = ARTIFACT_HEADER.size + json_size
    document = json.loads(payload[ARTIFACT_HEADER.size:end])
    weights = payload[end:]
    if _sha(weights) != document["weights_sha256"]:
        raise ValueError("Transformer artifact weight checksum mismatch")
    config = TransformerCodecConfig(**document["config"])
    model = TransformerBlockAutoencoder(config)
    with np.load(io.BytesIO(weights), allow_pickle=False) as arrays:
        state = {key: torch.from_numpy(arrays[key].copy()) for key in arrays.files}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, document


def fit_transformer_codec(calibration, output_dir, *, config=None, manifest=None,
                          validation_mask=None, verbose=False):
    config = config or TransformerCodecConfig()
    config.validate()
    manifest = dict(manifest or {})
    if manifest.get("test_data_used", False):
        raise ValueError("Test data cannot be used for codec calibration")
    path = Path(calibration) if isinstance(calibration, (str, Path)) else None
    lengths = seeds = None
    if path is not None:
        with np.load(path, allow_pickle=False) as source:
            blocks = np.asarray(source["blocks"], np.float32)
            if "valid_lengths" in source:
                lengths = source["valid_lengths"].copy()
            for name in ("seed", "seeds"):
                if name in source:
                    seeds = source[name].copy()
                    break
        calibration_sha = _sha(path.read_bytes())
    else:
        blocks = np.asarray(calibration, np.float32)
        calibration_sha = _sha(blocks.tobytes())
    expected_sha = manifest.get("npz_sha256")
    if expected_sha is not None and expected_sha != calibration_sha:
        raise ValueError("Calibration content differs from the supplied manifest checksum")
    if "status" in manifest and manifest["status"] != "completed":
        raise ValueError("A supplied calibration manifest must describe a completed collection")
    normalized, _, _, valid = _normalize(blocks, lengths)
    if validation_mask is None:
        if seeds is None or np.asarray(seeds).shape != (len(blocks),):
            raise ValueError("Supply per-block calibration seed or an explicit train-only validation_mask")
        validation_mask = np.asarray(seeds) == config.validation_seed
    validation_mask = np.asarray(validation_mask, bool)
    if validation_mask.shape != (len(blocks),) or not validation_mask.any() or validation_mask.all():
        raise ValueError("Calibration must have nonempty disjoint optimization and held-out subsets")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    artifact_path = out / "codec_model.bin"
    if artifact_path.exists():
        raise FileExistsError("Preserve the existing immutable codec artifact; choose another output directory")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = TransformerBlockAutoencoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    x = torch.from_numpy(normalized)
    mask = torch.from_numpy(valid.astype(np.float32))
    train_index = np.flatnonzero(~validation_mask)
    validation_index = np.flatnonzero(validation_mask)
    rng = np.random.default_rng(config.seed)
    model.eval()
    initial_loss = initial_weight = 0.
    with torch.inference_mode():
        for start in range(0, len(validation_index), config.batch_size):
            index = validation_index[start:start + config.batch_size]
            initial_loss += float((((model(x[index]) - x[index])**2) * mask[index]).sum())
            initial_weight += float(mask[index].sum())
    initial_loss /= initial_weight
    history = []
    best, best_epoch, best_state, stale = float("inf"), 0, None, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        loss_sum = weight_sum = 0.
        for start in range(0, len(train_index), config.batch_size):
            if start == 0:
                order = rng.permutation(train_index)
            index = order[start:start + config.batch_size]
            prediction = model(x[index])
            loss = (((prediction - x[index])**2) * mask[index]).sum() / mask[index].sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(mask[index].sum())
            weight_sum += float(mask[index].sum())
        model.eval()
        val_loss = val_weight = 0.
        with torch.inference_mode():
            for start in range(0, len(validation_index), config.batch_size):
                index = validation_index[start:start + config.batch_size]
                prediction = model(x[index])
                val_loss += float((((prediction - x[index])**2) * mask[index]).sum())
                val_weight += float(mask[index].sum())
        val_loss /= val_weight
        history.append(dict(epoch=epoch, normalized_train_MSE=loss_sum/weight_sum,
                            normalized_heldout_calibration_MSE=val_loss))
        if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == config.epochs):
            print(f"Transformer {config.latent_tokens}x{config.latent_dim} epoch {epoch}/{config.epochs} "
                  f"trainMSE={loss_sum/weight_sum:.6f} heldoutMSE={val_loss:.6f}",flush=True)
        if val_loss < best:
            best, best_epoch = val_loss, epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if config.patience and stale >= config.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    metadata = dict(calibration_sha256=calibration_sha, calibration_path=str(path) if path else None,
                    training_blocks=len(train_index), heldout_calibration_blocks=len(validation_index),
                    optimization_seeds=sorted(set(np.asarray(seeds)[~validation_mask].tolist())) if seeds is not None else None,
                    heldout_calibration_seeds=sorted(set(np.asarray(seeds)[validation_mask].tolist())) if seeds is not None else None,
                    selected_epoch=best_epoch, best_heldout_calibration_MSE=best,
                    initial_heldout_calibration_MSE=initial_loss,
                    heldout_MSE_reduction_fraction=1-best/initial_loss if initial_loss else 0.,
                    manifest=manifest, test_data_used=False,
                    codec_source_sha256=_sha(Path(__file__).read_bytes()),
                    quantization_aware_training=True, no_classifier_or_test_outcome_used=True)
    artifact = _artifact_bytes(model, config, metadata)
    restored, document = _read_artifact(artifact)
    for key, value in model.state_dict().items():
        if not torch.equal(value, restored.state_dict()[key]):
            raise AssertionError("Serialized FP32 model artifact changed learned weights")
    artifact_path.write_bytes(artifact)
    (out / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    result = dict(artifact_path=str(artifact_path.resolve()), model_id=_sha(artifact),
                  artifact_bytes=len(artifact), parameter_count=document["parameter_count"],
                  config=asdict(config), **metadata)
    (out / "artifact_manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _endpoint_pair(channel):
    if not isinstance(channel, (tuple, list)) or len(channel) != 4:
        raise ValueError("Channel must be (hop,direction,sender,receiver)")
    hop, direction, sender, receiver = channel
    if direction != "up":
        raise ValueError("This codec is declared only for model-update uplinks")
    if hop == "edge":
        return ("client", _integer(sender,"client_id")), ("group", _integer(receiver,"group_id"))
    if hop == "cloud" and receiver == "central":
        return ("group", _integer(sender,"group_id")), ("central", 0)
    raise ValueError("Unsupported update channel")


def export_inference_artifact(source, destination=None):
    source = Path(source)
    payload = source.read_bytes()
    _, document = _read_artifact(payload)
    magic, document_size, weights_size = ARTIFACT_HEADER.unpack_from(payload)
    weights = payload[ARTIFACT_HEADER.size + document_size:]
    document['metadata'] = dict(training_artifact_sha256=_sha(payload),
                                export_scope='Unchanged FP32 weights; provenance retained in training artifact')
    encoded = _json(document)
    result = ARTIFACT_HEADER.pack(magic, len(encoded), weights_size) + encoded + weights
    _read_artifact(result)
    target = Path(destination) if destination is not None else source.parent/'inference'/'codec_model.bin'
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != result:
        raise FileExistsError('Preserve existing inference artifact; choose another destination')
    target.write_bytes(result)
    return dict(artifact_path=str(target.resolve()), model_id=_sha(result),
                unchanged_weights_sha256=_sha(weights), weights_byte_identical=True,
                parameter_count=document['parameter_count'], artifact_bytes=len(result))
