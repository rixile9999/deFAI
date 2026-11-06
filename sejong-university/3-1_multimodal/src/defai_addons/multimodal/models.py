from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .data import MODALITIES, N_CLASSES
from .encoders import build_encoders, masked_mean
from .fusion import GatedMultimodalFusion, HierarchicalFeatureFusion
from .generation import AutoregressiveSequenceDecoder
from .joint import JointProjection, contrastive_alignment, joint_consistency
from .transformer import CrossModalTransformerDecoder, ModalityEmbeddingIntegration, MultimodalTransformerEncoder


class _Base(nn.Module):
    def __init__(self, d: int = 64, d_joint: int = 32, n_classes: int = N_CLASSES, lambda_align: float = 0.1,
                 lambda_cons: float = 0.0):
        super().__init__()
        self.d = d
        self.encoders = build_encoders(d)
        self.joint = JointProjection(d, d_joint)
        self.lambda_align, self.lambda_cons = lambda_align, lambda_cons
        self.n_classes = n_classes

    def encode(self, batch: Dict[str, torch.Tensor]):
        mask = batch["mask"]
        toks, pads, lows, pooled, emb = [], [], [], [], {}
        for k, m in enumerate(MODALITIES):
            t, p, low = self.encoders[m](batch[m])
            t = self.joint.tokens(m, t)
            pv = masked_mean(t, p)
            keep = mask[:, k].float().unsqueeze(-1)
            toks.append(t * keep.unsqueeze(1))
            pads.append(p)
            lows.append(low * keep)
            pooled.append(pv * keep)
            emb[m] = self.joint.embed(m, pv)
        return toks, pads, torch.stack(lows, 1), torch.stack(pooled, 1), emb

    def aux_loss(self, emb, mask, feats=None, fused=None):
        loss = self.lambda_align * contrastive_alignment(emb, mask)
        if self.lambda_cons > 0 and feats is not None:
            loss = loss + self.lambda_cons * joint_consistency(feats, fused, mask)
        return loss


class MultimodalTransformer(_Base):
    def __init__(self, d: int = 64, n_heads: int = 4, enc_layers: int = 2, dec_layers: int = 1,
                 use_integration: bool = True, use_decoder: bool = True, fusion: str = "gmu", use_hff: bool = True,
                 dropout: float = 0.1, sequence_vocab: Optional[int] = None, sequence_max_len: int = 128, **kw):
        super().__init__(d, **kw)
        self.integration = ModalityEmbeddingIntegration(d, len(MODALITIES), learnable_weights=use_integration)
        self.encoder = MultimodalTransformerEncoder(d, enc_layers, n_heads, dropout=dropout)
        self.decoder = CrossModalTransformerDecoder(d, dec_layers, n_heads, dropout=dropout) if use_decoder else None
        self.fusion_kind = fusion
        self.gmu = GatedMultimodalFusion(d) if fusion == "gmu" else None
        self.hff = HierarchicalFeatureFusion(d) if use_hff else None
        self.head = nn.Sequential(nn.LayerNorm((3 if use_hff else 2) * d), nn.Linear((3 if use_hff else 2) * d, d),
                                  nn.GELU(), nn.Dropout(dropout), nn.Linear(d, self.n_classes))
        self.sequence_decoder = (AutoregressiveSequenceDecoder(d, sequence_vocab, n_heads, dec_layers,
                                                                sequence_max_len, dropout)
                                 if sequence_vocab is not None else None)

    def forward(self, batch, decoder_input_ids=None):
        mask = batch["mask"]
        toks, pads, low, pooled, emb = self.encode(batch)
        seq, pad, lens, alpha = self.integration(toks, pads, pooled, mask)
        H = self.encoder(seq, pad)
        fuse_tok = H[:, 0]
        if self.decoder is not None:
            outs, cross = self.decoder(H, pad, lens)
            mpads = []
            s = 1
            for L in lens:
                mpads.append(pad[:, s:s + L])
                s += L
            feats = torch.stack([masked_mean(o, p) for o, p in zip(outs, mpads)], 1)
        else:
            cross, feats, s = None, [], 1
            for L in lens:
                feats.append(masked_mean(H[:, s:s + L], pad[:, s:s + L]))
                s += L
            feats = torch.stack(feats, 1)
        feats = feats * mask.unsqueeze(-1)
        if self.gmu is not None:
            fused, gate = self.gmu(feats, mask)
        else:
            fused = feats.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            gate = mask.float() / mask.sum(1, keepdim=True).clamp(min=1)
        rep = [fused, fuse_tok]
        if self.hff is not None:
            rep.append(self.hff(low, pooled, mask))
        representation = torch.cat(rep, -1)
        logits = self.head(representation)
        result = {"logits": logits, "probabilities": logits.softmax(-1), "emb": emb, "aux": self.aux_loss(emb, mask),
                  "representation": representation, "encoder_memory": H, "encoder_padding_mask": pad,
                  "attn": {"integration_weight": alpha, "gate": gate, "cross_attn": cross}}
        if decoder_input_ids is not None:
            if self.sequence_decoder is None:
                raise ValueError("Set sequence_vocab to enable sequence prediction")
            result["sequence_logits"] = self.sequence_decoder(decoder_input_ids, H, pad)
        return result

    @torch.no_grad()
    def generate(self, batch, bos_id, eos_id, max_new_tokens=32):
        if self.sequence_decoder is None:
            raise ValueError("Set sequence_vocab to enable sequence generation")
        was_training = self.training
        self.eval()
        try:
            out = self(batch)
            return self.sequence_decoder.generate(out["encoder_memory"], out["encoder_padding_mask"],
                                                   bos_id, eos_id, max_new_tokens)
        finally:
            self.train(was_training)


class UnimodalModel(MultimodalTransformer):
    def __init__(self, modality: str, d: int = 64, **kw):
        super().__init__(d, **kw)
        self.only = MODALITIES.index(modality)

    def forward(self, batch):
        b = dict(batch)
        m = torch.zeros_like(batch["mask"])
        m[:, self.only] = True
        b["mask"] = m
        return super().forward(b)


MODEL_ZOO = {
    "y3_mm_transformer": lambda **kw: MultimodalTransformer(**kw),
    "y3_no_integration": lambda **kw: MultimodalTransformer(use_integration=False, **kw),
    "y3_no_decoder": lambda **kw: MultimodalTransformer(use_decoder=False, **kw),
    "y3_mean_fusion": lambda **kw: MultimodalTransformer(fusion="mean", **kw),
    "image_only": lambda **kw: UnimodalModel("image", **kw),
    "text_only": lambda **kw: UnimodalModel("text", **kw),
    "audio_only": lambda **kw: UnimodalModel("audio", **kw),
}


def build_model(name: str, **kw) -> nn.Module:
    if name in ("y1_joint_embedding", "y2_attention_fusion"):
        from .baselines.models import JointEmbeddingModel, AttentionFusionModel
        return {"y1_joint_embedding": JointEmbeddingModel, "y2_attention_fusion": AttentionFusionModel}[name](**kw)
    return MODEL_ZOO[name](**kw)


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def __getattr__(name):
    if name in ("JointEmbeddingModel", "AttentionFusionModel"):
        from .baselines import models as historical
        return getattr(historical, name)
    raise AttributeError(name)
