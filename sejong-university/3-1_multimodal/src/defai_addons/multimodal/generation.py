import torch
import torch.nn as nn

from .encoders import SinusoidalPE


class AutoregressiveSequenceDecoder(nn.Module):
    def __init__(self, d, vocab_size, n_heads=4, n_layers=1, max_len=128, dropout=0.1, pad_id=0):
        super().__init__()
        if vocab_size < 2 or max_len < 1 or not 0 <= pad_id < vocab_size:
            raise ValueError("Invalid vocabulary, padding id or sequence length")
        self.vocab_size, self.max_len, self.pad_id = vocab_size, max_len, pad_id
        self.embedding = nn.Embedding(vocab_size, d, padding_idx=pad_id)
        self.position = SinusoidalPE(d, max_len)
        layer = nn.TransformerDecoderLayer(d, n_heads, 4 * d, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(layer, n_layers, norm=nn.LayerNorm(d))
        self.output = nn.Linear(d, vocab_size)

    def forward(self, input_ids, memory, memory_pad):
        if input_ids.ndim != 2 or not 1 <= input_ids.size(1) <= self.max_len:
            raise ValueError("Expected a nonempty decoder sequence within max_len")
        if (input_ids[:, 0] == self.pad_id).any():
            raise ValueError("Decoder sequences must start with a non-padding BOS token")
        length = input_ids.size(1)
        causal = torch.ones(length, length, dtype=torch.bool, device=input_ids.device).triu(1)
        hidden = self.decoder(self.position(self.embedding(input_ids)), memory,
                              tgt_mask=causal, tgt_key_padding_mask=input_ids.eq(self.pad_id),
                              memory_key_padding_mask=memory_pad)
        return self.output(hidden)

    @torch.no_grad()
    def generate(self, memory, memory_pad, bos_id, eos_id, max_new_tokens=32):
        if not (0 <= bos_id < self.vocab_size and 0 <= eos_id < self.vocab_size):
            raise ValueError("BOS/EOS must be in the vocabulary")
        if bos_id == self.pad_id or eos_id == self.pad_id or bos_id == eos_id:
            raise ValueError("BOS, EOS and padding must differ")
        if not 1 <= max_new_tokens <= self.max_len:
            raise ValueError("max_new_tokens must be within decoder max_len")
        tokens = torch.full((memory.size(0), 1), bos_id, dtype=torch.long, device=memory.device)
        done = torch.zeros(memory.size(0), dtype=torch.bool, device=memory.device)
        for _ in range(max_new_tokens):
            logits = self(tokens, memory, memory_pad)[:, -1].clone()
            logits[:, self.pad_id] = -float("inf")
            logits[:, bos_id] = -float("inf")
            nxt = logits.argmax(-1).masked_fill(done, self.pad_id)
            tokens = torch.cat((tokens, nxt[:, None]), 1)
            done = done | nxt.eq(eos_id)
            if done.all():
                break
        return tokens[:, 1:]
