from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_mae_mask(
    batch_size: int,
    seq_len: int,
    mask_ratio: float,
    device: torch.device,
) -> torch.Tensor:
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1).")
    n_mask = max(1, int(round(seq_len * mask_ratio)))
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for b in range(batch_size):
        idx = torch.randperm(seq_len, device=device)[:n_mask]
        mask[b, idx] = True
    return mask


def _sinusoidal_positional_embedding(max_len: int, dim: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(max_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / dim))
    pe = torch.zeros(max_len, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class AudioPatchEmbedding(nn.Module):
    def __init__(
        self,
        n_mels: int,
        dim: int,
        patch_size: int,
        patch_freq: Optional[int] = None,
        patch_strategy: str = "time",
    ) -> None:
        super().__init__()
        self.n_mels = int(n_mels)
        self.dim = int(dim)
        self.patch_time = int(patch_size)
        self.patch_strategy = patch_strategy
        self.patch_freq = int(patch_freq if patch_freq is not None else n_mels)
        if self.patch_strategy == "time":
            self.patch_freq = self.n_mels
        self.patch_dim = self.patch_time * self.patch_freq
        self.proj = nn.Linear(self.patch_dim, dim)

    def patchify(self, x_logmel: torch.Tensor) -> Dict[str, Any]:
        if x_logmel.ndim != 3:
            raise ValueError("x_logmel must be [B, T, M].")
        bsz, t_len, n_mels = x_logmel.shape
        if n_mels != self.n_mels:
            raise ValueError(f"Expected n_mels={self.n_mels}, got {n_mels}.")

        pad_t = (self.patch_time - (t_len % self.patch_time)) % self.patch_time
        pad_f = (self.patch_freq - (n_mels % self.patch_freq)) % self.patch_freq
        x_pad = x_logmel
        if pad_t > 0 or pad_f > 0:
            x_pad = F.pad(x_pad, (0, pad_f, 0, pad_t), mode="constant", value=0.0)

        _, t_pad, f_pad = x_pad.shape
        n_time = t_pad // self.patch_time
        n_freq = f_pad // self.patch_freq

        patches = (
            x_pad.unsqueeze(1)
            .unfold(2, self.patch_time, self.patch_time)
            .unfold(3, self.patch_freq, self.patch_freq)
            .contiguous()
            .view(bsz, n_time * n_freq, self.patch_dim)
        )
        return {
            "patches": patches,
            "n_time": n_time,
            "n_freq": n_freq,
            "pad_t": pad_t,
            "pad_f": pad_f,
            "t_original": t_len,
        }

    def forward(self, x_logmel: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        patch_info = self.patchify(x_logmel)
        tokens = self.proj(patch_info["patches"])
        return tokens, patch_info


class AudioTransformerEncoder(nn.Module):
    def __init__(
        self,
        n_mels: int,
        dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        patch_size: int,
        max_len: int,
        pos_embed: str,
        patch_strategy: str = "time",
        patch_freq: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.max_len = int(max_len)
        self.pos_embed_type = str(pos_embed)
        self.patch_embedding = AudioPatchEmbedding(
            n_mels=n_mels,
            dim=dim,
            patch_size=patch_size,
            patch_freq=patch_freq,
            patch_strategy=patch_strategy,
        )
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, mean=0.0, std=0.02)

        ffn_dim = int(dim * float(mlp_ratio))
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)
        if self.pos_embed_type == "learned":
            self.pos_embedding = nn.Embedding(self.max_len, dim)
        else:
            self.register_buffer("sinusoidal_cache", torch.empty(0), persistent=False)

    def _position_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len={self.max_len}.")
        if self.pos_embed_type == "learned":
            idx = torch.arange(seq_len, device=device, dtype=torch.long)
            return self.pos_embedding(idx).unsqueeze(0)
        if self.pos_embed_type == "sinusoidal":
            if self.sinusoidal_cache.numel() == 0 or self.sinusoidal_cache.shape[0] < self.max_len:
                self.sinusoidal_cache = _sinusoidal_positional_embedding(self.max_len, self.dim, device)
            return self.sinusoidal_cache[:seq_len].unsqueeze(0)
        return torch.zeros((1, seq_len, self.dim), device=device)

    def forward(
        self,
        x_logmel: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        tokens, patch_info = self.patch_embedding(x_logmel)
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError("token_mask shape mismatch.")
            mask_tok = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
            tokens = torch.where(token_mask.unsqueeze(-1), mask_tok, tokens)
        seq_len = tokens.shape[1]
        tokens = tokens + self._position_embedding(seq_len, tokens.device)
        encoded = self.encoder(tokens)
        encoded = self.norm(encoded)
        return encoded, patch_info


class AudioMAEPretrain(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        n_mels: int,
        dec_dim: int,
        dec_depth: int,
        dec_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.n_mels = int(n_mels)
        self.enc_to_dec = nn.Linear(encoder.dim, dec_dim)
        dec_layer = nn.TransformerEncoderLayer(
            d_model=dec_dim,
            nhead=dec_heads,
            dim_feedforward=dec_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_depth)
        self.dec_norm = nn.LayerNorm(dec_dim)
        self.decoder_pred = nn.Linear(dec_dim, encoder.patch_embedding.patch_dim)

    def forward(
        self,
        x_logmel: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        encoded, patch_info = self.encoder(x_logmel, token_mask=mask)
        dec_tokens = self.enc_to_dec(encoded)
        dec_out = self.dec_norm(self.decoder(dec_tokens))
        pred = self.decoder_pred(dec_out)
        target = patch_info["patches"]

        if mask.any():
            loss = F.mse_loss(pred[mask], target[mask])
        else:
            loss = F.mse_loss(pred, target)
        return loss


class AudioTransformerCTC(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.dim, int(vocab_size))
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        x_logmel: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, patch_info = self.encoder(x_logmel, token_mask=None)
        bsz = encoded.shape[0]
        n_time = patch_info["n_time"]
        n_freq = patch_info["n_freq"]
        if n_freq > 1:
            encoded = encoded.view(bsz, n_time, n_freq, -1).mean(dim=2)
        logits = self.classifier(self.dropout(encoded))
        patch_time = self.encoder.patch_embedding.patch_time
        out_lengths = torch.div(lengths + (patch_time - 1), patch_time, rounding_mode="floor")
        out_lengths = torch.clamp(out_lengths, max=logits.shape[1])
        return logits, out_lengths

