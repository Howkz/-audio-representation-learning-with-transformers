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
    valid_token_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not (0.0 < mask_ratio < 1.0):
        raise ValueError("mask_ratio must be in (0, 1).")
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=device)
    for b in range(batch_size):
        if valid_token_mask is not None:
            valid_idx = torch.nonzero(valid_token_mask[b], as_tuple=False).flatten()
            if valid_idx.numel() == 0:
                continue
            n_mask = max(1, int(round(valid_idx.numel() * mask_ratio)))
            n_mask = min(n_mask, int(valid_idx.numel()))
            perm = torch.randperm(valid_idx.numel(), device=device)[:n_mask]
            idx = valid_idx[perm]
        else:
            n_mask = max(1, int(round(seq_len * mask_ratio)))
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

    def patchify(
        self,
        x_logmel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
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
            **self._patch_lengths(
                lengths=lengths,
                n_time=n_time,
                n_freq=n_freq,
                seq_len=n_time * n_freq,
                device=x_pad.device,
            ),
        }

    def _patch_lengths(
        self,
        lengths: Optional[torch.Tensor],
        n_time: int,
        n_freq: int,
        seq_len: int,
        device: torch.device,
    ) -> Dict[str, Any]:
        if lengths is None:
            return {}
        lengths = lengths.to(device=device, dtype=torch.long)
        time_token_lengths = torch.div(lengths + (self.patch_time - 1), self.patch_time, rounding_mode="floor")
        time_token_lengths = torch.clamp(time_token_lengths, min=0, max=n_time)
        token_lengths = torch.clamp(time_token_lengths * n_freq, min=0, max=seq_len)
        positions = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
        key_padding_mask = positions >= token_lengths.unsqueeze(1)
        return {
            "time_token_lengths": time_token_lengths,
            "token_lengths": token_lengths,
            "key_padding_mask": key_padding_mask,
        }

    def forward(
        self,
        x_logmel: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        patch_info = self.patchify(x_logmel, lengths=lengths)
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
        lengths: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        tokens, patch_info = self.patch_embedding(x_logmel, lengths=lengths)
        key_padding_mask = patch_info.get("key_padding_mask")
        if token_mask is not None:
            if token_mask.shape != tokens.shape[:2]:
                raise ValueError("token_mask shape mismatch.")
            if key_padding_mask is not None:
                token_mask = token_mask & (~key_padding_mask)
            mask_tok = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
            tokens = torch.where(token_mask.unsqueeze(-1), mask_tok, tokens)
        seq_len = tokens.shape[1]
        tokens = tokens + self._position_embedding(seq_len, tokens.device)
        encoded = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        encoded = self.norm(encoded)
        if key_padding_mask is not None:
            encoded = encoded.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)
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

    def _encode_visible_tokens(
        self,
        tokens: torch.Tensor,
        valid_mask: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len, dim = tokens.shape
        visible_mask = valid_mask & (~mask)
        if not visible_mask.any(dim=1).all():
            visible_mask = torch.where(
                valid_mask.any(dim=1, keepdim=True),
                visible_mask | (~visible_mask.any(dim=1, keepdim=True) & valid_mask),
                visible_mask,
            )

        pos_embed = self.encoder._position_embedding(seq_len, tokens.device).expand(bsz, -1, -1)
        visible_lengths = visible_mask.sum(dim=1).to(torch.long)
        max_visible = int(max(1, visible_lengths.max().item()))

        visible_tokens = tokens.new_zeros((bsz, max_visible, dim))
        visible_padding_mask = torch.ones((bsz, max_visible), dtype=torch.bool, device=tokens.device)

        for b in range(bsz):
            idx = torch.nonzero(visible_mask[b], as_tuple=False).flatten()
            count = int(idx.numel())
            if count == 0:
                continue
            visible_tokens[b, :count] = tokens[b, idx] + pos_embed[b, idx]
            visible_padding_mask[b, :count] = False

        encoded_visible = self.encoder.encoder(
            visible_tokens,
            src_key_padding_mask=visible_padding_mask,
        )
        encoded_visible = self.encoder.norm(encoded_visible)
        encoded_visible = encoded_visible.masked_fill(visible_padding_mask.unsqueeze(-1), 0.0)
        return encoded_visible, visible_mask

    def forward(
        self,
        x_logmel: torch.Tensor,
        mask: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        tokens, patch_info = self.encoder.patch_embedding(x_logmel, lengths=lengths)
        valid_mask = torch.ones_like(mask, dtype=torch.bool)
        if patch_info.get("key_padding_mask") is not None:
            valid_mask = ~patch_info["key_padding_mask"]

        encoded_visible, visible_mask = self._encode_visible_tokens(tokens, valid_mask=valid_mask, mask=mask)
        visible_dec_tokens = self.enc_to_dec(encoded_visible)

        bsz, seq_len, _ = tokens.shape
        mask_token_dec = self.enc_to_dec(self.encoder.mask_token).expand(bsz, seq_len, -1).clone()
        full_dec_tokens = mask_token_dec
        decoder_padding_mask = patch_info.get("key_padding_mask")

        for b in range(bsz):
            idx = torch.nonzero(visible_mask[b], as_tuple=False).flatten()
            count = int(idx.numel())
            if count == 0:
                continue
            full_dec_tokens[b, idx] = visible_dec_tokens[b, :count]

        dec_pos = self.enc_to_dec(self.encoder._position_embedding(seq_len, tokens.device)).expand(bsz, -1, -1)
        dec_tokens = full_dec_tokens + dec_pos
        dec_out = self.dec_norm(
            self.decoder(
                dec_tokens,
                src_key_padding_mask=decoder_padding_mask,
            )
        )
        pred = self.decoder_pred(dec_out)
        target = patch_info["patches"]

        loss_mask = mask & valid_mask
        if loss_mask.any():
            loss = F.mse_loss(pred[loss_mask], target[loss_mask])
        elif valid_mask.any():
            loss = F.mse_loss(pred[valid_mask], target[valid_mask])
        else:
            loss = F.mse_loss(pred, target)
        return loss


class AudioTransformerCTC(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        vocab_size: int,
        distill_projection_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.dim, int(vocab_size))
        self.dropout = nn.Dropout(0.1)
        projection_dim = (
            None if distill_projection_dim is None else int(distill_projection_dim)
        )
        self.distill_feature_dim = int(encoder.dim) if projection_dim is None else projection_dim
        if projection_dim is None or projection_dim == int(encoder.dim):
            self.distill_projection = nn.Identity()
        else:
            self.distill_projection = nn.Linear(int(encoder.dim), projection_dim)

    def encode_time_features(
        self,
        x_logmel: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded, patch_info = self.encoder(x_logmel, lengths=lengths, token_mask=None)
        bsz = encoded.shape[0]
        n_time = patch_info["n_time"]
        n_freq = patch_info["n_freq"]
        if n_freq > 1:
            encoded = encoded.view(bsz, n_time, n_freq, -1).mean(dim=2)
        out_lengths = patch_info.get("time_token_lengths")
        if out_lengths is None:
            patch_time = self.encoder.patch_embedding.patch_time
            out_lengths = torch.div(lengths + (patch_time - 1), patch_time, rounding_mode="floor")
        out_lengths = torch.clamp(out_lengths, max=encoded.shape[1])
        return encoded, out_lengths

    def forward(
        self,
        x_logmel: torch.Tensor,
        lengths: torch.Tensor,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, out_lengths = self.encode_time_features(x_logmel, lengths)
        logits = self.classifier(self.dropout(encoded))
        out_lengths = torch.clamp(out_lengths, max=logits.shape[1])
        if return_features:
            return logits, out_lengths, self.distill_projection(encoded)
        return logits, out_lengths

