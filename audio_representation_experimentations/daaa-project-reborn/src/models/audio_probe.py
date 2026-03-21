from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .audio_transformer import AudioTransformerEncoder


class AudioLinearProbe(nn.Module):
    def __init__(
        self,
        encoder: AudioTransformerEncoder,
        num_classes: int,
        dropout: float = 0.1,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(int(encoder.dim), int(num_classes))
        self.freeze_encoder = bool(freeze_encoder)

        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

    def _pool_encoded(
        self,
        encoded: torch.Tensor,
        patch_info: Dict[str, Any],
    ) -> torch.Tensor:
        key_padding_mask = patch_info.get("key_padding_mask")
        if key_padding_mask is None:
            return encoded.mean(dim=1)
        valid_mask = (~key_padding_mask).to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * valid_mask).sum(dim=1)
        denom = valid_mask.sum(dim=1).clamp_min(1.0)
        return pooled / denom

    def encode_pooled(
        self,
        x_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        if self.freeze_encoder:
            with torch.no_grad():
                encoded, patch_info = self.encoder(x_features, lengths=lengths, token_mask=None)
        else:
            encoded, patch_info = self.encoder(x_features, lengths=lengths, token_mask=None)
        pooled = self._pool_encoded(encoded, patch_info=patch_info)
        return pooled, patch_info

    def forward(
        self,
        x_features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        pooled, _ = self.encode_pooled(x_features=x_features, lengths=lengths)
        return self.classifier(self.dropout(pooled))
