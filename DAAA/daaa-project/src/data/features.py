from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torchaudio


def decode_audio(audio_dict: Dict[str, Any], target_sr: int) -> Tuple[torch.Tensor, int]:
    array = audio_dict["array"]
    sr = int(audio_dict["sampling_rate"])
    waveform = torch.tensor(array, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(1, -1)
    # Convert to mono for stable and memory-bounded processing.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def crop_or_pad(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current = waveform.shape[-1]
    if current > target_num_samples:
        waveform = waveform[..., :target_num_samples]
    elif current < target_num_samples:
        pad = target_num_samples - current
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def extract_logmel(
    waveform: torch.Tensor,
    sr: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=win_length,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        center=True,
        power=2.0,
    )(waveform)
    log_mel = torch.log(torch.clamp(mel, min=1e-6))
    # [1, M, T] -> [T, M]
    return log_mel.squeeze(0).transpose(0, 1).contiguous()

