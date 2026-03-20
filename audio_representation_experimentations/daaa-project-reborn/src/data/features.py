from __future__ import annotations
import io
import subprocess
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _hz_to_mel(freq_hz: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + (freq_hz / 700.0))


def _mel_to_hz(freq_mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    if f_max is None:
        f_max = float(sample_rate) / 2.0
    num_freqs = (n_fft // 2) + 1
    mel_min = _hz_to_mel(torch.tensor(f_min, dtype=torch.float32))
    mel_max = _hz_to_mel(torch.tensor(f_max, dtype=torch.float32))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, dtype=torch.float32)
    hz_points = _mel_to_hz(mel_points)
    bin_points = torch.floor((n_fft + 1) * hz_points / float(sample_rate)).to(torch.long)
    bin_points = torch.clamp(bin_points, min=0, max=num_freqs - 1)

    fb = torch.zeros((n_mels, num_freqs), dtype=torch.float32)
    for m in range(1, n_mels + 1):
        left = int(bin_points[m - 1].item())
        center = int(bin_points[m].item())
        right = int(bin_points[m + 1].item())

        if center <= left:
            center = min(left + 1, num_freqs - 1)
        if right <= center:
            right = min(center + 1, num_freqs - 1)
        if right <= left:
            continue

        for k in range(left, center):
            fb[m - 1, k] = float(k - left) / float(max(1, center - left))
        for k in range(center, right):
            fb[m - 1, k] = float(right - k) / float(max(1, right - center))
    return fb


def _resample_waveform_linear(waveform: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return waveform
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sampling rates must be > 0.")
    original_len = waveform.shape[-1]
    target_len = max(1, int(round(original_len * float(target_sr) / float(orig_sr))))
    x = waveform.unsqueeze(0)  # [1, 1, T]
    y = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
    return y.squeeze(0)


def _decode_audio_with_soundfile(audio_dict: Dict[str, Any]) -> Tuple[torch.Tensor, int]:
    try:
        import soundfile as sf
    except ModuleNotFoundError:
        sf = None  # type: ignore[assignment]

    audio_bytes = audio_dict.get("bytes")
    audio_path = audio_dict.get("path")
    data = None
    sr = None

    if sf is not None:
        if audio_bytes is not None:
            data, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        elif audio_path:
            data, sr = sf.read(audio_path, dtype="float32")

    if data is None or sr is None:
        try:
            import torchaudio
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError from exc

        if audio_bytes is not None:
            waveform, sr_loaded = torchaudio.load(io.BytesIO(audio_bytes))
        elif audio_path:
            waveform, sr_loaded = torchaudio.load(audio_path)
        else:
            raise ValueError("Audio dict must provide 'array' or ('bytes'/'path').")
        return waveform.to(torch.float32), int(sr_loaded)

    waveform = torch.tensor(data, dtype=torch.float32)
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim == 2:
        # soundfile returns [T, C] for multi-channel audio.
        waveform = waveform.transpose(0, 1).contiguous()
    else:
        waveform = waveform.reshape(1, -1)
    return waveform, int(sr)


def _decode_audio_with_ffmpeg(audio_dict: Dict[str, Any], target_sr: int) -> Tuple[torch.Tensor, int]:
    audio_bytes = audio_dict.get("bytes")
    audio_path = audio_dict.get("path")

    base_cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    if audio_bytes is not None:
        cmd = base_cmd + [
            "-i",
            "pipe:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(int(target_sr)),
            "pipe:1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ModuleNotFoundError(
                "No audio decoder available. Install 'soundfile' (pip) or 'torchaudio', "
                "or install ffmpeg in system PATH."
            ) from exc
    elif audio_path:
        cmd = base_cmd + [
            "-i",
            str(audio_path),
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(int(target_sr)),
            "pipe:1",
        ]
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except FileNotFoundError as exc:
            raise ModuleNotFoundError(
                "No audio decoder available. Install 'soundfile' (pip) or 'torchaudio', "
                "or install ffmpeg in system PATH."
            ) from exc
    else:
        raise ValueError("Audio dict must provide 'array' or ('bytes'/'path').")

    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(
            f"ffmpeg failed to decode audio (code={proc.returncode}). stderr={stderr}"
        )

    # `frombuffer` over subprocess stdout creates a read-only NumPy view.
    # Copy before converting to Torch to avoid undefined writes/warnings.
    pcm = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if pcm.size == 0:
        raise RuntimeError("ffmpeg produced empty audio output.")
    waveform = torch.from_numpy(pcm).unsqueeze(0)
    return waveform, int(target_sr)


def decode_audio(audio_dict: Dict[str, Any], target_sr: int) -> Tuple[torch.Tensor, int]:
    if "array" in audio_dict and audio_dict["array"] is not None:
        array = audio_dict["array"]
        sr = int(audio_dict["sampling_rate"])
        waveform = torch.tensor(array, dtype=torch.float32)
    else:
        try:
            waveform, sr = _decode_audio_with_soundfile(audio_dict)
        except ModuleNotFoundError:
            waveform, sr = _decode_audio_with_ffmpeg(audio_dict, target_sr=target_sr)

    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim > 2:
        waveform = waveform.reshape(1, -1)
    # Convert to mono for stable and memory-bounded processing.
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != target_sr:
        waveform = _resample_waveform_linear(waveform, sr, target_sr)
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


def pad_to_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    current = waveform.shape[-1]
    if current < target_num_samples:
        pad = target_num_samples - current
        waveform = torch.nn.functional.pad(waveform, (0, pad))
    return waveform


def apply_length_policy(
    waveform: torch.Tensor,
    target_num_samples: int | None,
    length_policy: str = "crop_or_pad",
) -> torch.Tensor:
    if target_num_samples is None or str(length_policy) == "none":
        return waveform
    if target_num_samples <= 0:
        raise ValueError("target_num_samples must be > 0 when a length policy is active.")

    policy = str(length_policy)
    if policy == "crop_or_pad":
        return crop_or_pad(waveform, int(target_num_samples))
    if policy == "pad_only":
        return pad_to_length(waveform, int(target_num_samples))
    raise ValueError(f"Unsupported length_policy='{length_policy}'.")


def extract_logmel(
    waveform: torch.Tensor,
    sr: int,
    n_mels: int,
    win_length: int,
    hop_length: int,
) -> torch.Tensor:
    if waveform.ndim != 2 or waveform.shape[0] != 1:
        raise ValueError("waveform must be [1, T].")
    if win_length <= 0 or hop_length <= 0:
        raise ValueError("win_length and hop_length must be > 0.")
    if n_mels <= 0:
        raise ValueError("n_mels must be > 0.")

    signal = waveform.squeeze(0)
    window = torch.hann_window(win_length, dtype=signal.dtype, device=signal.device)
    spec = torch.stft(
        signal,
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        normalized=False,
        onesided=True,
        return_complex=True,
    )  # [F, T]
    power = spec.abs().pow(2.0)

    fb = _build_mel_filterbank(sample_rate=sr, n_fft=win_length, n_mels=n_mels).to(power.device, power.dtype)
    mel = torch.matmul(fb, power)  # [M, T]
    log_mel = torch.log(torch.clamp(mel, min=1e-6))
    return log_mel.transpose(0, 1).contiguous()  # [T, M]


def normalize_logmel(
    logmel: torch.Tensor,
    feature_norm: str = "none",
    eps: float = 1e-5,
) -> torch.Tensor:
    mode = str(feature_norm)
    if mode == "none":
        return logmel
    if mode == "utterance":
        mean = logmel.mean(dim=0, keepdim=True)
        std = logmel.std(dim=0, keepdim=True, unbiased=False).clamp_min(float(eps))
        return (logmel - mean) / std
    raise ValueError(f"Unsupported feature_norm='{feature_norm}'.")


def _mask_random_span(matrix: torch.Tensor, axis: int, max_width: int) -> torch.Tensor:
    if max_width <= 0:
        return matrix
    size = int(matrix.shape[axis])
    if size <= 1:
        return matrix
    width = int(torch.randint(low=0, high=max_width + 1, size=(1,)).item())
    if width <= 0:
        return matrix
    start_max = max(1, size - width + 1)
    start = int(torch.randint(low=0, high=start_max, size=(1,)).item())
    masked = matrix.clone()
    if axis == 0:
        masked[start : start + width, :] = 0.0
    else:
        masked[:, start : start + width] = 0.0
    return masked


def apply_waveform_augmentations(
    waveform: torch.Tensor,
    augment_cfg: Dict[str, Any] | None,
) -> torch.Tensor:
    if not augment_cfg or not bool(augment_cfg.get("enabled", False)):
        return waveform

    augmented = waveform.clone()

    gain_prob = float(augment_cfg.get("gain_prob", 0.0))
    gain_db_max = float(augment_cfg.get("gain_db_max", 0.0))
    if gain_prob > 0.0 and gain_db_max > 0.0 and torch.rand(1).item() < gain_prob:
        gain_db = float(torch.empty(1).uniform_(-gain_db_max, gain_db_max).item())
        gain = float(10.0 ** (gain_db / 20.0))
        augmented = augmented * gain

    noise_prob = float(augment_cfg.get("noise_prob", 0.0))
    noise_snr_db_min = float(augment_cfg.get("noise_snr_db_min", 20.0))
    noise_snr_db_max = float(augment_cfg.get("noise_snr_db_max", 35.0))
    if noise_prob > 0.0 and torch.rand(1).item() < noise_prob:
        snr_db = float(torch.empty(1).uniform_(noise_snr_db_min, noise_snr_db_max).item())
        signal_rms = augmented.pow(2.0).mean().sqrt().clamp_min(1e-6)
        noise_rms = signal_rms / float(10.0 ** (snr_db / 20.0))
        noise = torch.randn_like(augmented) * noise_rms
        augmented = augmented + noise

    return augmented.clamp(min=-1.0, max=1.0)


def apply_logmel_augmentations(
    logmel: torch.Tensor,
    augment_cfg: Dict[str, Any] | None,
) -> torch.Tensor:
    if not augment_cfg or not bool(augment_cfg.get("enabled", False)):
        return logmel

    specaugment_prob = float(augment_cfg.get("specaugment_prob", 0.0))
    if specaugment_prob <= 0.0 or torch.rand(1).item() >= specaugment_prob:
        return logmel

    augmented = logmel.clone()
    num_time_masks = int(augment_cfg.get("num_time_masks", 0))
    max_time_mask_frames = int(augment_cfg.get("max_time_mask_frames", 0))
    num_freq_masks = int(augment_cfg.get("num_freq_masks", 0))
    max_freq_mask_bins = int(augment_cfg.get("max_freq_mask_bins", 0))

    for _ in range(max(0, num_time_masks)):
        augmented = _mask_random_span(augmented, axis=0, max_width=max_time_mask_frames)
    for _ in range(max(0, num_freq_masks)):
        augmented = _mask_random_span(augmented, axis=1, max_width=max_freq_mask_bins)
    return augmented
