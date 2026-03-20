from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

import torch

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:
    class TorchDataset:  # type: ignore[override]
        pass

from .text import normalize_transcript

try:
    from datasets import Audio as HFAudioFeature
    from datasets import Dataset, load_dataset
except ModuleNotFoundError:
    HFAudioFeature = None  # type: ignore[assignment]
    Dataset = Any  # type: ignore[assignment]
    load_dataset = None  # type: ignore[assignment]


TRANSCRIPT_CANDIDATES = [
    "text",
    "sentence",
    "raw_text",
    "normalized_text",
    "transcript",
]


@dataclass
class AudioPreprocessConfig:
    sample_rate: int
    max_duration_sec: Optional[float]
    n_mels: int
    win_length: int
    hop_length: int
    length_policy: str = "crop_or_pad"
    feature_norm: str = "none"


class InMemoryRowsDataset(TorchDataset):
    def __init__(self, rows: List[Dict[str, Any]], feature_keys: Optional[List[str]] = None) -> None:
        self.rows = rows
        keys = feature_keys if feature_keys is not None else (list(rows[0].keys()) if rows else [])
        # Keep a dict-like object compatible with existing summary calls.
        self.features = {k: None for k in keys}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.rows[index]


def _dataset_filter_keys() -> List[str]:
    return [
        "min_duration_sec",
        "max_duration_sec",
        "min_transcript_chars",
        "max_transcript_chars",
        "min_transcript_words",
        "max_transcript_words",
    ]


def dataset_filter_config(spec: Dict[str, Any]) -> Dict[str, Any]:
    return {key: spec[key] for key in _dataset_filter_keys() if key in spec and spec.get(key) is not None}


def _row_matches_filters(
    row: Dict[str, Any],
    transcript_key: Optional[str],
    filters: Dict[str, Any],
) -> bool:
    if not filters:
        return True
    resolved_key = resolve_transcript_key(row, transcript_key)
    text = normalize_transcript(str(row.get(resolved_key, ""))) if resolved_key is not None else ""
    num_chars = len(text)
    num_words = len(text.split()) if text else 0

    min_chars = filters.get("min_transcript_chars")
    max_chars = filters.get("max_transcript_chars")
    min_words = filters.get("min_transcript_words")
    max_words = filters.get("max_transcript_words")
    min_duration = filters.get("min_duration_sec")
    max_duration = filters.get("max_duration_sec")

    if min_duration is not None or max_duration is not None:
        duration_sec = _audio_duration_seconds(row.get("audio"))
        if duration_sec is not None:
            if min_duration is not None and duration_sec < float(min_duration):
                return False
            if max_duration is not None and duration_sec > float(max_duration):
                return False

    if min_chars is not None and num_chars < int(min_chars):
        return False
    if max_chars is not None and num_chars > int(max_chars):
        return False
    if min_words is not None and num_words < int(min_words):
        return False
    if max_words is not None and num_words > int(max_words):
        return False
    return True


def apply_dataset_filters(
    dataset: Dataset,
    transcript_key: Optional[str],
    spec: Optional[Dict[str, Any]] = None,
) -> Dataset:
    raw_filters = dataset_filter_config(spec or {})
    filters = dict(raw_filters)
    if spec is not None and str(spec.get("length_policy", "crop_or_pad")) != "none":
        filters.pop("min_duration_sec", None)
        filters.pop("max_duration_sec", None)
    if not filters:
        return dataset

    if isinstance(dataset, InMemoryRowsDataset):
        feature_keys = list(dataset.features.keys()) if hasattr(dataset.features, "keys") else None
        rows = [row for row in dataset.rows if _row_matches_filters(row, transcript_key=transcript_key, filters=filters)]
        return InMemoryRowsDataset(rows=rows, feature_keys=feature_keys)

    selected_indices: List[int] = []
    for idx in range(len(dataset)):
        if _row_matches_filters(dataset[idx], transcript_key=transcript_key, filters=filters):
            selected_indices.append(idx)

    if hasattr(dataset, "select"):
        return dataset.select(selected_indices)

    rows = [dataset[idx] for idx in selected_indices]
    feature_keys = list(getattr(dataset, "features", {}).keys()) if hasattr(getattr(dataset, "features", {}), "keys") else None
    return InMemoryRowsDataset(rows=rows, feature_keys=feature_keys)


def load_hf_audio_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: str,
    max_samples: Optional[int],
    streaming: bool = False,
) -> Dataset:
    if load_dataset is None:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install requirements.txt before running data/train/test."
        )
    def _streaming_split_candidates(split_value: str) -> List[str]:
        candidates: List[str] = [split_value]

        # Generic normalization used by some streaming builders
        # (e.g. librispeech clean/other -> base split names).
        normalized = split_value.replace(".clean", "").replace(".other", "")
        normalized = re.sub(r"\.\.+", ".", normalized).strip(".")
        if normalized and normalized not in candidates:
            candidates.append(normalized)

        # Explicit fallback patterns for LibriSpeech-style naming.
        m_train = re.match(r"^train\.(?:clean|other)\.(\d+)$", split_value)
        if m_train:
            alt = f"train.{m_train.group(1)}"
            if alt not in candidates:
                candidates.append(alt)
        if re.match(r"^validation\.(?:clean|other)$", split_value):
            if "validation" not in candidates:
                candidates.append("validation")
        if re.match(r"^test\.(?:clean|other)$", split_value):
            if "test" not in candidates:
                candidates.append("test")

        return candidates

    def _load(split_value: str, use_streaming: bool):
        load_kwargs = {"split": split_value, "cache_dir": cache_dir, "streaming": bool(use_streaming)}
        if dataset_config in (None, "", "null"):
            return load_dataset(dataset_name, **load_kwargs)
        return load_dataset(dataset_name, dataset_config, **load_kwargs)

    def _load_streaming_with_fallback(split_value: str):
        candidates = _streaming_split_candidates(split_value)
        last_exc: Optional[Exception] = None
        for candidate in candidates:
            try:
                ds = _load(candidate, use_streaming=True)
                if candidate != split_value:
                    print(
                        f"[DATASET] Streaming split fallback: requested='{split_value}' -> used='{candidate}' "
                        f"for dataset='{dataset_name}' config='{dataset_config}'"
                    )
                return ds
            except ValueError as exc:
                # Keep trying for split-name mismatches.
                if "Bad split" not in str(exc):
                    raise
                last_exc = exc
                continue
        if last_exc is not None:
            raise last_exc
        return _load(split_value, use_streaming=True)

    def _materialize_streaming(stream_ds, limit: Optional[int]) -> InMemoryRowsDataset:
        if HFAudioFeature is not None and hasattr(stream_ds, "cast_column"):
            try:
                stream_ds = stream_ds.cast_column("audio", HFAudioFeature(decode=False))
            except Exception as exc:
                print(
                    f"[DATASET][WARN] Unable to disable audio decoding for streaming dataset "
                    f"({dataset_name}, split={split}): {exc}"
                )

        rows: List[Dict[str, Any]] = []
        if limit is None:
            for row in stream_ds:
                rows.append(row)
        else:
            max_count = max(0, int(limit))
            for idx, row in enumerate(stream_ds):
                if idx >= max_count:
                    break
                rows.append(row)
        keys = list(rows[0].keys()) if rows else []
        return InMemoryRowsDataset(rows=rows, feature_keys=keys)

    if streaming:
        stream_ds = _load_streaming_with_fallback(split)
        return _materialize_streaming(stream_ds, max_samples)

    # Prefer split slicing to avoid downloading/loading full split for screening passes.
    if max_samples is not None:
        max_count = int(max_samples)
        split_sliced = f"{split}[:{max_count}]"
        try:
            return _load(split_sliced, use_streaming=False)
        except Exception:
            ds = _load(split, use_streaming=False)
            return ds.select(range(min(max_count, len(ds))))

    return _load(split, use_streaming=False)


def resolve_transcript_key(sample: Dict[str, Any], preferred: Optional[str]) -> Optional[str]:
    if preferred and preferred in sample:
        return preferred
    for key in TRANSCRIPT_CANDIDATES:
        if key in sample:
            return key
    return None


def _audio_duration_seconds(audio_value: Any) -> Optional[float]:
    if not isinstance(audio_value, dict):
        return None
    array = audio_value.get("array")
    sampling_rate = audio_value.get("sampling_rate")
    if array is not None and sampling_rate:
        try:
            num_samples = len(array)
            if num_samples > 0 and int(sampling_rate) > 0:
                return float(num_samples) / float(sampling_rate)
        except Exception:
            return None
    return None


class AudioFeatureDataset(TorchDataset):
    def __init__(
        self,
        hf_dataset: Dataset,
        audio_cfg: AudioPreprocessConfig,
        transcript_key: Optional[str] = None,
    ) -> None:
        self.ds = hf_dataset
        self.audio_cfg = audio_cfg
        sample = hf_dataset[0] if len(hf_dataset) > 0 else {}
        self.transcript_key = resolve_transcript_key(sample, transcript_key)
        self.max_samples = (
            int(audio_cfg.sample_rate * float(audio_cfg.max_duration_sec))
            if audio_cfg.max_duration_sec is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Lazy import to let dry-run execute even if audio deps are not installed yet.
        from .features import apply_length_policy, decode_audio, extract_logmel, normalize_logmel

        row = self.ds[index]
        waveform, sr = decode_audio(row["audio"], self.audio_cfg.sample_rate)
        waveform = apply_length_policy(
            waveform,
            target_num_samples=self.max_samples,
            length_policy=self.audio_cfg.length_policy,
        )
        waveform = waveform.to(torch.float32).contiguous()
        logmel = extract_logmel(
            waveform=waveform,
            sr=sr,
            n_mels=self.audio_cfg.n_mels,
            win_length=self.audio_cfg.win_length,
            hop_length=self.audio_cfg.hop_length,
        )
        logmel = normalize_logmel(logmel, feature_norm=self.audio_cfg.feature_norm)
        item = {
            "x_logmel": logmel,
            "length": int(logmel.shape[0]),
            "waveform": waveform.squeeze(0),
            "waveform_length": int(waveform.shape[-1]),
        }
        if self.transcript_key is not None:
            item["transcript"] = normalize_transcript(str(row.get(self.transcript_key, "")))
        return item


def collect_dataset_summary(
    dataset: Dataset,
    dataset_label: str,
    transcript_key: Optional[str] = None,
) -> Dict[str, Any]:
    features = getattr(dataset, "features", {})
    sample_keys = list(features.keys()) if hasattr(features, "keys") else []
    durations: List[float] = []
    transcript_chars: List[int] = []
    transcript_words: List[int] = []
    for idx in range(len(dataset)):
        row = dataset[idx]
        duration_sec = _audio_duration_seconds(row.get("audio"))
        if duration_sec is not None:
            durations.append(float(duration_sec))

        resolved_key = resolve_transcript_key(row, transcript_key)
        if resolved_key is not None:
            text = normalize_transcript(str(row.get(resolved_key, "")))
            transcript_chars.append(len(text))
            transcript_words.append(len(text.split()) if text else 0)

    return {
        "dataset_label": dataset_label,
        "num_examples": int(len(dataset)),
        "columns": sample_keys,
        "duration_avg_sec": float(sum(durations) / max(1, len(durations))) if durations else None,
        "duration_max_sec": float(max(durations)) if durations else None,
        "transcript_avg_chars": float(sum(transcript_chars) / max(1, len(transcript_chars))) if transcript_chars else None,
        "transcript_avg_words": float(sum(transcript_words) / max(1, len(transcript_words))) if transcript_words else None,
    }


def build_audio_preprocess_config(
    cfg: Dict[str, Any],
    dataset_spec: Optional[Dict[str, Any]] = None,
) -> AudioPreprocessConfig:
    audio = cfg["audio"]
    spec = dataset_spec or {}
    max_duration = spec.get("max_duration_sec", audio.get("max_duration_sec"))
    length_policy = spec.get("length_policy", audio.get("length_policy", "crop_or_pad"))
    return AudioPreprocessConfig(
        sample_rate=int(audio["sample_rate"]),
        max_duration_sec=None if max_duration is None else float(max_duration),
        n_mels=int(audio["n_mels"]),
        win_length=int(audio["win_length"]),
        hop_length=int(audio["hop_length"]),
        length_policy=str(length_policy),
        feature_norm=str(spec.get("feature_norm", audio.get("feature_norm", "none"))),
    )


def dataset_specs_for_data_step(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    specs.append(cfg["datasets"]["pretrain"])
    specs.append(cfg["datasets"]["asr_train"])
    specs.append(cfg["datasets"]["asr_valid"])
    specs.extend(cfg["datasets"]["asr_tests"])
    return specs
