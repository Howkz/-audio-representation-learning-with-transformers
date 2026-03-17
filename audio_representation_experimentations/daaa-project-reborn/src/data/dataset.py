from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional

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
    max_duration_sec: float
    n_mels: int
    win_length: int
    hop_length: int


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
        self.max_samples = int(audio_cfg.sample_rate * audio_cfg.max_duration_sec)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Lazy import to let dry-run execute even if audio deps are not installed yet.
        from .features import crop_or_pad, decode_audio, extract_logmel

        row = self.ds[index]
        waveform, sr = decode_audio(row["audio"], self.audio_cfg.sample_rate)
        waveform = crop_or_pad(waveform, self.max_samples)
        logmel = extract_logmel(
            waveform=waveform,
            sr=sr,
            n_mels=self.audio_cfg.n_mels,
            win_length=self.audio_cfg.win_length,
            hop_length=self.audio_cfg.hop_length,
        )
        item = {
            "x_logmel": logmel,
            "length": int(logmel.shape[0]),
        }
        if self.transcript_key is not None:
            item["transcript"] = normalize_transcript(str(row.get(self.transcript_key, "")))
        return item


def collect_dataset_summary(dataset: Dataset, dataset_label: str) -> Dict[str, Any]:
    features = getattr(dataset, "features", {})
    sample_keys = list(features.keys()) if hasattr(features, "keys") else []
    return {
        "dataset_label": dataset_label,
        "num_examples": int(len(dataset)),
        "columns": sample_keys,
    }


def build_audio_preprocess_config(cfg: Dict[str, Any]) -> AudioPreprocessConfig:
    audio = cfg["audio"]
    return AudioPreprocessConfig(
        sample_rate=int(audio["sample_rate"]),
        max_duration_sec=float(audio["max_duration_sec"]),
        n_mels=int(audio["n_mels"]),
        win_length=int(audio["win_length"]),
        hop_length=int(audio["hop_length"]),
    )


def dataset_specs_for_data_step(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    specs.append(cfg["datasets"]["pretrain"])
    specs.append(cfg["datasets"]["asr_train"])
    specs.append(cfg["datasets"]["asr_valid"])
    specs.extend(cfg["datasets"]["asr_tests"])
    return specs
