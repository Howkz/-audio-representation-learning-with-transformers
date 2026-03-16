from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from torch.utils.data import Dataset as TorchDataset
except ModuleNotFoundError:
    class TorchDataset:  # type: ignore[override]
        pass

from .text import normalize_transcript

try:
    from datasets import Dataset, load_dataset
except ModuleNotFoundError:
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


def load_hf_audio_dataset(
    dataset_name: str,
    dataset_config: Optional[str],
    split: str,
    cache_dir: str,
    max_samples: Optional[int],
) -> Dataset:
    if load_dataset is None:
        raise ModuleNotFoundError(
            "Missing dependency 'datasets'. Install requirements.txt before running data/train/test."
        )
    if dataset_config in (None, "", "null"):
        ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    else:
        ds = load_dataset(dataset_name, dataset_config, split=split, cache_dir=cache_dir)
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))
    return ds


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
    sample_keys = list(dataset.features.keys())
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
