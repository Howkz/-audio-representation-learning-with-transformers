from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is importable when running `python scripts/run_data.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_project_dirs, load_config
from src.data.dataset import (
    build_audio_preprocess_config,
    dataset_filter_config,
    dataset_specs_for_data_step,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate datasets for DAAA audio project.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan without downloading datasets.")
    parser.add_argument(
        "--materialize",
        action="store_true",
        help="Actually load datasets in DATA step. Default is plan-only to avoid heavy IO/disk pressure.",
    )
    return parser.parse_args()

def _format_dataset_filters(spec: Dict[str, Any]) -> str:
    filters = dataset_filter_config(spec)
    if not filters:
        return "none"
    return ",".join(f"{key}={value}" for key, value in filters.items())


def _format_augmentations(spec: Dict[str, Any]) -> str:
    augment_cfg = spec.get("augmentations", {})
    if not isinstance(augment_cfg, dict) or not bool(augment_cfg.get("enabled", False)):
        return "none"
    summary_keys = [
        "gain_prob",
        "gain_db_max",
        "noise_prob",
        "noise_snr_db_min",
        "noise_snr_db_max",
        "specaugment_prob",
        "num_time_masks",
        "max_time_mask_frames",
        "num_freq_masks",
        "max_freq_mask_bins",
    ]
    return ",".join(f"{key}={augment_cfg[key]}" for key in summary_keys if key in augment_cfg)


def _strict_asr_consistency_enabled(cfg: Dict[str, Any], spec: Dict[str, Any]) -> bool:
    data_cfg = cfg.get("data", {})
    return bool(spec.get("strict_asr_consistency", data_cfg.get("strict_asr_consistency", False)))


def _validate_asr_supervision_consistency(cfg: Dict[str, Any], spec: Dict[str, Any], local_audio_cfg) -> None:
    transcript_key = spec.get("transcript_key")
    if not transcript_key or not _strict_asr_consistency_enabled(cfg, spec):
        return
    if local_audio_cfg.length_policy != "none" and local_audio_cfg.max_duration_sec is not None:
        raise ValueError(
            f"ASR split {spec['name']}:{spec['split']} uses transcript_key='{transcript_key}' "
            f"with length_policy='{local_audio_cfg.length_policy}' and max_duration_sec={local_audio_cfg.max_duration_sec}. "
            "This may crop audio while keeping the full transcript."
        )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    cache_dir = cfg["experiment"]["cache_dir"]
    processed_dir = Path(cfg["experiment"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    default_streaming = bool(cfg.get("data", {}).get("streaming", False))

    if args.dry_run:
        print("[DATA] Dry-run enabled. Planned datasets:")
        for spec in dataset_specs_for_data_step(cfg):
            local_audio_cfg = build_audio_preprocess_config(cfg, spec)
            _validate_asr_supervision_consistency(cfg, spec, local_audio_cfg)
            print(
                f"  - {spec['name']} | config={spec.get('config')} | split={spec['split']} "
                f"| max={spec.get('max_samples')} | streaming={bool(spec.get('streaming', default_streaming))} "
                f"| max_duration_sec={local_audio_cfg.max_duration_sec} | length_policy={local_audio_cfg.length_policy} "
                f"| feature_type={local_audio_cfg.feature_type} | feature_norm={local_audio_cfg.feature_norm} | filters={_format_dataset_filters(spec)} "
                f"| augmentations={_format_augmentations(spec)}"
            )
        return

    summaries: List[Dict[str, Any]] = []
    specs = dataset_specs_for_data_step(cfg)
    if not args.materialize:
        print("[DATA] Plan-only mode: skipping dataset materialization (use --materialize to force loading).")
        for spec in specs:
            dataset_name = spec["name"]
            dataset_config = spec.get("config")
            split = spec["split"]
            max_samples = spec.get("max_samples")
            streaming = bool(spec.get("streaming", default_streaming))
            local_audio_cfg = build_audio_preprocess_config(cfg, spec)
            _validate_asr_supervision_consistency(cfg, spec, local_audio_cfg)
            dataset_label = f"{dataset_name}:{split}"
            columns = ["audio"]
            transcript_key = spec.get("transcript_key")
            if transcript_key:
                columns.append(str(transcript_key))
            summaries.append(
                {
                    "dataset_label": dataset_label,
                    "num_examples": int(max_samples) if max_samples is not None else -1,
                    "columns": columns,
                    "max_samples": max_samples,
                    "dataset_config": dataset_config,
                    "streaming": streaming,
                    "planned_only": True,
                    "audio_preprocess": {
                        "sample_rate": int(local_audio_cfg.sample_rate),
                        "max_duration_sec": None if local_audio_cfg.max_duration_sec is None else float(local_audio_cfg.max_duration_sec),
                        "feature_type": str(local_audio_cfg.feature_type),
                        "length_policy": str(local_audio_cfg.length_policy),
                        "feature_norm": str(local_audio_cfg.feature_norm),
                        "filters": dataset_filter_config(spec),
                        "augmentations": local_audio_cfg.augmentations,
                        "n_mels": int(local_audio_cfg.n_mels),
                        "win_length": int(local_audio_cfg.win_length),
                        "hop_length": int(local_audio_cfg.hop_length),
                    },
                }
            )
            print(
                f"[DATA] Planned dataset={dataset_name} config={dataset_config} "
                f"split={split} max_samples={max_samples} streaming={streaming} "
                f"max_duration_sec={local_audio_cfg.max_duration_sec} "
                f"feature_type={local_audio_cfg.feature_type} "
                f"length_policy={local_audio_cfg.length_policy} "
                f"feature_norm={local_audio_cfg.feature_norm} "
                f"filters={_format_dataset_filters(spec)} "
                f"augmentations={_format_augmentations(spec)}"
            )
    else:
        # Import data loader lazily to avoid importing heavy dataset backends
        # when DATA is used in plan-only mode.
        from src.data.dataset import apply_dataset_filters, collect_dataset_summary, load_hf_audio_dataset, resolve_transcript_key

        for spec in specs:
            dataset_name = spec["name"]
            dataset_config = spec.get("config")
            split = spec["split"]
            max_samples = spec.get("max_samples")
            streaming = bool(spec.get("streaming", default_streaming))
            local_audio_cfg = build_audio_preprocess_config(cfg, spec)
            _validate_asr_supervision_consistency(cfg, spec, local_audio_cfg)
            print(
                f"[DATA] Loading dataset={dataset_name} config={dataset_config} "
                f"split={split} max_samples={max_samples} streaming={streaming}"
            )
            ds = load_hf_audio_dataset(
                dataset_name=dataset_name,
                dataset_config=dataset_config,
                split=split,
                cache_dir=cache_dir,
                max_samples=max_samples,
                streaming=streaming,
            )
            transcript_key = resolve_transcript_key(ds[0], spec.get("transcript_key")) if len(ds) > 0 else spec.get("transcript_key")
            ds = apply_dataset_filters(ds, transcript_key=transcript_key, spec=spec)
            dataset_label = f"{dataset_name}:{split}"
            summary = collect_dataset_summary(ds, dataset_label=dataset_label, transcript_key=transcript_key)
            summary["max_samples"] = max_samples
            summary["dataset_config"] = dataset_config
            summary["streaming"] = streaming
            summary["planned_only"] = False
            summary["audio_preprocess"] = {
                "sample_rate": int(local_audio_cfg.sample_rate),
                "max_duration_sec": None if local_audio_cfg.max_duration_sec is None else float(local_audio_cfg.max_duration_sec),
                "feature_type": str(local_audio_cfg.feature_type),
                "length_policy": str(local_audio_cfg.length_policy),
                "feature_norm": str(local_audio_cfg.feature_norm),
                "filters": dataset_filter_config(spec),
                "augmentations": local_audio_cfg.augmentations,
                "n_mels": int(local_audio_cfg.n_mels),
                "win_length": int(local_audio_cfg.win_length),
                "hop_length": int(local_audio_cfg.hop_length),
            }
            summaries.append(summary)
            print(f"[DATA] Prepared {dataset_label} with {len(ds)} samples.")

    audio_cfg = cfg["audio"]
    manifest = {
        "experiment": cfg["experiment"]["name"],
        "audio_preprocess": {
            "sample_rate": int(audio_cfg["sample_rate"]),
            "max_duration_sec": None if audio_cfg.get("max_duration_sec") is None else float(audio_cfg["max_duration_sec"]),
            "feature_type": str(audio_cfg.get("feature_type", "logmel")),
            "feature_norm": str(audio_cfg.get("feature_norm", "none")),
            "n_mels": int(audio_cfg["n_mels"]),
            "win_length": int(audio_cfg["win_length"]),
            "hop_length": int(audio_cfg["hop_length"]),
        },
        "datasets": summaries,
    }
    manifest_path = processed_dir / "datasets_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[DATA] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
