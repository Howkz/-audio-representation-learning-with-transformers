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


def dataset_specs_for_data_step(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    specs.append(cfg["datasets"]["pretrain"])
    specs.append(cfg["datasets"]["asr_train"])
    specs.append(cfg["datasets"]["asr_valid"])
    specs.extend(cfg["datasets"]["asr_tests"])
    return specs


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
            print(
                f"  - {spec['name']} | config={spec.get('config')} | split={spec['split']} "
                f"| max={spec.get('max_samples')} | streaming={bool(spec.get('streaming', default_streaming))}"
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
                }
            )
            print(
                f"[DATA] Planned dataset={dataset_name} config={dataset_config} "
                f"split={split} max_samples={max_samples} streaming={streaming}"
            )
    else:
        # Import data loader lazily to avoid importing heavy dataset backends
        # when DATA is used in plan-only mode.
        from src.data.dataset import collect_dataset_summary, load_hf_audio_dataset

        for spec in specs:
            dataset_name = spec["name"]
            dataset_config = spec.get("config")
            split = spec["split"]
            max_samples = spec.get("max_samples")
            streaming = bool(spec.get("streaming", default_streaming))
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
            dataset_label = f"{dataset_name}:{split}"
            summary = collect_dataset_summary(ds, dataset_label=dataset_label)
            summary["max_samples"] = max_samples
            summary["dataset_config"] = dataset_config
            summary["streaming"] = streaming
            summary["planned_only"] = False
            summaries.append(summary)
            print(f"[DATA] Prepared {dataset_label} with {len(ds)} samples.")

    audio_cfg = cfg["audio"]
    manifest = {
        "experiment": cfg["experiment"]["name"],
        "audio_preprocess": {
            "sample_rate": int(audio_cfg["sample_rate"]),
            "max_duration_sec": float(audio_cfg["max_duration_sec"]),
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
