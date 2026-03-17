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
    collect_dataset_summary,
    dataset_specs_for_data_step,
    load_hf_audio_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate datasets for DAAA audio project.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true", help="Validate config and print plan without downloading datasets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    cache_dir = cfg["experiment"]["cache_dir"]
    processed_dir = Path(cfg["experiment"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("[DATA] Dry-run enabled. Planned datasets:")
        for spec in dataset_specs_for_data_step(cfg):
            print(f"  - {spec['name']} | config={spec.get('config')} | split={spec['split']} | max={spec.get('max_samples')}")
        return

    summaries: List[Dict[str, Any]] = []
    specs = dataset_specs_for_data_step(cfg)
    for spec in specs:
        dataset_name = spec["name"]
        dataset_config = spec.get("config")
        split = spec["split"]
        max_samples = spec.get("max_samples")
        print(
            f"[DATA] Loading dataset={dataset_name} config={dataset_config} "
            f"split={split} max_samples={max_samples}"
        )
        ds = load_hf_audio_dataset(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            split=split,
            cache_dir=cache_dir,
            max_samples=max_samples,
        )
        dataset_label = f"{dataset_name}:{split}"
        summary = collect_dataset_summary(ds, dataset_label=dataset_label)
        summary["max_samples"] = max_samples
        summary["dataset_config"] = dataset_config
        summaries.append(summary)
        print(f"[DATA] Prepared {dataset_label} with {len(ds)} samples.")

    audio_cfg = build_audio_preprocess_config(cfg)
    manifest = {
        "experiment": cfg["experiment"]["name"],
        "audio_preprocess": {
            "sample_rate": audio_cfg.sample_rate,
            "max_duration_sec": audio_cfg.max_duration_sec,
            "n_mels": audio_cfg.n_mels,
            "win_length": audio_cfg.win_length,
            "hop_length": audio_cfg.hop_length,
        },
        "datasets": summaries,
    }
    manifest_path = processed_dir / "datasets_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"[DATA] Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
