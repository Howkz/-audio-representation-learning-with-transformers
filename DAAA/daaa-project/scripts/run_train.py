from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ensure_project_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAE pretraining and CTC fine-tuning.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--continue-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without launching training.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if args.dry_run:
        print("[TRAIN] Dry-run enabled.")
        print(f"[TRAIN] Seeds: {cfg['experiment']['seeds']}")
        print(f"[TRAIN] Pretrain dataset: {cfg['datasets']['pretrain']['name']}:{cfg['datasets']['pretrain']['split']}")
        print(f"[TRAIN] ASR train dataset: {cfg['datasets']['asr_train']['name']}:{cfg['datasets']['asr_train']['split']}")
        print(f"[TRAIN] ASR valid dataset: {cfg['datasets']['asr_valid']['name']}:{cfg['datasets']['asr_valid']['split']}")
        return

    from src.data.dataset import (
        AudioFeatureDataset,
        build_audio_preprocess_config,
        load_hf_audio_dataset,
    )
    from src.data.text import CharCTCTokenizer
    from src.training.loops import run_finetune_seed, run_pretrain_seed
    from src.training.results import aggregate_partial_to_final, write_run_partial
    from src.training.utils import set_seed

    def _build_dataset(local_cfg, spec, local_audio_cfg):
        ds = load_hf_audio_dataset(
            dataset_name=spec["name"],
            dataset_config=spec.get("config"),
            split=spec["split"],
            cache_dir=local_cfg["experiment"]["cache_dir"],
            max_samples=spec.get("max_samples"),
        )
        return AudioFeatureDataset(ds, audio_cfg=local_audio_cfg, transcript_key=spec.get("transcript_key"))

    audio_cfg = build_audio_preprocess_config(cfg)
    pretrain_ds = _build_dataset(cfg, cfg["datasets"]["pretrain"], audio_cfg)
    asr_train_ds = _build_dataset(cfg, cfg["datasets"]["asr_train"], audio_cfg)
    asr_valid_ds = _build_dataset(cfg, cfg["datasets"]["asr_valid"], audio_cfg)
    tokenizer = CharCTCTokenizer()

    benchmark_dir = Path(cfg["experiment"]["results_dir"]) / "benchmark_results"
    partial_path = benchmark_dir / "train_audio_transformer_partial.json"
    final_path = benchmark_dir / "train_audio_transformer_final.json"

    for seed in cfg["experiment"]["seeds"]:
        set_seed(int(seed))
        print(f"\n[TRAIN] Seed={seed} start")
        pretrain_ckpt, pretrain_metrics = run_pretrain_seed(
            cfg=cfg,
            seed=int(seed),
            dataset=pretrain_ds,
            force_continue_completed=args.continue_completed,
        )
        final_ckpt, finetune_metrics = run_finetune_seed(
            cfg=cfg,
            seed=int(seed),
            train_dataset=asr_train_ds,
            valid_dataset=asr_valid_ds,
            pretrain_encoder_path=pretrain_ckpt,
            tokenizer=tokenizer,
            force_continue_completed=args.continue_completed,
        )
        run_payload = {
            **pretrain_metrics,
            **finetune_metrics,
            "seed": float(seed),
        }
        write_run_partial(
            partial_path=partial_path,
            run_id=str(seed),
            payload=run_payload,
            model_name=cfg["experiment"]["name"],
            architecture="Audio Transformer Encoder + CTC",
            adaptation="MAE pretrain -> CTC fine-tune",
        )
        print(f"[TRAIN] Seed={seed} done. Final checkpoint: {final_ckpt}")

    aggregate = aggregate_partial_to_final(partial_path=partial_path, final_path=final_path)
    print(f"[TRAIN] Aggregated metrics written to {final_path}")
    print(f"[TRAIN] Final metric keys: {list(aggregate.get('metrics', {}).keys())}")


if __name__ == "__main__":
    main()
