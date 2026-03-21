from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

# Ensure project root is importable when running `python scripts/run_train.py`.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_project_dirs, load_config


def _pretrain_mode(cfg):
    pretrain_cfg = cfg.get("pretrain", {})
    if isinstance(pretrain_cfg, dict) and pretrain_cfg.get("mode") is not None:
        return str(pretrain_cfg.get("mode", "none")).lower()
    if bool(cfg["training"]["pretrain"].get("enabled", True)):
        return "mae"
    return "none"


def _pretrain_enabled(cfg):
    return _pretrain_mode(cfg) == "mae"


def _adaptation_label(cfg):
    pretrain_mode = _pretrain_mode(cfg)
    distill_cfg = cfg.get("distillation", {})
    if isinstance(distill_cfg, dict) and bool(distill_cfg.get("enabled", False)):
        teacher_cfg = cfg.get("teacher", {})
        source = str(teacher_cfg.get("source", "external"))
        family = str(teacher_cfg.get("family", source))
        if pretrain_mode == "mae":
            return f"MAE pretrain -> CTC fine-tune + distillation ({source}:{family})"
        return f"CTC fine-tune + distillation ({source}:{family})"
    if pretrain_mode == "mae":
        return "MAE pretrain -> CTC fine-tune"
    return "CTC fine-tune only (no MAE)"


def _probe_cfg(cfg):
    probe_cfg = cfg.get("probe", {})
    return probe_cfg if isinstance(probe_cfg, dict) else {}


def _probe_enabled(cfg) -> bool:
    return bool(_probe_cfg(cfg).get("enabled", False))


def _resolve_probe_encoder_path(cfg, *, pretrain_ckpt, finetune_ckpt):
    source = str(_probe_cfg(cfg).get("encoder_source", "pretrain_if_available_else_finetune")).lower()
    if source == "pretrain":
        return pretrain_ckpt
    if source == "finetune":
        return finetune_ckpt
    if source == "random":
        return None
    return pretrain_ckpt if pretrain_ckpt is not None else finetune_ckpt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MAE pretraining and CTC fine-tuning.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--continue-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate pipeline without launching training.")
    return parser.parse_args()


def _is_oom_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda error: out of memory" in message


def _reduce_stage_batch(cfg, stage: str) -> bool:
    stage_cfg = cfg["training"][stage]
    current_batch = int(stage_cfg["batch_size"])
    if current_batch <= 1:
        return False
    next_batch = max(1, current_batch // 2)
    stage_cfg["batch_size"] = next_batch
    stage_cfg["grad_accum_steps"] = int(stage_cfg.get("grad_accum_steps", 1)) * 2
    return True


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _finetune_seed_completed(cfg, seed: int) -> bool:
    root = Path(cfg["experiment"]["output_dir"]) / "checkpoints"
    exp_id = cfg["experiment"].get("id")
    if exp_id:
        root = root / str(exp_id)
    run_completed = root / "finetune" / f"seed_{seed}" / "run_completed.txt"
    final_model = root / "finetune" / f"seed_{seed}" / "ctc_final.pt"
    return run_completed.exists() and final_model.exists()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    ensure_project_dirs(cfg)

    if args.dry_run:
        print("[TRAIN] Dry-run enabled.")
        print(f"[TRAIN] Seeds: {cfg['experiment']['seeds']}")
        print(f"[TRAIN] Pretraining mode: {_pretrain_mode(cfg)}")
        print(f"[TRAIN] Distillation enabled: {bool(cfg.get('distillation', {}).get('enabled', False))}")
        if _probe_enabled(cfg):
            probe_cfg = _probe_cfg(cfg)
            print(
                f"[TRAIN] Probe enabled source={probe_cfg.get('encoder_source', 'pretrain_if_available_else_finetune')} "
                f"freeze_encoder={probe_cfg.get('freeze_encoder', True)} "
                f"epochs={probe_cfg.get('epochs')} batch_size={probe_cfg.get('batch_size')} "
                f"grad_acc={probe_cfg.get('grad_accum_steps')}"
            )
        diagnostics_cfg = cfg.get("diagnostics", {})
        if bool(diagnostics_cfg.get("forensics_enabled", False)):
            print(
                f"[TRAIN] Forensics enabled train_probe_size={diagnostics_cfg.get('train_probe_size')} "
                f"max_examples={diagnostics_cfg.get('max_saved_examples')} "
                f"max_frames={diagnostics_cfg.get('max_saved_frames')} "
                f"topk={diagnostics_cfg.get('topk_tokens')}"
            )
        if bool(cfg.get("distillation", {}).get("enabled", False)):
            teacher_cfg = cfg.get("teacher", {})
            distill_cfg = cfg.get("distillation", {})
            print(
                f"[TRAIN] Teacher source={teacher_cfg.get('source')} "
                f"family={teacher_cfg.get('family')} target={teacher_cfg.get('target')} "
                f"model={teacher_cfg.get('model_name')} "
                f"checkpoint={teacher_cfg.get('checkpoint_path')}"
            )
            if int(distill_cfg.get("warmup_steps", 0)) > 0:
                print(
                    f"[TRAIN] KD warmup steps={distill_cfg.get('warmup_steps')} "
                    f"start_factor={distill_cfg.get('warmup_start_factor', 0.0)} "
                    f"target_lambda_kd={distill_cfg.get('lambda_kd')}"
                )
        if bool(cfg.get("anti_overemit", {}).get("enabled", False)):
            overemit_cfg = cfg.get("anti_overemit", {})
            print(
                f"[TRAIN] Anti-overemit enabled "
                f"lambda={overemit_cfg.get('lambda')} "
                f"density_scale={overemit_cfg.get('density_scale')} "
                f"density_margin={overemit_cfg.get('density_margin')}"
            )
        if bool(cfg.get("anti_underemit", {}).get("enabled", False)):
            underemit_cfg = cfg.get("anti_underemit", {})
            print(
                f"[TRAIN] Anti-underemit enabled "
                f"lambda={underemit_cfg.get('lambda')} "
                f"density_scale={underemit_cfg.get('density_scale')} "
                f"density_margin={underemit_cfg.get('density_margin')} "
                f"min_density={underemit_cfg.get('min_density')}"
            )
        print(f"[TRAIN] Pretrain dataset: {cfg['datasets']['pretrain']['name']}:{cfg['datasets']['pretrain']['split']}")
        print(f"[TRAIN] ASR train dataset: {cfg['datasets']['asr_train']['name']}:{cfg['datasets']['asr_train']['split']}")
        print(f"[TRAIN] ASR valid dataset: {cfg['datasets']['asr_valid']['name']}:{cfg['datasets']['asr_valid']['split']}")
        if _probe_enabled(cfg):
            print(f"[TRAIN] Probe train dataset: {cfg['datasets']['probe_train']['name']}:{cfg['datasets']['probe_train']['split']}")
            print(f"[TRAIN] Probe valid dataset: {cfg['datasets']['probe_valid']['name']}:{cfg['datasets']['probe_valid']['split']}")
        return

    from src.data.dataset import (
        AudioClassificationDataset,
        AudioFeatureDataset,
        apply_dataset_filters,
        build_audio_preprocess_config,
        build_label_vocab,
        load_hf_audio_dataset,
        resolve_transcript_key,
    )
    from src.data.text import CharCTCTokenizer
    from src.training.loops import run_finetune_seed, run_pretrain_seed
    from src.training.probe import run_linear_probe_seed
    from src.training.results import aggregate_partial_to_final, write_run_partial
    from src.training.utils import set_seed

    def _load_filtered_dataset(local_cfg, spec):
        default_streaming = bool(local_cfg.get("data", {}).get("streaming", False))
        ds = load_hf_audio_dataset(
            dataset_name=spec["name"],
            dataset_config=spec.get("config"),
            split=spec["split"],
            cache_dir=local_cfg["experiment"]["cache_dir"],
            max_samples=spec.get("max_samples"),
            streaming=bool(spec.get("streaming", default_streaming)),
        )
        transcript_key = resolve_transcript_key(ds[0], spec.get("transcript_key")) if len(ds) > 0 else spec.get("transcript_key")
        ds = apply_dataset_filters(ds, transcript_key=transcript_key, spec=spec)
        if len(ds) == 0:
            raise ValueError(
                f"Dataset vide après filtrage: {spec['name']}:{spec['split']}. "
                "Assouplis les filtres de transcript/durée ou désactive le streaming pour ce split."
            )
        return ds, transcript_key

    def _build_asr_dataset(local_cfg, spec, local_audio_cfg):
        ds, transcript_key = _load_filtered_dataset(local_cfg, spec)
        is_training_split = spec is cfg["datasets"]["asr_train"]
        return AudioFeatureDataset(
            ds,
            audio_cfg=local_audio_cfg,
            transcript_key=transcript_key,
            enable_augmentations=is_training_split,
            source_dataset=spec.get("name"),
            source_split=spec.get("split"),
        )

    def _build_probe_dataset(local_cfg, spec, local_audio_cfg, label_to_id, *, is_training_split: bool):
        ds, _ = _load_filtered_dataset(local_cfg, spec)
        probe_cfg = _probe_cfg(local_cfg)
        label_fields = probe_cfg.get("label_fields")
        if isinstance(label_fields, str):
            label_fields = [label_fields]
        return AudioClassificationDataset(
            ds,
            audio_cfg=local_audio_cfg,
            label_to_id=label_to_id,
            label_key=probe_cfg.get("label_key"),
            label_fields=label_fields if isinstance(label_fields, list) else None,
            enable_augmentations=is_training_split and bool(probe_cfg.get("enable_augmentations", False)),
            source_dataset=spec.get("name"),
            source_split=spec.get("split"),
        )

    pretrain_enabled = _pretrain_enabled(cfg)
    pretrain_ds = None
    if pretrain_enabled:
        pretrain_ds = _build_asr_dataset(
            cfg,
            cfg["datasets"]["pretrain"],
            build_audio_preprocess_config(cfg, cfg["datasets"]["pretrain"]),
        )
    asr_train_ds = _build_asr_dataset(cfg, cfg["datasets"]["asr_train"], build_audio_preprocess_config(cfg, cfg["datasets"]["asr_train"]))
    asr_valid_ds = _build_asr_dataset(cfg, cfg["datasets"]["asr_valid"], build_audio_preprocess_config(cfg, cfg["datasets"]["asr_valid"]))
    probe_train_ds = None
    probe_valid_ds = None
    label_to_id = None
    if _probe_enabled(cfg):
        probe_cfg = _probe_cfg(cfg)
        probe_train_raw, _ = _load_filtered_dataset(cfg, cfg["datasets"]["probe_train"])
        label_fields = probe_cfg.get("label_fields")
        if isinstance(label_fields, str):
            label_fields = [label_fields]
        label_to_id = build_label_vocab(
            probe_train_raw,
            label_key=probe_cfg.get("label_key"),
            label_fields=label_fields if isinstance(label_fields, list) else None,
        )
        probe_train_ds = _build_probe_dataset(
            cfg,
            cfg["datasets"]["probe_train"],
            build_audio_preprocess_config(cfg, cfg["datasets"]["probe_train"]),
            label_to_id,
            is_training_split=True,
        )
        probe_valid_ds = _build_probe_dataset(
            cfg,
            cfg["datasets"]["probe_valid"],
            build_audio_preprocess_config(cfg, cfg["datasets"]["probe_valid"]),
            label_to_id,
            is_training_split=False,
        )
    tokenizer = CharCTCTokenizer()

    exp_id = cfg["experiment"].get("id")
    benchmark_dir = Path(cfg["experiment"]["results_dir"]) / "benchmark_results"
    filename_suffix = f"_{exp_id}" if exp_id else ""
    partial_path = benchmark_dir / f"train_audio_transformer{filename_suffix}_partial.json"
    final_path = benchmark_dir / f"train_audio_transformer{filename_suffix}_final.json"

    if (not args.continue_completed) and all(_finetune_seed_completed(cfg, int(seed)) for seed in cfg["experiment"]["seeds"]):
        print("[TRAIN] Tous les seeds sont déjà marqués comme terminés.")
        print("[TRAIN] Aucun entraînement ne sera relancé. Supprimer outputs/results pour repartir de zéro.")
        if final_path.exists():
            with open(final_path, "r", encoding="utf-8") as handle:
                final_data = json.load(handle)
            print(f"[TRAIN] Final metric keys: {list(final_data.get('metrics', {}).keys())}")
        else:
            print(f"[TRAIN] Fichier final absent: {final_path}")
        return

    print(f"[TRAIN] Experiment id: {exp_id if exp_id else 'N/A'}")
    print(f"[TRAIN] Pretraining mode: {_pretrain_mode(cfg)}")
    print(f"[TRAIN] Distillation enabled: {bool(cfg.get('distillation', {}).get('enabled', False))}")
    adaptation_label = _adaptation_label(cfg)

    for seed in cfg["experiment"]["seeds"]:
        set_seed(int(seed))
        print(f"\n[TRAIN] Seed={seed} start")
        pretrain_ckpt = None
        pretrain_metrics = {}
        if pretrain_enabled:
            pretrain_cfg = copy.deepcopy(cfg)
            for attempt in range(3):
                try:
                    pretrain_ckpt, pretrain_metrics = run_pretrain_seed(
                        cfg=pretrain_cfg,
                        seed=int(seed),
                        dataset=pretrain_ds,
                        force_continue_completed=args.continue_completed,
                    )
                    break
                except RuntimeError as exc:
                    if not _is_oom_error(exc):
                        raise
                    if not _reduce_stage_batch(pretrain_cfg, stage="pretrain"):
                        raise
                    _clear_cuda_cache()
                    stage_cfg = pretrain_cfg["training"]["pretrain"]
                    print(
                        f"[TRAIN][OOM] Seed={seed} pretrain retry {attempt + 1}/2 "
                        f"with batch_size={stage_cfg['batch_size']} grad_acc={stage_cfg['grad_accum_steps']}"
                    )
        else:
            print(f"[TRAIN] Seed={seed}: skipping MAE pretraining (CTC-only mode).")
        finetune_cfg = copy.deepcopy(cfg)
        for attempt in range(3):
            try:
                final_ckpt, finetune_metrics = run_finetune_seed(
                    cfg=finetune_cfg,
                    seed=int(seed),
                    train_dataset=asr_train_ds,
                    valid_dataset=asr_valid_ds,
                    pretrain_encoder_path=pretrain_ckpt,
                    tokenizer=tokenizer,
                    force_continue_completed=args.continue_completed,
                )
                break
            except RuntimeError as exc:
                if not _is_oom_error(exc):
                    raise
                if not _reduce_stage_batch(finetune_cfg, stage="finetune"):
                    raise
                _clear_cuda_cache()
                stage_cfg = finetune_cfg["training"]["finetune"]
                print(
                    f"[TRAIN][OOM] Seed={seed} finetune retry {attempt + 1}/2 "
                    f"with batch_size={stage_cfg['batch_size']} grad_acc={stage_cfg['grad_accum_steps']}"
                )
        probe_metrics = {}
        if _probe_enabled(cfg) and probe_train_ds is not None and probe_valid_ds is not None and label_to_id is not None:
            probe_cfg = copy.deepcopy(cfg)
            probe_encoder_path = _resolve_probe_encoder_path(
                cfg,
                pretrain_ckpt=pretrain_ckpt,
                finetune_ckpt=final_ckpt,
            )
            for attempt in range(3):
                try:
                    _, probe_metrics = run_linear_probe_seed(
                        cfg=probe_cfg,
                        seed=int(seed),
                        train_dataset=probe_train_ds,
                        valid_dataset=probe_valid_ds,
                        label_to_id=label_to_id,
                        encoder_checkpoint_path=probe_encoder_path,
                        force_continue_completed=args.continue_completed,
                    )
                    break
                except RuntimeError as exc:
                    if not _is_oom_error(exc):
                        raise
                    probe_stage = probe_cfg.setdefault("probe", {})
                    current_batch = int(probe_stage.get("batch_size", 1))
                    if current_batch <= 1:
                        raise
                    probe_stage["batch_size"] = max(1, current_batch // 2)
                    probe_stage["grad_accum_steps"] = int(probe_stage.get("grad_accum_steps", 1)) * 2
                    _clear_cuda_cache()
                    print(
                        f"[TRAIN][OOM] Seed={seed} probe retry {attempt + 1}/2 "
                        f"with batch_size={probe_stage['batch_size']} grad_acc={probe_stage['grad_accum_steps']}"
                    )
        run_payload = {
            **pretrain_metrics,
            **finetune_metrics,
            **probe_metrics,
            "seed": float(seed),
        }
        write_run_partial(
            partial_path=partial_path,
            run_id=str(seed),
            payload=run_payload,
            model_name=cfg["experiment"]["name"],
            architecture="Audio Transformer Encoder + CTC",
            adaptation=adaptation_label,
        )
        print(f"[TRAIN] Seed={seed} done. Final checkpoint: {final_ckpt}")

        if pretrain_enabled and bool(cfg["experiment"].get("cleanup_pretrain_checkpoints_after_finetune", False)):
            pretrain_dir = Path(cfg["experiment"]["output_dir"]) / "checkpoints"
            if exp_id:
                pretrain_dir = pretrain_dir / str(exp_id)
            pretrain_dir = pretrain_dir / "pretrain" / f"seed_{seed}"
            if pretrain_dir.exists():
                for ckpt_file in pretrain_dir.glob("checkpoint_*.pt"):
                    ckpt_file.unlink(missing_ok=True)
                latest_ptr = pretrain_dir / "latest.pt"
                if latest_ptr.exists():
                    latest_ptr.unlink()
                print(f"[TRAIN] Low-storage cleanup applied for pretrain seed={seed}.")

    aggregate = aggregate_partial_to_final(partial_path=partial_path, final_path=final_path)
    print(f"[TRAIN] Aggregated metrics written to {final_path}")
    print(f"[TRAIN] Final metric keys: {list(aggregate.get('metrics', {}).keys())}")


if __name__ == "__main__":
    main()
