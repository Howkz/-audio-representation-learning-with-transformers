import argparse
import json
import os
import random
import time
from typing import Dict, List, Optional, Tuple

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK_NAME = "ex3"
ARCHITECTURE = "Decodeur uniquement"
SEEDS = [42, 123, 456]
DEBUG_SAMPLES = 12

MODEL_KEY = "distilgpt2-ex3-zeroshot"
MODEL_NAME = "distilgpt2-ex3-zeroshot"
PRETRAINED_NAME = "distilbert/distilgpt2"
ADAPTATION = "Zero-shot direct prompting"
STRATEGY_NAME = "zero-shot-direct"

FORBIDDEN_ADVICE_PATTERNS = [
    "this is legal advice",
    "i am your lawyer",
    "consult an attorney",
    "hire a lawyer",
]


def build_prompt(clause_text: str) -> str:
    """Construit le prompt zero-shot pour expliquer une clause."""
    return (
        "Explain the following legal clause in plain English for a non-expert.\n"
        "Do not add obligations that are not in the clause.\n\n"
        f"Clause:\n{clause_text}\n\n"
        "Plain-English explanation:\n"
    )


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(description="Exercice 3 - DistilGPT-2 zero-shot prompting")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--continue-completed", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-input-tokens", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seeds", type=str, default=None)
    return parser.parse_args()


def parse_seeds(debug_mode: bool, seeds_text: Optional[str]) -> List[int]:
    """Retourne la liste des seeds actives (une seule en debug)."""
    if debug_mode:
        return [42]
    if seeds_text is None:
        return SEEDS
    parsed = [int(x.strip()) for x in seeds_text.split(",") if x.strip()]
    if not parsed:
        raise ValueError("No valid seed found in --seeds.")
    return parsed


def set_seed(seed: int) -> None:
    """Initialise les seeds Python/NumPy/PyTorch pour la reproductibilite."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_text(text: str) -> str:
    """Normalise un texte (trim + espaces)."""
    return " ".join(str(text).strip().split())


def _word_count(text: str) -> int:
    """Compte les mots d'un texte apres normalisation."""
    return len(normalize_text(text).split()) if text else 0


def _contains_forbidden_advice(text: str) -> bool:
    """Detecte les patterns de conseil juridique interdits."""
    lowered = text.lower()
    return any(pattern in lowered for pattern in FORBIDDEN_ADVICE_PATTERNS)


def build_ledgar_eval_split(debug_mode: bool) -> Tuple[Dataset, List[str], str]:
    """Charge le split d'evaluation LEDGAR avec reduction optionnelle en debug."""
    dataset = load_dataset("lex_glue", "ledgar")
    split_name = "test" if "test" in dataset else "validation"
    split = dataset[split_name]
    if debug_mode:
        split = split.select(range(min(DEBUG_SAMPLES, len(split))))
    label_names = dataset["train"].features["label"].names
    return split, label_names, split_name


def load_model_and_tokenizer():
    """Charge DistilGPT-2 et son tokenizer, puis place le modele sur le bon device."""
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    tokenizer.padding_side = "left"

    model_kwargs = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(PRETRAINED_NAME, **model_kwargs)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


def _is_cuda_runtime_error(exc: RuntimeError) -> bool:
    """Identifie les RuntimeError lies au backend CUDA."""
    msg = str(exc).lower()
    cuda_markers = [
        "cuda",
        "cublas",
        "out of memory",
        "launch failure",
        "device-side assert",
        "unspecified launch failure",
    ]
    return any(marker in msg for marker in cuda_markers)


def save_generation_records(records: List[Dict], seed: int) -> str:
    """Sauvegarde les generations d'un run dans `ex3_results/`."""
    os.makedirs("ex3_results", exist_ok=True)
    path = f"ex3_results/{TASK_NAME}_{MODEL_NAME}_{STRATEGY_NAME}_seed{seed}.json"
    payload = {
        "task": TASK_NAME,
        "model_name": MODEL_NAME,
        "strategy": STRATEGY_NAME,
        "seed": seed,
        "num_records": len(records),
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_mid_generation_checkpoint(
    output_dir: str,
    seed: int,
    run_index: int,
    records: List[Dict],
    processed: int,
    total: int,
) -> str:
    """Sauvegarde un checkpoint intermediaire (50%) pour reprise de generation."""
    path = os.path.join(output_dir, "generation_checkpoint_50pct.json")
    payload = {
        "task": TASK_NAME,
        "model_name": MODEL_NAME,
        "strategy": STRATEGY_NAME,
        "seed": seed,
        "run_index": run_index,
        "checkpoint_pct": 50,
        "processed": processed,
        "total": total,
        "num_records": len(records),
        "records": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def load_mid_generation_checkpoint(output_dir: str, seed: int, run_index: int) -> Tuple[List[Dict], int]:
    """Recharge un checkpoint intermediaire compatible avec le run courant."""
    path = os.path.join(output_dir, "generation_checkpoint_50pct.json")
    if not os.path.exists(path):
        return [], 0
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if str(payload.get("model_name")) != MODEL_NAME:
        return [], 0
    if str(payload.get("strategy")) != STRATEGY_NAME:
        return [], 0
    if int(payload.get("seed", -1)) != int(seed):
        return [], 0
    if int(payload.get("run_index", -1)) != int(run_index):
        return [], 0

    records = payload.get("records", [])
    processed = int(payload.get("processed", len(records)))
    processed = max(0, min(processed, len(records)))
    return records[:processed], processed


def save_run_metrics(run_index: int, metrics: Dict[str, float]) -> None:
    """Sauvegarde les metriques du run dans le fichier partiel benchmark."""
    os.makedirs("benchmark_results", exist_ok=True)
    file_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME}_partial.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {
            "model_name": MODEL_NAME,
            "architecture": ARCHITECTURE,
            "adaptation_technique": ADAPTATION,
            "runs": {},
        }
    data["runs"][str(run_index)] = metrics
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def compute_and_save_statistics() -> None:
    """Agrege tous les runs en moyenne/ecart-type et sauve le fichier final."""
    partial_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME}_partial.json"
    final_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME}_final.json"
    if not os.path.exists(partial_path):
        return
    with open(partial_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    runs = data.get("runs", {})
    if not runs:
        return
    aggregated = {
        "model_name": data["model_name"],
        "architecture": data["architecture"],
        "adaptation_technique": data["adaptation_technique"],
        "runs": runs,
        "metrics": {},
    }
    metric_keys = list(next(iter(runs.values())).keys())
    for key in metric_keys:
        values = [run_data[key] for run_data in runs.values() if isinstance(run_data.get(key), (int, float))]
        if values:
            aggregated["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)


@torch.inference_mode()
def run_generation(
    model,
    tokenizer,
    device,
    eval_dataset: Dataset,
    label_names: List[str],
    output_dir: str,
    seed: int,
    run_index: int,
    batch_size: int,
    start_index: int,
    initial_records: Optional[List[Dict]],
    max_input_tokens: int,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    top_p: float,
) -> Tuple[List[Dict], Dict[str, float]]:
    """Genere les explications sur le split d'eval et calcule les metriques de run."""
    start_time = time.time()
    records: List[Dict] = [dict(r) for r in initial_records] if initial_records else []
    generated_word_counts: List[int] = []
    empty_count = 0
    forbidden_count = 0

    if batch_size <= 0:
        raise ValueError("--batch-size must be >= 1.")

    num_items = len(eval_dataset)
    midpoint_target = max(1, (num_items + 1) // 2)
    midpoint_saved = len(records) >= midpoint_target

    for rec in records:
        # Recalcule les stats de base a partir des enregistrements deja presents (reprise).
        explanation = normalize_text(str(rec.get("generated_explanation", "")))
        words = _word_count(explanation)
        generated_word_counts.append(words)
        if words == 0:
            empty_count += 1
        if _contains_forbidden_advice(explanation):
            forbidden_count += 1

    current_batch_size = batch_size
    start_idx = start_index
    progress = tqdm(total=max(0, num_items - start_index), desc="Ex3 generation", leave=False)
    while start_idx < num_items:
        end_idx = min(start_idx + current_batch_size, num_items)
        batch = eval_dataset[start_idx:end_idx]
        clause_texts = [str(x) for x in batch["text"]]
        label_ids = [int(x) for x in batch["label"]]
        prompts = [build_prompt(clause_text) for clause_text in clause_texts]

        enc = tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            max_length=max_input_tokens,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_lens = enc["attention_mask"].sum(dim=1).tolist()

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = temperature
            generate_kwargs["top_p"] = top_p

        try:
            out = model.generate(**enc, **generate_kwargs)
        except RuntimeError as exc:
            if torch.cuda.is_available() and _is_cuda_runtime_error(exc):
                if current_batch_size > 1:
                    # Auto-recovery: reduction progressive du batch-size en cas d'erreur CUDA.
                    next_bs = max(1, current_batch_size // 2)
                    print(
                        f"[WARN] Run {run_index} CUDA error detectee a l'index {start_idx}. "
                        f"Reduction batch-size: {current_batch_size} -> {next_bs}."
                    )
                    current_batch_size = next_bs
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue

                print(
                    f"[WARN] Run {run_index} CUDA error persistante en batch-size=1. "
                    "Tentative de fallback avec use_cache=False."
                )
                fallback_kwargs = dict(generate_kwargs)
                fallback_kwargs["use_cache"] = False
                out = model.generate(**enc, **fallback_kwargs)
            else:
                raise

        for i in range(len(clause_texts)):
            label_id = label_ids[i]
            label_name = label_names[label_id] if 0 <= label_id < len(label_names) else str(label_id)
            continuation = out[i][int(input_lens[i]):]
            explanation = normalize_text(tokenizer.decode(continuation, skip_special_tokens=True))

            words = _word_count(explanation)
            generated_word_counts.append(words)
            if words == 0:
                empty_count += 1
            if _contains_forbidden_advice(explanation):
                forbidden_count += 1

            records.append(
                {
                    "label_id": label_id,
                    "label_name": label_name,
                    "clause_text": clause_texts[i],
                    "generated_explanation": explanation,
                }
            )

        if not midpoint_saved and len(records) >= midpoint_target:
            checkpoint_path = save_mid_generation_checkpoint(
                output_dir=output_dir,
                seed=seed,
                run_index=run_index,
                records=records,
                processed=len(records),
                total=num_items,
            )
            print(f"[INFO] Run {run_index} checkpoint 50% sauvegarde: {checkpoint_path}")
            midpoint_saved = True

        start_idx = end_idx
        progress.update(len(clause_texts))

    progress.close()

    n = max(1, len(records))
    runtime = time.time() - start_time
    metrics = {
        "num_samples": float(len(records)),
        "runtime_sec": float(runtime),
        "samples_per_sec": float(len(records) / max(runtime, 1e-9)),
        "avg_generated_words": float(np.mean(generated_word_counts) if generated_word_counts else 0.0),
        "empty_rate_pct": float(100.0 * empty_count / n),
        "forbidden_advice_rate_pct": float(100.0 * forbidden_count / n),
    }
    return records, metrics


def main() -> None:
    """Orchestre les runs multi-seeds: generation, sauvegarde, resume et stats."""
    args = parse_args()
    seeds = parse_seeds(args.debug, args.seeds)
    do_sample = not args.greedy

    eval_dataset, label_names, split_name = build_ledgar_eval_split(debug_mode=args.debug)
    print(f"[INFO] {MODEL_NAME}: {len(eval_dataset)} clauses evaluees sur split '{split_name}', strategie={STRATEGY_NAME}.")

    model, tokenizer, device = load_model_and_tokenizer()
    print(f"[INFO] Device: {device}.")
    print(f"[INFO] Batch size: {args.batch_size}.")

    try:
        for run_index, seed in enumerate(seeds):
            set_seed(seed)
            output_dir = f"./checkpoints/{MODEL_KEY}_run_{run_index}"
            os.makedirs(output_dir, exist_ok=True)
            run_completed_file = os.path.join(output_dir, "run_completed.txt")
            if os.path.exists(run_completed_file) and not args.continue_completed:
                print(f"[INFO] Run {run_index} deja complete ({output_dir}); skip.")
                continue

            resume_records, resume_start_idx = load_mid_generation_checkpoint(
                output_dir=output_dir,
                seed=seed,
                run_index=run_index,
            )
            if resume_start_idx > 0:
                print(
                    f"[INFO] Run {run_index} reprise depuis checkpoint 50% "
                    f"({resume_start_idx}/{len(eval_dataset)} clauses)."
                )

            records, run_metrics = run_generation(
                model=model,
                tokenizer=tokenizer,
                device=device,
                eval_dataset=eval_dataset,
                label_names=label_names,
                output_dir=output_dir,
                seed=seed,
                run_index=run_index,
                batch_size=args.batch_size,
                start_index=resume_start_idx,
                initial_records=resume_records,
                max_input_tokens=args.max_input_tokens,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            save_path = save_generation_records(records, seed)
            run_metrics["seed"] = float(seed)
            save_run_metrics(run_index, run_metrics)
            with open(run_completed_file, "w", encoding="utf-8") as f:
                f.write("done")
            checkpoint_file = os.path.join(output_dir, "generation_checkpoint_50pct.json")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            print(f"[INFO] Run {run_index} termine. Generations: {save_path}")
    finally:
        del model
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    if not args.debug:
        compute_and_save_statistics()


if __name__ == "__main__":
    main()
