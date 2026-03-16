import argparse
import inspect
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint

try:
    from peft import LoraConfig, TaskType, get_peft_model
except ImportError:
    LoraConfig = None
    TaskType = None
    get_peft_model = None


TASK_NAME = "ex2"
SEEDS = [42, 123, 456]
DEBUG_SAMPLES = 200
VAL_SPLIT_SEED = 42
LOGGING_STEPS = 50

MODEL_KEY = "gpt2-lora"
MODEL_NAME = "gpt2-sum-lora-fast"
PRETRAINED_NAME = "gpt2"
ARCHITECTURE = "Decodeur uniquement"
ADAPTATION = "Prompted causal summarization + LoRA (parent)"
PROMPT_TEMPLATE = "Summarize the following legal bill in plain English.\n\n{text}\n\nSummary:"
MAX_SOURCE_LENGTH = 320
MAX_TARGET_LENGTH = 96
MAX_TOTAL_LENGTH = 448
LEARNING_RATE = 2e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 2.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
GENERATION_MAX_NEW_TOKENS = 96
GENERATION_NUM_BEAMS = 1
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1
TEST_GENERATION_CHECKPOINT_RATIO = 0.5
TEST_GENERATION_CHECKPOINT_FILE = "test_generation_progress.json"


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
    """Normalise un texte (trim, lower, espaces)."""
    return " ".join(str(text).strip().lower().split())


def _tokenize_for_metrics(text: str) -> List[str]:
    """Tokenise legerement un texte normalise pour les metriques."""
    return normalize_text(text).split()


def _build_ngrams(tokens: List[str], n: int) -> Counter:
    """Construit un compteur de n-grammes."""
    if len(tokens) < n or n <= 0:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def rouge_n_f1_single(prediction: str, reference: str, n: int) -> float:
    """Calcule ROUGE-N F1 pour une paire prediction/reference."""
    pred_ngrams = _build_ngrams(_tokenize_for_metrics(prediction), n)
    ref_ngrams = _build_ngrams(_tokenize_for_metrics(reference), n)
    if not pred_ngrams or not ref_ngrams:
        return 0.0
    overlap = sum(min(pred_count, ref_ngrams.get(gram, 0)) for gram, pred_count in pred_ngrams.items())
    pred_total = sum(pred_ngrams.values())
    ref_total = sum(ref_ngrams.values())
    if pred_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0
    precision = overlap / pred_total
    recall = overlap / ref_total
    denom = precision + recall
    return 0.0 if denom == 0 else float((2.0 * precision * recall) / denom)


def rouge_n_f1_corpus(predictions: List[str], references: List[str], n: int) -> float:
    """Calcule ROUGE-N F1 moyen sur tout le corpus."""
    if not predictions:
        return 0.0
    return float(np.mean([rouge_n_f1_single(pred, ref, n) for pred, ref in zip(predictions, references)]))


def bleu4_corpus(predictions: List[str], references: List[str]) -> float:
    """Calcule un BLEU-4 corpus-level avec lissage simple."""
    if not predictions:
        return 0.0
    total_pred_length = 0
    total_ref_length = 0
    precisions = []
    smoothing = 1e-12
    for n in (1, 2, 3, 4):
        clipped_count = 0
        total_count = 0
        for pred, ref in zip(predictions, references):
            pred_tokens = _tokenize_for_metrics(pred)
            ref_tokens = _tokenize_for_metrics(ref)
            pred_ngrams = _build_ngrams(pred_tokens, n)
            ref_ngrams = _build_ngrams(ref_tokens, n)
            clipped_count += sum(min(count, ref_ngrams.get(gram, 0)) for gram, count in pred_ngrams.items())
            total_count += sum(pred_ngrams.values())
            if n == 1:
                total_pred_length += len(pred_tokens)
                total_ref_length += len(ref_tokens)
        precisions.append((clipped_count + smoothing) / (total_count + smoothing))
    if total_pred_length == 0:
        return 0.0
    bp = 1.0 if total_pred_length > total_ref_length else math.exp(1.0 - (total_ref_length / total_pred_length))
    bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
    return float(bleu * 100.0)


def compute_summarization_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Retourne BLEU, ROUGE-1 et ROUGE-2 pour les resumes generes."""
    return {
        "bleu": bleu4_corpus(predictions, references),
        "rouge1": rouge_n_f1_corpus(predictions, references, n=1) * 100.0,
        "rouge2": rouge_n_f1_corpus(predictions, references, n=2) * 100.0,
    }


def save_run_metrics(metrics: Dict[str, float], run_index: int) -> None:
    """Sauvegarde les metriques d'un run dans le JSON partiel."""
    os.makedirs("benchmark_results", exist_ok=True)
    file_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME.replace('/', '-')}_partial.json"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"model_name": MODEL_NAME, "architecture": ARCHITECTURE, "adaptation_technique": ADAPTATION, "runs": {}}
    data["runs"][str(run_index)] = metrics
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def compute_and_save_statistics() -> None:
    """Agrege les runs partiels en moyenne/ecart-type et sauve le JSON final."""
    partial_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME.replace('/', '-')}_partial.json"
    final_path = f"benchmark_results/{TASK_NAME}_{MODEL_NAME.replace('/', '-')}_final.json"
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
        values = [run_data[key] for run_data in runs.values() if isinstance(run_data[key], (int, float))]
        if values:
            aggregated["metrics"][key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=4)


def _build_training_arguments(**kwargs) -> TrainingArguments:
    """Construit TrainingArguments avec compatibilite inter-versions."""
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in signature and "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def load_billsum_dataset() -> Dict[str, Dataset]:
    """Charge BillSum depuis le hub avec fallback de nom."""
    try:
        return load_dataset("FiscalNote/billsum")
    except Exception:
        return load_dataset("billsum")


def maybe_cap_dataset(dataset: Dataset, max_samples: Optional[int], shuffle_seed: int) -> Dataset:
    """Limite eventuellement un dataset apres melange."""
    if max_samples is None:
        return dataset
    return dataset.shuffle(seed=shuffle_seed).select(range(min(max_samples, len(dataset))))


def build_raw_splits(debug_mode: bool, max_train_samples: Optional[int], max_test_samples: Optional[int]) -> Tuple[Dataset, Optional[Dataset], Dataset]:
    """Construit les splits bruts train/(eval)/test selon debug et caps."""
    ds = load_billsum_dataset()
    test_split_name = "test" if "test" in ds else "validation"
    train_full = ds["train"].shuffle(seed=VAL_SPLIT_SEED)
    test_raw = ds[test_split_name].shuffle(seed=VAL_SPLIT_SEED)
    train_raw = train_full
    eval_raw = None
    if debug_mode:
        train_raw = maybe_cap_dataset(train_raw, DEBUG_SAMPLES, VAL_SPLIT_SEED)
        test_raw = maybe_cap_dataset(test_raw, DEBUG_SAMPLES, VAL_SPLIT_SEED)
    else:
        train_raw = maybe_cap_dataset(train_raw, max_train_samples, VAL_SPLIT_SEED)
        test_raw = maybe_cap_dataset(test_raw, max_test_samples, VAL_SPLIT_SEED)
    return train_raw, eval_raw, test_raw


def build_causal_prompt(text: str) -> str:
    """Construit le prompt causal pour la summarization."""
    return PROMPT_TEMPLATE.format(text=normalize_text(text))


def tokenize_split(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenise un split en format causal LM supervise."""
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id

    def preprocess(batch):
        """Prepare input_ids/labels avec masquage de la partie prompt."""
        all_input_ids, all_attention_mask, all_labels = [], [], []
        for text, summary in zip(batch["text"], batch["summary"]):
            prompt = build_causal_prompt(text)
            # Prompt et resume sont tokenises separement pour masquer la loss sur le prompt.
            prompt_ids = tokenizer(prompt, truncation=True, max_length=MAX_SOURCE_LENGTH, add_special_tokens=False)["input_ids"]
            summary_ids = tokenizer(normalize_text(summary), truncation=True, max_length=MAX_TARGET_LENGTH, add_special_tokens=False)["input_ids"]
            input_ids = prompt_ids + summary_ids + [eos_token_id]
            labels = ([-100] * len(prompt_ids)) + summary_ids + [eos_token_id]
            if len(input_ids) > MAX_TOTAL_LENGTH:
                # Coupe de securite pour rester sous la longueur max du modele.
                input_ids = input_ids[:MAX_TOTAL_LENGTH]
                labels = labels[:MAX_TOTAL_LENGTH]
            all_input_ids.append(input_ids)
            all_attention_mask.append([1] * len(input_ids))
            all_labels.append(labels)
        return {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "labels": all_labels}

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names, load_from_cache_file=False, keep_in_memory=True)


class CausalDataCollator:
    def __init__(self, tokenizer, pad_to_multiple_of: Optional[int]):
        """Initialise le collator causal avec padding dynamique."""
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Padde inputs et labels en preservant les -100 pour la loss."""
        model_features = [{"input_ids": f["input_ids"], "attention_mask": f["attention_mask"]} for f in features]
        labels = [f["labels"] for f in features]
        batch = self.tokenizer.pad(model_features, padding=True, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors="pt")
        max_len = batch["input_ids"].shape[1]
        left_padding = self.tokenizer.padding_side == "left"
        padded_labels = []
        for label in labels:
            pad_len = max_len - len(label)
            padded_labels.append(([-100] * pad_len + label) if left_padding else (label + [-100] * pad_len))
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        return batch


def build_trainer(output_dir: str, seed: int, model, train_dataset: Dataset, eval_dataset: Optional[Dataset], tokenizer, debug_mode: bool, num_train_epochs_override: Optional[float]):
    """Construit le Trainer pour l'entrainement causal LoRA."""
    fp16_enabled = torch.cuda.is_available()
    num_epochs = num_train_epochs_override if num_train_epochs_override is not None else (1.0 if debug_mode else NUM_TRAIN_EPOCHS)
    args = _build_training_arguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=num_epochs,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="no",  # Pas d'eval intermediaire: on privilegie le debit d'entrainement.
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=False,
        logging_strategy="steps",
        logging_steps=10 if debug_mode else LOGGING_STEPS,
        fp16=fp16_enabled,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    collator = CausalDataCollator(tokenizer, 8 if fp16_enabled else None)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": collator,
        "compute_metrics": None,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Trainer(**trainer_kwargs)


def extract_summary_from_generated_text(prompt: str, generated_text: str) -> str:
    """Extrait la partie resume du texte genere."""
    lower = generated_text.lower()
    marker = "summary:"
    if marker in lower:
        idx = lower.rfind(marker)
        return generated_text[idx + len(marker) :].strip()
    if generated_text.startswith(prompt):
        return generated_text[len(prompt) :].strip()
    return generated_text.strip()


def load_generation_progress(progress_path: str, expected_total_samples: int, expected_max_new_tokens: int) -> Tuple[List[str], List[str], int]:
    """Charge un checkpoint de generation partielle pour reprise test."""
    if not os.path.exists(progress_path):
        return [], [], 0
    try:
        with open(progress_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        print(f"[WARN] Impossible de lire le checkpoint de generation ({progress_path}): {exc}. Reprise depuis 0.")
        return [], [], 0

    saved_total = int(data.get("total_samples", -1))
    saved_max_new = int(data.get("max_new_tokens", -1))
    processed = int(data.get("processed_samples", 0))
    predictions = data.get("predictions", [])
    references = data.get("references", [])

    if saved_total != expected_total_samples or saved_max_new != expected_max_new_tokens:
        print("[INFO] Checkpoint de generation incompatible (taille test ou max_new_tokens differents). Reprise depuis 0.")
        return [], [], 0
    if not isinstance(predictions, list) or not isinstance(references, list):
        print("[WARN] Checkpoint de generation invalide (types). Reprise depuis 0.")
        return [], [], 0
    if processed != len(predictions) or processed != len(references):
        print("[WARN] Checkpoint de generation invalide (longueurs). Reprise depuis 0.")
        return [], [], 0
    if processed < 0 or processed > expected_total_samples:
        print("[WARN] Checkpoint de generation invalide (processed_samples). Reprise depuis 0.")
        return [], [], 0

    if processed > 0:
        print(f"[INFO] Reprise generation test: {processed}/{expected_total_samples} exemples deja completes.")
    return predictions, references, processed


def save_generation_progress(
    progress_path: str,
    predictions: List[str],
    references: List[str],
    processed_samples: int,
    total_samples: int,
    max_new_tokens: int,
) -> None:
    """Sauvegarde l'avancement de generation test sur disque."""
    payload = {
        "processed_samples": int(processed_samples),
        "total_samples": int(total_samples),
        "max_new_tokens": int(max_new_tokens),
        "predictions": predictions,
        "references": references,
    }
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)


@torch.no_grad()
def evaluate_on_test(
    model,
    tokenizer,
    test_raw: Dataset,
    generation_max_new_tokens_override: Optional[int],
    generation_progress_path: Optional[str] = None,
) -> Dict[str, float]:
    """Genere les resumes sur test (avec reprise) puis calcule les metriques."""
    model.eval()
    device = next(model.parameters()).device
    max_new = generation_max_new_tokens_override if generation_max_new_tokens_override is not None else GENERATION_MAX_NEW_TOKENS
    total_samples = len(test_raw)
    halfway_sample = max(1, int(math.ceil(total_samples * TEST_GENERATION_CHECKPOINT_RATIO)))
    if generation_progress_path:
        predictions, references, start_sample = load_generation_progress(
            generation_progress_path,
            expected_total_samples=total_samples,
            expected_max_new_tokens=max_new,
        )
    else:
        predictions, references, start_sample = [], [], 0

    total_batches = (total_samples + PER_DEVICE_EVAL_BATCH_SIZE - 1) // PER_DEVICE_EVAL_BATCH_SIZE
    saved_halfway = start_sample >= halfway_sample

    for start in tqdm(
        range(start_sample, total_samples, PER_DEVICE_EVAL_BATCH_SIZE),
        total=total_batches,
        initial=start_sample // PER_DEVICE_EVAL_BATCH_SIZE,
        desc="Test generation",
        leave=False,
    ):
        end = min(start + PER_DEVICE_EVAL_BATCH_SIZE, total_samples)
        batch = test_raw[start:end]
        prompts = [build_causal_prompt(text) for text in batch["text"]]
        refs = [normalize_text(summary) for summary in batch["summary"]]
        enc = tokenizer(prompts, truncation=True, max_length=MAX_SOURCE_LENGTH, padding=True, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}
        generated = model.generate(
            **enc,
            max_new_tokens=max_new,
            num_beams=GENERATION_NUM_BEAMS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        for prompt, generated_text in zip(prompts, decoded):
            # Nettoie la sortie brute pour ne garder que le resume.
            predictions.append(normalize_text(extract_summary_from_generated_text(prompt, generated_text)))
        references.extend(refs)

        processed_samples = len(predictions)
        if generation_progress_path and not saved_halfway and processed_samples >= halfway_sample:
            save_generation_progress(
                generation_progress_path,
                predictions=predictions,
                references=references,
                processed_samples=processed_samples,
                total_samples=total_samples,
                max_new_tokens=max_new,
            )
            saved_halfway = True
            print(
                f"[INFO] Checkpoint generation test sauvegarde a {processed_samples}/{total_samples} "
                f"({100.0 * processed_samples / max(1, total_samples):.1f}%)."
            )

    if generation_progress_path and os.path.exists(generation_progress_path):
        os.remove(generation_progress_path)
    return compute_summarization_metrics(predictions, references)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(description="Exercise 2 - gpt2-lora")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--continue-completed", action="store_true")
    parser.add_argument("--num-train-epochs", type=float, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--generation-max-new-tokens", type=int, default=None)
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


def main() -> None:
    """Execute le pipeline complet: train, generation test, sauvegarde metriques."""
    args = parse_args()
    if get_peft_model is None:
        raise ImportError("peft is required for this script (`pip install peft`).")
    seeds = parse_seeds(args.debug, args.seeds)
    train_raw, eval_raw, test_raw = build_raw_splits(args.debug, args.max_train_samples, args.max_test_samples)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    train_dataset = tokenize_split(train_raw, tokenizer)
    eval_dataset = tokenize_split(eval_raw, tokenizer) if eval_raw is not None else None
    for run_index, seed in enumerate(seeds):
        set_seed(seed)
        output_dir = f"./checkpoints/{MODEL_KEY}_run_{run_index}"
        os.makedirs(output_dir, exist_ok=True)
        run_completed_file = os.path.join(output_dir, "run_completed.txt")
        if os.path.exists(run_completed_file) and not args.continue_completed:
            print(f"[INFO] Run {run_index} already completed in {output_dir}; skipping.")
            continue
        model = AutoModelForCausalLM.from_pretrained(PRETRAINED_NAME)
        peft_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["c_attn", "c_proj"],
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        model.config.use_cache = False
        model.config.pad_token_id = tokenizer.pad_token_id
        trainer = build_trainer(output_dir, seed, model, train_dataset, eval_dataset, tokenizer, args.debug, args.num_train_epochs)
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        trainer.train(resume_from_checkpoint=last_checkpoint) if last_checkpoint else trainer.train()
        print(f"[INFO] Run {run_index}: generation sur test en cours...")
        generation_progress_path = os.path.join(output_dir, TEST_GENERATION_CHECKPOINT_FILE)
        final_metrics = evaluate_on_test(
            trainer.model,
            tokenizer,
            test_raw,
            args.generation_max_new_tokens,
            generation_progress_path=generation_progress_path,
        )
        save_run_metrics(final_metrics, run_index)
        with open(run_completed_file, "w", encoding="utf-8") as f:
            f.write("done")
        del trainer
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not args.debug:
        compute_and_save_statistics()


if __name__ == "__main__":
    main()

