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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
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

MODEL_KEY = "t5-small-lora"
MODEL_NAME = "t5-small-sum-lora-fast"
PRETRAINED_NAME = "t5-small"
ARCHITECTURE = "Encodeur-Decodeur"
ADAPTATION = "LoRA fine-tuning (PEFT) + warmup cosine"
PROMPT_PREFIX = "summarize legal bill: "
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 1e-4
PER_DEVICE_TRAIN_BATCH_SIZE = 2
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_TRAIN_EPOCHS = 2.0
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.05
LR_SCHEDULER_TYPE = "cosine"
GENERATION_MAX_NEW_TOKENS = 96
GENERATION_NUM_BEAMS = 4
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1


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


def _build_seq2seq_training_arguments(**kwargs) -> Seq2SeqTrainingArguments:
    """Construit Seq2SeqTrainingArguments avec compatibilite inter-versions."""
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in signature and "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return Seq2SeqTrainingArguments(**kwargs)


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


def tokenize_split(dataset: Dataset, tokenizer) -> Dataset:
    """Tokenise un split en format seq2seq (source -> cible)."""
    def preprocess(batch):
        """Prepare un batch tokenise avec labels seq2seq."""
        # Prefix explicite pour stabiliser le comportement instructionnel du modele.
        sources = [PROMPT_PREFIX + normalize_text(text) for text in batch["text"]]
        targets = [normalize_text(summary) for summary in batch["summary"]]
        model_inputs = tokenizer(sources, truncation=True, max_length=MAX_SOURCE_LENGTH)
        labels = tokenizer(text_target=targets, truncation=True, max_length=MAX_TARGET_LENGTH)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names, load_from_cache_file=False, keep_in_memory=True)


def build_test_generation_dataset(test_raw: Dataset, tokenizer) -> Dataset:
    """Construit le dataset dedie a la generation pour l'evaluation test."""
    prompts = [PROMPT_PREFIX + normalize_text(text) for text in test_raw["text"]]
    references = [normalize_text(summary) for summary in test_raw["summary"]]
    generation_dataset = Dataset.from_dict({"prompt": prompts, "reference": references})

    def preprocess(batch):
        """Tokenise uniquement les prompts et conserve les references."""
        model_inputs = tokenizer(batch["prompt"], truncation=True, max_length=MAX_SOURCE_LENGTH)
        model_inputs["reference"] = batch["reference"]
        return model_inputs

    return generation_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["prompt"],
        load_from_cache_file=False,
        keep_in_memory=True,
    )


def build_trainer(output_dir: str, seed: int, model, train_dataset: Dataset, eval_dataset: Optional[Dataset], tokenizer, debug_mode: bool, num_train_epochs_override: Optional[float]):
    """Construit le Seq2SeqTrainer pour l'entrainement LoRA."""
    fp16_enabled = torch.cuda.is_available()
    num_epochs = num_train_epochs_override if num_train_epochs_override is not None else (1.0 if debug_mode else NUM_TRAIN_EPOCHS)
    args = _build_seq2seq_training_arguments(
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
        predict_with_generate=False,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, pad_to_multiple_of=8 if fp16_enabled else None)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "data_collator": collator,
        "compute_metrics": None,
    }
    if eval_dataset is not None:
        trainer_kwargs["eval_dataset"] = eval_dataset
    signature = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    return Seq2SeqTrainer(**trainer_kwargs)


@torch.inference_mode()
def evaluate_on_test(model, tokenizer, test_generation_dataset: Dataset, generation_max_new_tokens_override: Optional[int]) -> Dict[str, float]:
    """Genere les resumes sur test et calcule les metriques."""
    model.eval()
    device = next(model.parameters()).device
    max_new = generation_max_new_tokens_override if generation_max_new_tokens_override is not None else GENERATION_MAX_NEW_TOKENS
    predictions, references = [], []
    total_batches = (len(test_generation_dataset) + PER_DEVICE_EVAL_BATCH_SIZE - 1) // PER_DEVICE_EVAL_BATCH_SIZE
    previous_use_cache = getattr(model.config, "use_cache", None)
    if previous_use_cache is not None:
        model.config.use_cache = True
    try:
        for start in tqdm(
            range(0, len(test_generation_dataset), PER_DEVICE_EVAL_BATCH_SIZE),
            total=total_batches,
            desc="Test generation",
            leave=False,
        ):
            end = min(start + PER_DEVICE_EVAL_BATCH_SIZE, len(test_generation_dataset))
            batch = test_generation_dataset[start:end]
            refs = batch["reference"]
            features = [
                {"input_ids": input_ids, "attention_mask": attention_mask}
                for input_ids, attention_mask in zip(batch["input_ids"], batch["attention_mask"])
            ]
            enc = tokenizer.pad(features, padding=True, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            generated = model.generate(
                **enc,
                max_new_tokens=max_new,
                num_beams=GENERATION_NUM_BEAMS,
                do_sample=False,
                # Generation deterministe pour comparer proprement les runs.
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            decoded = [normalize_text(x) for x in tokenizer.batch_decode(generated, skip_special_tokens=True)]
            predictions.extend(decoded)
            references.extend(refs)
    finally:
        if previous_use_cache is not None:
            model.config.use_cache = previous_use_cache
    return compute_summarization_metrics(predictions, references)


def parse_args() -> argparse.Namespace:
    """Parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(description="Exercise 2 - t5-small-lora")
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
    seeds = parse_seeds(args.debug, args.seeds)
    train_raw, eval_raw, test_raw = build_raw_splits(args.debug, args.max_train_samples, args.max_test_samples)
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_NAME, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = tokenize_split(train_raw, tokenizer)
    eval_dataset = tokenize_split(eval_raw, tokenizer) if eval_raw is not None else None
    test_generation_dataset = build_test_generation_dataset(test_raw, tokenizer)
    for run_index, seed in enumerate(seeds):
        set_seed(seed)
        output_dir = f"./checkpoints/{MODEL_KEY}_run_{run_index}"
        os.makedirs(output_dir, exist_ok=True)
        run_completed_file = os.path.join(output_dir, "run_completed.txt")
        if os.path.exists(run_completed_file) and not args.continue_completed:
            print(f"[INFO] Run {run_index} already completed in {output_dir}; skipping.")
            continue
        model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_NAME)
        if get_peft_model is None:
            raise ImportError("peft is required for this script (`pip install peft`).")
        peft_cfg = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q", "k", "v", "o"],
            bias="none",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()
        model.config.use_cache = False
        trainer = build_trainer(output_dir, seed, model, train_dataset, eval_dataset, tokenizer, args.debug, args.num_train_epochs)
        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        trainer.train(resume_from_checkpoint=last_checkpoint) if last_checkpoint else trainer.train()
        final_metrics = evaluate_on_test(trainer.model, tokenizer, test_generation_dataset, args.generation_max_new_tokens)
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
