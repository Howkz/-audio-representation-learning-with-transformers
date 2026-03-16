import argparse
import inspect
import json
import os
import random
import time

# Force l'usage exclusif de PyTorch pour eviter les imports TensorFlow inutiles.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "bert-base-uncased-linear-probing-full-fast"
PRETRAINED_MODEL_NAME = "bert-base-uncased"
MODEL_NAME_SHORT = "bert_linear_probing_full_fast"
ARCHITECTURE = "Encodeur"
ADAPTATION = "Linear probing (BERT encodeur gele, full dataset, speed-optimized)"
TASK_NAME = "ex1"

MAX_LENGTH = 128
DEBUG_MODE = False
DEBUG_SAMPLES = 100
SEEDS = [42, 123, 456]

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
# Linear probing strict: seule la tete est entrainee (initialisation aleatoire),
# donc il faut un LR plus eleve et plus d'epochs que le full fine-tuning.
LEARNING_RATE = 2e-3
NUM_TRAIN_EPOCHS = 2.0
WEIGHT_DECAY = 0.0
LOGGING_STEPS = 100
EARLY_STOPPING_PATIENCE = 1


def set_seed(seed: int):
    """Initialise toutes les seeds pour un comportement reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Determinisme strict pour le rendu TP.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def accuracy_score_np(labels, predictions):
    """Calcule l'accuracy a partir de labels et predictions discretes."""
    # Accuracy simple en NumPy pour eviter une dependance supplementaire (sklearn).
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if labels.size == 0:
        return 0.0
    return float(np.mean(labels == predictions))


def f1_macro_np(labels, predictions):
    """Calcule la F1 macro sans dependance externe."""
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    classes = np.unique(np.concatenate([labels, predictions]))
    if classes.size == 0:
        return 0.0

    f1_values = []
    for cls in classes:
        tp = np.sum((predictions == cls) & (labels == cls))
        fp = np.sum((predictions == cls) & (labels != cls))
        fn = np.sum((predictions != cls) & (labels == cls))
        denom = (2 * tp) + fp + fn
        f1_values.append(0.0 if denom == 0 else float((2 * tp) / denom))
    return float(np.mean(f1_values))


def binary_roc_auc_np(binary_labels, scores):
    """Calcule l'AUC binaire a partir de scores continus."""
    # Calcul manuel de l'AUC via la courbe ROC (ordre decroissant des scores).
    binary_labels = np.asarray(binary_labels).astype(np.int32)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(np.sum(binary_labels))
    negatives = int(binary_labels.shape[0] - positives)
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = binary_labels[order]

    distinct_value_indices = np.where(np.diff(sorted_scores))[0]
    threshold_indices = np.r_[distinct_value_indices, sorted_labels.size - 1]

    tps = np.cumsum(sorted_labels)[threshold_indices]
    fps = (1 + threshold_indices) - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    tpr = tps / positives
    fpr = fps / negatives
    return float(np.trapz(tpr, fpr))


def roc_auc_ovr_np(labels, probs):
    """Calcule l'AUROC multiclasse en mode one-vs-rest."""
    # Multiclasse = moyenne macro des AUC "un contre tous".
    labels = np.asarray(labels)
    probs = np.asarray(probs)
    num_classes = probs.shape[1]
    unique_labels = np.unique(labels)
    if unique_labels.size != num_classes:
        raise ValueError("AUROC OVR requires all classes to be present in labels.")

    aucs = []
    for class_idx in range(num_classes):
        class_labels = (labels == class_idx).astype(np.int32)
        auc = binary_roc_auc_np(class_labels, probs[:, class_idx])
        if np.isnan(auc):
            raise ValueError("AUROC undefined for at least one class.")
        aucs.append(auc)
    return float(np.mean(aucs))


def compute_metrics_from_logits(logits, labels):
    """Convertit les logits en metriques de classification."""
    # Les logits bruts sont convertis en predictions discretes et probabilites.
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score_np(labels, predictions)
    f1 = f1_macro_np(labels, predictions)

    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    try:
        auroc = roc_auc_ovr_np(labels, probs)
    except ValueError:
        auroc = float("nan")
    return {"accuracy": acc, "f1_macro": f1, "auroc": auroc}


import os
import json
import numpy as np


def save_run_metrics(metrics, run_index, model_name, architecture, adaptation, task_name):
    """
    Sauvegarde les metriques d'un run individuel de maniere incrementale.
    Protege contre la perte de donnees en cas de crash.
    """
    os.makedirs("benchmark_results", exist_ok=True)
    file_path = f"benchmark_results/{task_name}_{model_name.replace('/', '-')}_partial.json"

    # Charger le fichier existant si d'autres runs ont deja ete completes
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
    else:
        data = {
            "model_name": model_name,
            "architecture": architecture,
            "adaptation_technique": adaptation,
            "runs": {}
        }

    # Ajouter ou ecraser les metriques du run actuel
    data["runs"][str(run_index)] = metrics

    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Metriques du run {run_index} sauvegardees en securite dans {file_path}")


def compute_and_save_statistics(model_name, task_name):
    """
    Lit le fichier partiel contenant les runs individuels,
    calcule la moyenne et l'ecart-type, et sauvegarde le JSON final agrege.
    """
    partial_path = f"benchmark_results/{task_name}_{model_name.replace('/', '-')}_partial.json"
    final_path = f"benchmark_results/{task_name}_{model_name.replace('/', '-')}_final.json"

    if not os.path.exists(partial_path):
        print("Erreur : Aucun resultat partiel trouve pour calculer les statistiques.")
        return

    with open(partial_path, "r") as f:
        data = json.load(f)

    runs = data["runs"]
    if not runs:
        print("Erreur : Le fichier partiel ne contient aucun run.")
        return

    aggregated = {
        "model_name": data["model_name"],
        "architecture": data["architecture"],
        "adaptation_technique": data["adaptation_technique"],
        "runs": runs,
        "metrics": {}
    }

    # Recuperer les cles de la premiere metrique
    first_run_key = list(runs.keys())[0]
    metric_keys = runs[first_run_key].keys()

    print(f"\n--- Resultats Finaux Agreges ({len(runs)} runs) ---")
    for key in metric_keys:
        values = [run_data[key] for run_data in runs.values() if isinstance(run_data[key], (int, float))]
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            aggregated["metrics"][key] = {"mean": mean_val, "std": std_val}
            print(f"{key} : {mean_val:.4f} +- {std_val:.4f}")

    with open(final_path, "w") as f:
        json.dump(aggregated, f, indent=4)

    print(f"Resultats finaux calcules et sauvegardes dans {final_path}")


class ThroughputCallback(TrainerCallback):
    def __init__(self, effective_batch_size: int):
        """Initialise l'etat interne pour suivre le debit d'entrainement."""
        self.effective_batch_size = effective_batch_size
        self.start_time = None
        self.last_time = None
        self.last_step = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Demarre le chronometrage au debut du training."""
        now = time.time()
        self.start_time = now
        self.last_time = now
        self.last_step = state.global_step
        print(f"[speed] max_steps={state.max_steps} | effective_batch={self.effective_batch_size}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Affiche periodiquement steps/s, exemples/s et ETA."""
        if self.start_time is None or state.global_step <= self.last_step:
            return
        now = time.time()
        step_delta = state.global_step - self.last_step
        time_delta = now - self.last_time
        if time_delta <= 0:
            return
        steps_per_sec = step_delta / time_delta
        examples_per_sec = steps_per_sec * self.effective_batch_size
        elapsed_min = (now - self.start_time) / 60.0
        if steps_per_sec > 0 and state.max_steps:
            remaining = max(state.max_steps - state.global_step, 0)
            eta_min = (remaining / steps_per_sec) / 60.0
            eta_text = f"{eta_min:.1f} min"
        else:
            eta_text = "N/A"
        print(
            f"[speed] step {state.global_step}/{state.max_steps} | "
            f"{steps_per_sec:.2f} steps/s | {examples_per_sec:.1f} ex/s | "
            f"elapsed {elapsed_min:.1f} min | ETA {eta_text}"
        )
        self.last_step = state.global_step
        self.last_time = now


def parse_args():
    """Parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(description="Exercice 1 - BERT linear probing full-fast sur LEDGAR")
    parser.add_argument("--debug", action="store_true", help="Mode debug: 1 seed et 100 exemples max par split.")
    parser.add_argument(
        "--num-train-epochs",
        type=float,
        default=None,
        help="Override des epochs totales (ex: 2.0).",
    )
    parser.add_argument(
        "--continue-completed",
        action="store_true",
        help="Reprendre meme les runs marques completes a partir du dernier checkpoint.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=MAX_LENGTH,
        help="Longueur max tokenisation (defaut: 128).",
    )
    return parser.parse_args()


def _build_training_arguments(**kwargs):
    """Construit TrainingArguments avec compatibilite inter-versions."""
    # Compatibilite descendante entre versions Transformers:
    # certaines utilisent `evaluation_strategy`, d'autres `eval_strategy`.
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in signature and "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def load_ledgar():
    """Charge LEDGAR et retourne dataset + metadonnees de labels."""
    # Chargement centralise du dataset pour partager exactement la meme source
    # entre full fine-tuning, LoRA et linear probing.
    dataset = load_dataset("lex_glue", "ledgar")
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    return dataset, label_names, num_labels


def tokenize_splits(dataset, tokenizer, debug_mode, max_length):
    """Tokenise les splits train/validation/test pour la classification."""
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    if debug_mode:
        # En debug: sous-ensemble petit et deterministe pour iterer rapidement.
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLES, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(DEBUG_SAMPLES, len(eval_dataset))))
        test_dataset = test_dataset.select(range(min(DEBUG_SAMPLES, len(test_dataset))))

    def preprocess(batch):
        """Prepare un batch tokenise avec la cle `labels`."""
        # On stocke explicitement la cible dans "labels" (attendu par Trainer).
        model_inputs = tokenizer(batch["text"], truncation=True, max_length=max_length)
        model_inputs["labels"] = batch["label"]
        return model_inputs

    map_kwargs = {
        "batched": True,
        "load_from_cache_file": False,
        "keep_in_memory": True,
    }
    train_dataset = train_dataset.map(preprocess, remove_columns=train_dataset.column_names, **map_kwargs)
    eval_dataset = eval_dataset.map(preprocess, remove_columns=eval_dataset.column_names, **map_kwargs)
    test_dataset = test_dataset.map(preprocess, remove_columns=test_dataset.column_names, **map_kwargs)
    return train_dataset, eval_dataset, test_dataset


def build_model(num_labels, label_names):
    """Instancie BERT puis gele l'encodeur pour linear probing."""
    # Mapping utile pour logs/evaluations lisibles et sauvegarde du modele.
    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        PRETRAINED_MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )

    # Linear probing: on gele tout BERT et on entraine uniquement la tete de classification.
    for param in model.parameters():
        param.requires_grad = False
    if hasattr(model, "classifier"):
        for param in model.classifier.parameters():
            param.requires_grad = True
    elif hasattr(model, "score"):
        for param in model.score.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Impossible de trouver la tete de classification pour le linear probing.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Linear probing: {trainable}/{total} parametres entrainables ({100.0 * trainable / total:.2f}%).")
    return model


def build_trainer(
    output_dir,
    seed,
    model,
    train_dataset,
    eval_dataset,
    tokenizer,
    debug_mode,
    num_train_epochs_override=None,
):
    """Construit le Trainer configure pour linear probing."""
    # fp16 active seulement sur GPU pour accelerer l'entrainement sans casser CPU.
    fp16_enabled = torch.cuda.is_available()
    logging_steps = 10 if debug_mode else LOGGING_STEPS
    num_epochs = num_train_epochs_override if num_train_epochs_override is not None else (1 if debug_mode else NUM_TRAIN_EPOCHS)

    training_args = _build_training_arguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        num_train_epochs=num_epochs,
        weight_decay=WEIGHT_DECAY,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=fp16_enabled,
        dataloader_num_workers=2,
        dataloader_pin_memory=True,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        # Padding multiple de 8: meilleur alignement memoire avec kernels GPU/fp16.
        pad_to_multiple_of=8 if fp16_enabled else None,
    )

    world_size = max(1, getattr(training_args, "world_size", 1))
    effective_batch_size = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS * world_size
    speed_callback = ThroughputCallback(effective_batch_size=effective_batch_size)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        # Pas de compute_metrics pendant le train: acceleration importante.
        "compute_metrics": None,
        "callbacks": [
            EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE),
            speed_callback,
        ],
    }
    trainer_signature = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    return trainer


def evaluate_on_test(trainer, test_dataset):
    """Evalue le modele sur le split test."""
    # Evaluation finale hors boucle d'entrainement, avec toutes les metriques.
    predictions = trainer.predict(test_dataset)
    logits = predictions.predictions
    labels = predictions.label_ids
    return compute_metrics_from_logits(logits, labels)


def main():
    """Execute la boucle complete d'entrainement/evaluation multi-seeds."""
    args = parse_args()
    debug_mode = DEBUG_MODE or args.debug
    # En debug: une seule seed pour gagner du temps de developpement.
    seeds = [42] if debug_mode else SEEDS

    dataset, label_names, num_labels = load_ledgar()
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL_NAME, use_fast=True)
    train_dataset, eval_dataset, test_dataset = tokenize_splits(dataset, tokenizer, debug_mode, args.max_length)

    for run, seed in enumerate(seeds):
        print(f"--- Debut du Run {run + 1}/{len(seeds)} avec la seed {seed} ---")
        set_seed(seed)
        output_dir = f"./checkpoints/{MODEL_NAME_SHORT}_run_{run}"
        os.makedirs(output_dir, exist_ok=True)

        run_completed_file = os.path.join(output_dir, "run_completed.txt")
        # Marqueur simple de completion pour rendre le script "resume-safe".
        if os.path.exists(run_completed_file) and not args.continue_completed:
            print(f"Run {run + 1} deja termine, passage au suivant.")
            continue
        if os.path.exists(run_completed_file) and args.continue_completed:
            print(f"Run {run + 1} marque termine, reprise forcee activee (--continue-completed).")

        model = build_model(num_labels, label_names)
        trainer = build_trainer(
            output_dir=output_dir,
            seed=seed,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            debug_mode=debug_mode,
            num_train_epochs_override=args.num_train_epochs,
        )

        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        # Reprise automatique si un checkpoint existe deja dans le dossier du run.
        if last_checkpoint is not None:
            print(f"Reprise a partir du checkpoint : {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        print(f"Evaluation finale du run {run} sur le jeu de test complet...")
        final_metrics = evaluate_on_test(trainer, test_dataset)
        save_run_metrics(
            metrics=final_metrics,
            run_index=run,
            model_name=MODEL_NAME,
            architecture=ARCHITECTURE,
            adaptation=ADAPTATION,
            task_name=TASK_NAME,
        )

        with open(run_completed_file, "w") as f:
            f.write("done")

        del trainer
        del model
        # Limite les OOM entre runs quand plusieurs seeds sont enchainees.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not debug_mode:
        compute_and_save_statistics(model_name=MODEL_NAME, task_name=TASK_NAME)


if __name__ == "__main__":
    main()
