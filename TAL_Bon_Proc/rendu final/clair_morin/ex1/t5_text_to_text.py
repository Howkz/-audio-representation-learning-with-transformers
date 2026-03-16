import argparse
import inspect
import json
import os
import random
from difflib import get_close_matches

# Force l'usage exclusif de PyTorch pour eviter les imports TensorFlow inutiles.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "t5-small"
MODEL_NAME_SHORT = "t5"
ARCHITECTURE = "Encodeur-Decodeur"
ADAPTATION = "Text-to-text full fine-tuning"
TASK_NAME = "ex1"

MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 32
GENERATED_LABEL_MAX_LENGTH = 32
DEBUG_MODE = False
DEBUG_SAMPLES = 100
SEEDS = [42, 123, 456]

PER_DEVICE_TRAIN_BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 1
LEARNING_RATE = 5e-5
NUM_TRAIN_EPOCHS = 3
WEIGHT_DECAY = 0.01
EVAL_STEPS = 500
SAVE_STEPS = 500
LOGGING_STEPS = 100
EARLY_STOPPING_PATIENCE = 3


def set_seed(seed: int):
    """Initialise toutes les seeds pour un comportement reproductible."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Assurer le determinisme (peut ralentir legerement mais obligatoire ici)
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
        # Extraire toutes les valeurs pour cette metrique specifique
        # On ignore les cles non numeriques generees parfois par le Trainer (ex: eval_loss) si on veut,
        # mais on peut tout moyenner.
        values = [run_data[key] for run_data in runs.values() if isinstance(run_data[key], (int, float))]

        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)

            aggregated["metrics"][key] = {
                "mean": mean_val,
                "std": std_val
            }
            print(f"{key} : {mean_val:.4f} +- {std_val:.4f}")

    with open(final_path, "w") as f:
        json.dump(aggregated, f, indent=4)

    print(f"Resultats finaux calcules et sauvegardes dans {final_path}")


def parse_args():
    """Parse les arguments CLI du script."""
    parser = argparse.ArgumentParser(description="Exercice 1 - T5 text-to-text sur LEDGAR")
    parser.add_argument("--debug", action="store_true", help="Mode debug: 1 seed et 100 exemples max par split.")
    return parser.parse_args()


def _build_training_arguments(**kwargs):
    """Construit Seq2SeqTrainingArguments avec compatibilite inter-versions."""
    # Compatibilite descendante entre versions Transformers.
    signature = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in signature and "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return Seq2SeqTrainingArguments(**kwargs)


def _normalize_text(text):
    """Normalise un texte pour des comparaisons robustes."""
    # Normalisation robuste pour comparer labels generes et labels de reference.
    return " ".join(text.strip().lower().split())


def _build_label_lookup(label_names):
    """Construit un dictionnaire texte-normalise -> id de label."""
    return {_normalize_text(name): idx for idx, name in enumerate(label_names)}


def _map_text_to_label_id(text, label_lookup):
    """Mappe un texte vers un id de label (exact, fuzzy puis inclusion)."""
    # Mapping tolerant: exact match, puis approx. (difflib), puis inclusion.
    normalized = _normalize_text(text)
    if normalized in label_lookup:
        return label_lookup[normalized]

    close = get_close_matches(normalized, list(label_lookup.keys()), n=1, cutoff=0.6)
    if close:
        return label_lookup[close[0]]

    for label_text, label_id in label_lookup.items():
        if label_text in normalized or normalized in label_text:
            return label_id

    return 0


def _sanitize_token_ids(token_ids, tokenizer):
    """Nettoie les ids pour eviter des erreurs de decoding."""
    # Nettoie les sorties de generation pour garantir un decode stable.
    arr = np.asarray(token_ids)
    # Cas logits -> ids
    if arr.ndim == 3:
        arr = np.argmax(arr, axis=-1)
    arr = arr.astype(np.int64, copy=False)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    arr = np.where(arr < 0, pad_id, arr)
    # Evite les ids hors plage qui peuvent provoquer OverflowError dans le tokenizer fast
    vocab_max = int(getattr(tokenizer, "vocab_size", 0) or 0)
    if vocab_max > 0:
        arr = np.where(arr >= vocab_max, pad_id, arr)
    return arr


def build_compute_metrics(tokenizer, label_names):
    """Construit la fonction de metriques pour la sortie text-to-text."""
    label_lookup = _build_label_lookup(label_names)
    num_labels = len(label_names)

    def _compute_metrics(eval_pred):
        """Calcule accuracy/F1/AUROC a partir des textes generes."""
        # Pour Seq2Seq, `predictions` peut etre un tuple; on garde les ids utiles.
        predictions, labels = eval_pred
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        pred_ids = _sanitize_token_ids(predictions, tokenizer)
        label_ids = np.asarray(labels, dtype=np.int64)
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        label_ids = np.where(label_ids != -100, label_ids, pad_id)
        label_ids = _sanitize_token_ids(label_ids, tokenizer)

        decoded_preds = tokenizer.batch_decode(pred_ids.tolist(), skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(label_ids.tolist(), skip_special_tokens=True)

        # Convertit le texte genere en classes discretes pour reutiliser les memes metriques.
        pred_ids = np.array([_map_text_to_label_id(pred, label_lookup) for pred in decoded_preds], dtype=np.int64)
        true_ids = np.array([_map_text_to_label_id(label, label_lookup) for label in decoded_labels], dtype=np.int64)

        acc = accuracy_score_np(true_ids, pred_ids)
        f1 = f1_macro_np(true_ids, pred_ids)

        probs = np.full((len(pred_ids), num_labels), 1e-9, dtype=np.float32)
        # Approximation one-hot des probabilites a partir de la classe predite.
        probs[np.arange(len(pred_ids)), pred_ids] = 1.0 - (num_labels - 1) * 1e-9
        try:
            auroc = roc_auc_ovr_np(true_ids, probs)
        except ValueError:
            auroc = float("nan")

        return {"accuracy": acc, "f1_macro": f1, "auroc": auroc}

    return _compute_metrics


def load_ledgar():
    """Charge LEDGAR et retourne dataset + metadonnees de labels."""
    # Chargement centralise pour garder la meme source de donnees entre modeles.
    dataset = load_dataset("lex_glue", "ledgar")
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    return dataset, label_names, num_labels


def tokenize_splits(dataset, tokenizer, label_names, debug_mode):
    """Tokenise les splits en format prompt/cible pour T5."""
    train_dataset = dataset["train"]
    eval_dataset = dataset["validation"]
    test_dataset = dataset["test"]

    if debug_mode:
        # En debug: sous-ensemble petit et deterministe pour iterer rapidement.
        train_dataset = train_dataset.select(range(min(DEBUG_SAMPLES, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(DEBUG_SAMPLES, len(eval_dataset))))
        test_dataset = test_dataset.select(range(min(DEBUG_SAMPLES, len(test_dataset))))

    def preprocess(batch):
        """Prepare un batch prompt/cible en format seq2seq."""
        # Format text-to-text: prompt d'entree + label textuel comme cible.
        prompts = [f"Classify legal clause: {text}" for text in batch["text"]]
        targets = [label_names[label_id] for label_id in batch["label"]]

        model_inputs = tokenizer(
            prompts,
            truncation=True,
            max_length=MAX_INPUT_LENGTH,
        )
        labels = tokenizer(
            text_target=targets,
            truncation=True,
            max_length=MAX_TARGET_LENGTH,
        )
        model_inputs["labels"] = labels["input_ids"]
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


def build_model():
    """Instancie le modele T5 seq2seq pre-entraine."""
    # Full fine-tuning seq2seq standard pour T5.
    return AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)


def build_trainer(output_dir, seed, model, train_dataset, eval_dataset, tokenizer, label_names, debug_mode):
    """Construit le Seq2SeqTrainer configure pour T5."""
    # fp16 active seulement sur GPU pour accelerer l'entrainement sans casser CPU.
    fp16_enabled = torch.cuda.is_available()
    logging_steps = 10 if debug_mode else LOGGING_STEPS
    num_epochs = 1 if debug_mode else NUM_TRAIN_EPOCHS

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
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=fp16_enabled,
        dataloader_num_workers=0,
        report_to="none",
        # On desactive la generation pendant le train pour eviter un overhead massif.
        predict_with_generate=False,
        generation_max_length=GENERATED_LABEL_MAX_LENGTH,
        generation_num_beams=1,
        seed=seed,
        data_seed=seed,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        # Padding multiple de 8: meilleur alignement memoire avec kernels GPU/fp16.
        pad_to_multiple_of=8 if fp16_enabled else None,
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        # Pas de compute_metrics pendant le train: acceleration importante.
        "compute_metrics": None,
        "callbacks": [EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    }
    trainer_signature = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_signature:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Seq2SeqTrainer(**trainer_kwargs)
    return trainer


def evaluate_on_test(trainer, test_dataset, tokenizer, label_names):
    """Evalue le modele sur test avec generation activee."""
    # La generation est couteuse: on l'active uniquement au moment du test final.
    # On active la generation uniquement pour l'evaluation finale.
    previous_flag = trainer.args.predict_with_generate
    trainer.args.predict_with_generate = True
    try:
        predictions = trainer.predict(
            test_dataset,
            max_length=GENERATED_LABEL_MAX_LENGTH,
            num_beams=1,
        )
    finally:
        trainer.args.predict_with_generate = previous_flag

    compute_metrics = build_compute_metrics(tokenizer, label_names)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    return metrics


def main():
    """Execute la boucle complete d'entrainement/evaluation multi-seeds."""
    args = parse_args()
    debug_mode = DEBUG_MODE or args.debug
    # En debug: une seule seed pour accelerer les iterations.
    seeds = [42] if debug_mode else SEEDS

    dataset, label_names, _ = load_ledgar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    train_dataset, eval_dataset, test_dataset = tokenize_splits(dataset, tokenizer, label_names, debug_mode)

    for run, seed in enumerate(seeds):
        print(f"--- Debut du Run {run + 1}/{len(seeds)} avec la seed {seed} ---")
        set_seed(seed)
        output_dir = f"./checkpoints/{MODEL_NAME_SHORT}_run_{run}"
        os.makedirs(output_dir, exist_ok=True)

        run_completed_file = os.path.join(output_dir, "run_completed.txt")
        # Marqueur simple pour eviter de relancer un run deja fini.
        if os.path.exists(run_completed_file):
            print(f"Run {run + 1} deja termine, passage au suivant.")
            continue

        model = build_model()
        trainer = build_trainer(
            output_dir=output_dir,
            seed=seed,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            label_names=label_names,
            debug_mode=debug_mode,
        )

        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        # Reprise automatique possible hors mode debug.
        if last_checkpoint is not None and not debug_mode:
            print(f"Reprise a partir du checkpoint : {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        print(f"Evaluation finale du run {run} sur le jeu de test...")
        clean_metrics = evaluate_on_test(
            trainer=trainer,
            test_dataset=test_dataset,
            tokenizer=tokenizer,
            label_names=label_names,
        )

        save_run_metrics(
            metrics=clean_metrics,
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
