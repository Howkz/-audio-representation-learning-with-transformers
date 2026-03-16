import argparse
import inspect
import json
import os
import random

# Force l'usage exclusif de PyTorch pour eviter les imports TensorFlow inutiles.
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

MODEL_NAME = "gpt2"
MODEL_NAME_SHORT = "gpt2"
ARCHITECTURE = "Decodeur uniquement"
ADAPTATION = "LoRA + classification head"
TASK_NAME = "ex1"

MAX_LENGTH = 256
DEBUG_MODE = False
DEBUG_SAMPLES = 100
SEEDS = [42, 123, 456]

PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-4
NUM_TRAIN_EPOCHS = 2.0
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


def compute_metrics(eval_pred):
    """Calcule les metriques de classification depuis la sortie du Trainer."""
    # `Trainer` renvoie (logits, labels) pour les modeles de classification.
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Accuracy
    acc = accuracy_score_np(labels, predictions)
    # F1 Macro
    f1 = f1_macro_np(labels, predictions)

    # AUROC necessite les probabilites, on applique un softmax sur les logits
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1).numpy()
    try:
        # multi_class="ovr" est obligatoire pour plus de 2 classes
        auroc = roc_auc_ovr_np(labels, probs)
    except ValueError:
        # En cas de classes absentes dans le batch de test
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
    parser = argparse.ArgumentParser(description="Exercice 1 - GPT-2 LoRA classification sur LEDGAR")
    parser.add_argument("--debug", action="store_true", help="Mode debug: 1 seed et 100 exemples max par split.")
    return parser.parse_args()


def _build_training_arguments(**kwargs):
    """Construit TrainingArguments avec compatibilite inter-versions."""
    # Compatibilite descendante entre versions Transformers.
    signature = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in kwargs and "evaluation_strategy" not in signature and "eval_strategy" in signature:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def load_ledgar():
    """Charge LEDGAR et retourne dataset + metadonnees de labels."""
    # Chargement centralise pour garder la meme source de donnees entre modeles.
    dataset = load_dataset("lex_glue", "ledgar")
    label_names = dataset["train"].features["label"].names
    num_labels = len(label_names)
    return dataset, label_names, num_labels


def tokenize_splits(dataset, tokenizer, debug_mode):
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
        # Format classification standard: texte -> input ids + label entier.
        model_inputs = tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
        )
        model_inputs["labels"] = batch["label"]
        return model_inputs

    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(preprocess, batched=True, remove_columns=eval_dataset.column_names)
    test_dataset = test_dataset.map(preprocess, batched=True, remove_columns=test_dataset.column_names)

    return train_dataset, eval_dataset, test_dataset


def build_model(num_labels, label_names, tokenizer):
    """Construit GPT-2 de classification puis applique LoRA."""
    # Mapping explicite pour des sorties de prediction lisibles.
    id2label = {idx: name for idx, name in enumerate(label_names)}
    label2id = {name: idx for idx, name in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # En mode train, desactiver le cache reduit la memoire et evite certains conflits.
    model.config.use_cache = False

    peft_config = LoraConfig(
        # GPT-2 utilise des projections Conv1D; ces modules couvrent attention + MLP.
        task_type=TaskType.SEQ_CLS,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["c_attn", "c_proj", "c_fc"],
        modules_to_save=["score"],
        bias="none",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model


def build_trainer(output_dir, seed, model, train_dataset, eval_dataset, tokenizer, debug_mode):
    """Construit le Trainer configure pour GPT-2 LoRA."""
    # fp16 active seulement sur GPU pour accelerer l'entrainement sans casser CPU.
    fp16_enabled = torch.cuda.is_available()
    eval_steps = 50 if debug_mode else EVAL_STEPS
    save_steps = 50 if debug_mode else SAVE_STEPS
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
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_strategy="steps",
        logging_steps=logging_steps,
        fp16=fp16_enabled,
        dataloader_num_workers=0,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        # Padding multiple de 8: meilleur alignement memoire avec kernels GPU/fp16.
        pad_to_multiple_of=8 if fp16_enabled else None,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    return trainer


def main():
    """Execute la boucle complete d'entrainement/evaluation multi-seeds."""
    args = parse_args()
    debug_mode = DEBUG_MODE or args.debug
    # En debug: une seule seed pour accelerer les iterations.
    seeds = [42] if debug_mode else SEEDS

    dataset, label_names, num_labels = load_ledgar()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    if tokenizer.pad_token is None:
        # GPT-2 n'a pas de pad token natif: on reutilise EOS pour le batching.
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset, eval_dataset, test_dataset = tokenize_splits(dataset, tokenizer, debug_mode)

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

        model = build_model(num_labels, label_names, tokenizer)
        trainer = build_trainer(
            output_dir=output_dir,
            seed=seed,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            debug_mode=debug_mode,
        )

        last_checkpoint = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
        # Reprise automatique possible hors mode debug.
        if last_checkpoint is not None and not debug_mode:
            print(f"Reprise a partir du checkpoint : {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        print(f"Evaluation du run {run} sur le jeu de test...")
        raw_metrics = trainer.evaluate(eval_dataset=test_dataset)
        clean_metrics = {
            "accuracy": raw_metrics.get("eval_accuracy", 0.0),
            "f1_macro": raw_metrics.get("eval_f1_macro", 0.0),
            "auroc": raw_metrics.get("eval_auroc", 0.0),
        }

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
