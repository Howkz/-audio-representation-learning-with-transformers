from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
try:
    from torchmetrics.text import WordErrorRate
except Exception:
    try:
        from torchmetrics.text.wer import WordErrorRate  # type: ignore
    except Exception:
        WordErrorRate = None  # type: ignore[assignment]

from src.data.text import CharCTCTokenizer, normalize_transcript


def _collapsed_char_sequence(text: str) -> str:
    return "".join(ch for ch in text if ch != " ")


def _adjacent_repeat_ratio(text: str) -> float:
    chars = _collapsed_char_sequence(text)
    if len(chars) <= 1:
        return 0.0
    repeated_pairs = sum(1 for left, right in zip(chars, chars[1:]) if left == right)
    return float(repeated_pairs / max(1, len(chars) - 1))


def _dominant_char_ratio(text: str) -> float:
    chars = _collapsed_char_sequence(text)
    if not chars:
        return 0.0
    counts = Counter(chars)
    return float(max(counts.values()) / max(1, len(chars)))


def greedy_decode_batch(
    logits: torch.Tensor,
    tokenizer: CharCTCTokenizer,
    lengths: Optional[torch.Tensor] = None,
) -> List[str]:
    # logits: [B, T, V]
    argmax_ids = logits.argmax(dim=-1).detach().cpu()
    if lengths is None:
        lengths_cpu = torch.full(
            (argmax_ids.shape[0],),
            fill_value=argmax_ids.shape[1],
            dtype=torch.long,
        )
    else:
        lengths_cpu = lengths.detach().cpu().to(torch.long)

    predictions: List[str] = []
    max_len = int(argmax_ids.shape[1])
    for seq_ids, seq_len in zip(argmax_ids, lengths_cpu):
        trunc_len = int(max(0, min(max_len, int(seq_len.item()))))
        predictions.append(normalize_transcript(tokenizer.decode(seq_ids[:trunc_len].tolist())))
    return predictions


def collect_ctc_batch_diagnostics(
    logits: torch.Tensor,
    out_lengths: torch.Tensor,
    target_lengths: Optional[torch.Tensor],
    references: Sequence[str],
    tokenizer: CharCTCTokenizer,
    max_examples: int = 0,
) -> Tuple[List[str], Dict[str, float], List[Dict[str, Any]]]:
    argmax_ids = logits.argmax(dim=-1).detach().cpu()
    out_lengths_cpu = out_lengths.detach().cpu().to(torch.long)
    target_lengths_cpu = None
    if target_lengths is not None:
        target_lengths_cpu = target_lengths.detach().cpu().to(torch.long)

    predictions: List[str] = []
    examples: List[Dict[str, Any]] = []
    totals: Dict[str, float] = {
        "num_samples": float(argmax_ids.shape[0]),
        "decoded_steps": 0.0,
        "blank_steps": 0.0,
        "empty_predictions": 0.0,
        "exact_matches": 0.0,
        "invalid_length_samples": 0.0,
        "sum_pred_chars": 0.0,
        "sum_ref_chars": 0.0,
        "sum_out_lengths": 0.0,
        "sum_target_lengths": 0.0,
        "sum_length_margin": 0.0,
        "sum_adjacent_repeat_ratio": 0.0,
        "sum_dominant_char_ratio": 0.0,
    }

    max_len = int(argmax_ids.shape[1])
    for idx, seq_ids in enumerate(argmax_ids):
        out_len = int(max(0, min(max_len, int(out_lengths_cpu[idx].item()))))
        decoded_ids = seq_ids[:out_len]
        pred = normalize_transcript(tokenizer.decode(decoded_ids.tolist()))
        ref = normalize_transcript(references[idx])
        predictions.append(pred)

        target_len = -1
        if target_lengths_cpu is not None:
            target_len = int(target_lengths_cpu[idx].item())
            totals["sum_target_lengths"] += float(target_len)
            totals["sum_length_margin"] += float(out_len - target_len)
            if out_len < target_len:
                totals["invalid_length_samples"] += 1.0

        totals["decoded_steps"] += float(out_len)
        totals["blank_steps"] += float((decoded_ids == tokenizer.blank_id).sum().item())
        totals["sum_out_lengths"] += float(out_len)
        totals["sum_pred_chars"] += float(len(pred.replace(" ", "")))
        totals["sum_ref_chars"] += float(len(ref.replace(" ", "")))
        totals["sum_adjacent_repeat_ratio"] += _adjacent_repeat_ratio(pred)
        totals["sum_dominant_char_ratio"] += _dominant_char_ratio(pred)
        if not pred:
            totals["empty_predictions"] += 1.0
        if pred == ref:
            totals["exact_matches"] += 1.0

        if len(examples) < max_examples:
            examples.append(
                {
                    "index_in_batch": int(idx),
                    "reference": ref,
                    "prediction": pred,
                    "out_length": int(out_len),
                    "target_length": int(target_len),
                    "decoded_token_ids": decoded_ids.tolist(),
                }
            )

    return predictions, totals, examples


def finalize_ctc_diagnostics(totals: Dict[str, float]) -> Dict[str, float]:
    num_samples = max(1.0, float(totals.get("num_samples", 0.0)))
    decoded_steps = max(0.0, float(totals.get("decoded_steps", 0.0)))
    has_target_lengths = float(totals.get("sum_target_lengths", 0.0)) > 0.0

    diagnostics = {
        "blank_ratio": float(totals.get("blank_steps", 0.0) / max(1.0, decoded_steps)),
        "empty_pred_ratio": float(totals.get("empty_predictions", 0.0) / num_samples),
        "nonempty_pred_ratio": float(1.0 - (totals.get("empty_predictions", 0.0) / num_samples)),
        "exact_match_ratio": float(totals.get("exact_matches", 0.0) / num_samples),
        "avg_pred_chars": float(totals.get("sum_pred_chars", 0.0) / num_samples),
        "avg_ref_chars": float(totals.get("sum_ref_chars", 0.0) / num_samples),
        "avg_out_length": float(totals.get("sum_out_lengths", 0.0) / num_samples),
    }
    diagnostics["pred_to_ref_char_ratio"] = float(
        diagnostics["avg_pred_chars"] / max(1e-6, diagnostics["avg_ref_chars"])
    )
    diagnostics["short_pred_ratio"] = float(max(0.0, 1.0 - diagnostics["pred_to_ref_char_ratio"]))
    diagnostics["long_pred_ratio"] = float(max(0.0, diagnostics["pred_to_ref_char_ratio"] - 1.0))
    diagnostics["length_deviation_ratio"] = float(abs(1.0 - diagnostics["pred_to_ref_char_ratio"]))
    diagnostics["adjacent_repeat_ratio"] = float(totals.get("sum_adjacent_repeat_ratio", 0.0) / num_samples)
    diagnostics["dominant_char_ratio"] = float(totals.get("sum_dominant_char_ratio", 0.0) / num_samples)
    if has_target_lengths:
        diagnostics.update(
            {
                "avg_target_length": float(totals.get("sum_target_lengths", 0.0) / num_samples),
                "avg_length_margin": float(totals.get("sum_length_margin", 0.0) / num_samples),
                "invalid_length_ratio": float(totals.get("invalid_length_samples", 0.0) / num_samples),
            }
        )
    else:
        diagnostics.update(
            {
                "avg_target_length": 0.0,
                "avg_length_margin": 0.0,
                "invalid_length_ratio": 0.0,
            }
        )
    return diagnostics


def _levenshtein_distance(ref_words: List[str], hyp_words: List[str]) -> int:
    n = len(ref_words)
    m = len(hyp_words)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,      # deletion
                curr[j - 1] + 1,  # insertion
                prev[j - 1] + cost,  # substitution
            )
        prev = curr
    return prev[m]


def compute_wer(predictions: Iterable[str], references: Iterable[str]) -> float:
    preds = [normalize_transcript(x) for x in predictions]
    refs = [normalize_transcript(x) for x in references]
    if len(preds) != len(refs):
        raise ValueError("Predictions and references must have the same length.")

    if WordErrorRate is not None:
        try:
            metric = WordErrorRate()
            return float(metric(preds, refs).item())
        except Exception:
            pass

    total_ref_words = 0
    total_errors = 0
    for pred, ref in zip(preds, refs):
        ref_words = ref.split() if ref else []
        pred_words = pred.split() if pred else []
        total_ref_words += len(ref_words)
        total_errors += _levenshtein_distance(ref_words, pred_words)
    if total_ref_words == 0:
        return 0.0
    return float(total_errors / total_ref_words)


def compute_char_accuracy(predictions: Iterable[str], references: Iterable[str]) -> float:
    preds = [normalize_transcript(x) for x in predictions]
    refs = [normalize_transcript(x) for x in references]
    if len(preds) != len(refs):
        raise ValueError("Predictions and references must have the same length.")

    total_ref_chars = 0
    total_errors = 0
    for pred, ref in zip(preds, refs):
        ref_chars = list(ref)
        pred_chars = list(pred)
        total_ref_chars += max(1, len(ref_chars))
        total_errors += _levenshtein_distance(ref_chars, pred_chars)
    accuracy = 1.0 - (float(total_errors) / float(max(1, total_ref_chars)))
    return float(max(0.0, accuracy))
