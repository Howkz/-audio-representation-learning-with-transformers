from __future__ import annotations

from typing import Iterable, List

import torch

from src.data.text import CharCTCTokenizer, normalize_transcript


def greedy_decode_batch(logits: torch.Tensor, tokenizer: CharCTCTokenizer) -> List[str]:
    # logits: [B, T, V]
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    return [normalize_transcript(tokenizer.decode(seq)) for seq in ids]


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
