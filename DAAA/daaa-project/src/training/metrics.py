from __future__ import annotations

from typing import Iterable, List

import torch
from jiwer import wer

from src.data.text import CharCTCTokenizer, normalize_transcript


def greedy_decode_batch(logits: torch.Tensor, tokenizer: CharCTCTokenizer) -> List[str]:
    # logits: [B, T, V]
    ids = logits.argmax(dim=-1).detach().cpu().tolist()
    return [normalize_transcript(tokenizer.decode(seq)) for seq in ids]


def compute_wer(predictions: Iterable[str], references: Iterable[str]) -> float:
    preds = [normalize_transcript(x) for x in predictions]
    refs = [normalize_transcript(x) for x in references]
    return float(wer(refs, preds))

