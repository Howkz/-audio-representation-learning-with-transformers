from __future__ import annotations

from typing import Any, Dict, List

import torch

from .text import CharCTCTokenizer


def pad_collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    max_len = max(item["length"] for item in batch)
    n_mels = batch[0]["x_logmel"].shape[1]
    max_waveform_len = max(int(item.get("waveform_length", 0)) for item in batch)
    bsz = len(batch)

    x = torch.zeros((bsz, max_len, n_mels), dtype=torch.float32)
    lengths = torch.zeros((bsz,), dtype=torch.long)
    waveforms = torch.zeros((bsz, max_waveform_len), dtype=torch.float32)
    waveform_lengths = torch.zeros((bsz,), dtype=torch.long)

    transcripts = []
    for i, item in enumerate(batch):
        t = item["x_logmel"].shape[0]
        x[i, :t] = item["x_logmel"]
        lengths[i] = t
        waveform = item.get("waveform")
        if waveform is not None:
            waveform_len = int(item.get("waveform_length", waveform.shape[0]))
            waveforms[i, :waveform_len] = waveform[:waveform_len]
            waveform_lengths[i] = waveform_len
        transcripts.append(item.get("transcript", ""))

    return {
        "x_logmel": x,
        "lengths": lengths,
        "waveforms": waveforms,
        "waveform_lengths": waveform_lengths,
        "transcripts": transcripts,
    }


def ctc_collate(batch: List[Dict[str, Any]], tokenizer: CharCTCTokenizer) -> Dict[str, Any]:
    padded = pad_collate(batch)
    labels = []
    label_lengths = []
    for transcript in padded["transcripts"]:
        encoded = tokenizer.encode(transcript)
        labels.extend(encoded)
        label_lengths.append(len(encoded))

    padded["targets"] = torch.tensor(labels, dtype=torch.long)
    padded["target_lengths"] = torch.tensor(label_lengths, dtype=torch.long)
    return padded

