from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
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


def _unique_char_ratio(text: str) -> float:
    chars = _collapsed_char_sequence(text)
    if not chars:
        return 0.0
    return float(len(set(chars)) / max(1, len(chars)))


def _ctc_collapse_token_ids(token_ids: Sequence[int], blank_id: int) -> List[int]:
    collapsed: List[int] = []
    previous: Optional[int] = None
    for token_id in token_ids:
        token_id = int(token_id)
        if token_id == previous:
            continue
        previous = token_id
        if token_id == int(blank_id):
            continue
        collapsed.append(token_id)
    return collapsed


def _adjacent_repeat_ratio_from_ids(token_ids: Sequence[int]) -> float:
    if len(token_ids) <= 1:
        return 0.0
    repeated_pairs = sum(1 for left, right in zip(token_ids, token_ids[1:]) if int(left) == int(right))
    return float(repeated_pairs / max(1, len(token_ids) - 1))


def _token_switch_ratio_from_ids(token_ids: Sequence[int]) -> float:
    if len(token_ids) <= 1:
        return 0.0
    switches = sum(1 for left, right in zip(token_ids, token_ids[1:]) if int(left) != int(right))
    return float(switches / max(1, len(token_ids) - 1))


def _dominant_ratio_from_ids(token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    counts = Counter(int(token_id) for token_id in token_ids)
    return float(max(counts.values()) / max(1, len(token_ids)))


def _unique_ratio_from_ids(token_ids: Sequence[int]) -> float:
    if not token_ids:
        return 0.0
    return float(len({int(token_id) for token_id in token_ids}) / max(1, len(token_ids)))


def _accumulate_stat_block(block: Dict[str, float], name: str, values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    values = values.detach().to(dtype=torch.float32)
    block[f"{name}_sum"] = float(block.get(f"{name}_sum", 0.0) + float(values.sum().item()))
    block[f"{name}_sumsq"] = float(block.get(f"{name}_sumsq", 0.0) + float((values * values).sum().item()))
    block[f"{name}_count"] = float(block.get(f"{name}_count", 0.0) + float(values.numel()))


def _summarize_stat_block(block: Dict[str, float], name: str) -> Tuple[float, float]:
    count = float(block.get(f"{name}_count", 0.0))
    if count <= 0.0:
        return 0.0, 0.0
    total = float(block.get(f"{name}_sum", 0.0))
    total_sq = float(block.get(f"{name}_sumsq", 0.0))
    mean = total / count
    variance = max(0.0, (total_sq / count) - (mean * mean))
    return float(mean), float(variance ** 0.5)


def _update_histogram(target: Dict[str, float], token_ids: Sequence[int]) -> None:
    for token_id in token_ids:
        key = str(int(token_id))
        target[key] = float(target.get(key, 0.0) + 1.0)


def _histogram_to_rows(
    histogram: Dict[str, float],
    tokenizer: Optional[CharCTCTokenizer],
) -> List[Dict[str, Any]]:
    total = max(1.0, float(sum(float(v) for v in histogram.values())))
    rows: List[Dict[str, Any]] = []
    for key, count in sorted(histogram.items(), key=lambda item: (-float(item[1]), int(item[0]))):
        token_id = int(key)
        token = str(token_id)
        if tokenizer is not None and 0 <= token_id < tokenizer.vocab_size:
            token = tokenizer.id_to_char[token_id]
        rows.append(
            {
                "token_id": int(token_id),
                "token": token,
                "count": float(count),
                "ratio": float(float(count) / total),
            }
        )
    return rows


def greedy_decode_batch(
    logits: torch.Tensor,
    tokenizer: CharCTCTokenizer,
    lengths: Optional[torch.Tensor] = None,
) -> List[str]:
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
    max_frames: int = 96,
    topk_tokens: int = 5,
    sample_ids: Optional[Sequence[str]] = None,
    source_datasets: Optional[Sequence[str]] = None,
    source_splits: Optional[Sequence[str]] = None,
    waveform_lengths: Optional[torch.Tensor] = None,
    sample_rate: Optional[int] = None,
    student_features: Optional[torch.Tensor] = None,
    teacher_forensics: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], Dict[str, Any], List[Dict[str, Any]]]:
    blank_id = int(tokenizer.blank_id)
    probs = torch.softmax(logits.float(), dim=-1)
    topk_limit = max(1, min(int(topk_tokens), int(probs.shape[-1])))
    topk_values, topk_indices = torch.topk(probs, k=topk_limit, dim=-1)
    argmax_ids = topk_indices[..., 0]
    top1_probs = topk_values[..., 0]
    if topk_limit >= 2:
        top1_margin = topk_values[..., 0] - topk_values[..., 1]
    else:
        top1_margin = topk_values[..., 0]
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    blank_probs = probs[..., blank_id]
    nonblank_probs = 1.0 - blank_probs

    out_lengths_device = out_lengths.to(logits.device, dtype=torch.long)
    valid_mask = (
        torch.arange(logits.shape[1], device=logits.device, dtype=torch.long).unsqueeze(0)
        < out_lengths_device.unsqueeze(1)
    )

    teacher_distribution = None
    teacher_distribution_valid_mask = None
    teacher_raw_lengths = None
    teacher_features = None
    teacher_feature_valid_mask = None
    if teacher_forensics is not None:
        teacher_distribution = teacher_forensics.get("distribution")
        teacher_distribution_valid_mask = teacher_forensics.get("distribution_valid_mask")
        teacher_raw_lengths = teacher_forensics.get("raw_lengths")
        teacher_features = teacher_forensics.get("features")
        teacher_feature_valid_mask = teacher_forensics.get("feature_valid_mask")

    predictions: List[str] = []
    examples: List[Dict[str, Any]] = []
    totals: Dict[str, Any] = {
        "num_samples": float(logits.shape[0]),
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
        "sum_unique_char_ratio": 0.0,
        "sum_collapsed_repeat_ratio": 0.0,
        "sum_collapsed_dominant_char_ratio": 0.0,
        "sum_collapsed_unique_char_ratio": 0.0,
        "sum_collapse_compression_ratio": 0.0,
        "sum_raw_blank_argmax_ratio": 0.0,
        "sum_raw_nonblank_argmax_ratio": 0.0,
        "sum_raw_unique_token_ratio": 0.0,
        "sum_raw_dominant_token_ratio": 0.0,
        "sum_raw_adjacent_repeat_ratio": 0.0,
        "sum_raw_token_switch_ratio": 0.0,
        "student_frame_stats": {},
        "alignment_stats": {},
        "teacher_distribution_stats": {},
        "teacher_feature_stats": {},
        "raw_top1_token_histogram": {},
        "collapsed_token_histogram": {},
        "sample_id_counts": {},
    }

    _accumulate_stat_block(totals["student_frame_stats"], "blank_prob", blank_probs[valid_mask])
    _accumulate_stat_block(totals["student_frame_stats"], "nonblank_prob", nonblank_probs[valid_mask])
    _accumulate_stat_block(totals["student_frame_stats"], "entropy", entropy[valid_mask])
    _accumulate_stat_block(totals["student_frame_stats"], "top1_confidence", top1_probs[valid_mask])
    _accumulate_stat_block(totals["student_frame_stats"], "top1_margin", top1_margin[valid_mask])

    out_lengths_cpu = out_lengths.detach().cpu().to(torch.long)
    target_lengths_cpu = target_lengths.detach().cpu().to(torch.long) if target_lengths is not None else None
    waveform_lengths_cpu = waveform_lengths.detach().cpu().to(torch.long) if waveform_lengths is not None else None
    argmax_ids_cpu = argmax_ids.detach().cpu().to(torch.long)
    top1_probs_cpu = top1_probs.detach().cpu().to(torch.float32)
    top1_margin_cpu = top1_margin.detach().cpu().to(torch.float32)
    blank_probs_cpu = blank_probs.detach().cpu().to(torch.float32)
    entropy_cpu = entropy.detach().cpu().to(torch.float32)
    topk_indices_cpu = topk_indices.detach().cpu().to(torch.long)
    topk_values_cpu = topk_values.detach().cpu().to(torch.float32)

    teacher_dist_cpu = teacher_distribution.detach().cpu().to(torch.float32) if isinstance(teacher_distribution, torch.Tensor) else None
    teacher_dist_mask_cpu = (
        teacher_distribution_valid_mask.detach().cpu().to(torch.bool)
        if isinstance(teacher_distribution_valid_mask, torch.Tensor)
        else None
    )
    teacher_features_cpu = teacher_features.detach().cpu().to(torch.float32) if isinstance(teacher_features, torch.Tensor) else None
    teacher_feature_mask_cpu = (
        teacher_feature_valid_mask.detach().cpu().to(torch.bool)
        if isinstance(teacher_feature_valid_mask, torch.Tensor)
        else None
    )
    student_features_cpu = student_features.detach().cpu().to(torch.float32) if isinstance(student_features, torch.Tensor) else None
    teacher_raw_lengths_cpu = teacher_raw_lengths.detach().cpu().to(torch.long) if isinstance(teacher_raw_lengths, torch.Tensor) else None

    teacher_topk_indices_cpu = None
    teacher_top1_probs_cpu = None
    if teacher_dist_cpu is not None and teacher_dist_mask_cpu is not None:
        teacher_topk_limit = max(1, min(topk_limit, int(teacher_dist_cpu.shape[-1])))
        teacher_topk_values_cpu, teacher_topk_indices_cpu = torch.topk(teacher_dist_cpu, k=teacher_topk_limit, dim=-1)
        teacher_top1_probs_cpu = teacher_topk_values_cpu[..., 0]
        if teacher_topk_limit >= 2:
            teacher_top1_margin_cpu = teacher_topk_values_cpu[..., 0] - teacher_topk_values_cpu[..., 1]
        else:
            teacher_top1_margin_cpu = teacher_topk_values_cpu[..., 0]
        teacher_entropy_cpu = -(teacher_dist_cpu * teacher_dist_cpu.clamp_min(1e-8).log()).sum(dim=-1)
        teacher_blank_probs_cpu = teacher_dist_cpu[..., blank_id]
        teacher_nonblank_probs_cpu = 1.0 - teacher_blank_probs_cpu

        _accumulate_stat_block(
            totals["teacher_distribution_stats"],
            "entropy",
            teacher_entropy_cpu[teacher_dist_mask_cpu],
        )
        _accumulate_stat_block(
            totals["teacher_distribution_stats"],
            "top1_confidence",
            teacher_top1_probs_cpu[teacher_dist_mask_cpu],
        )
        _accumulate_stat_block(
            totals["teacher_distribution_stats"],
            "top1_margin",
            teacher_top1_margin_cpu[teacher_dist_mask_cpu],
        )
        _accumulate_stat_block(
            totals["teacher_distribution_stats"],
            "blank_prob",
            teacher_blank_probs_cpu[teacher_dist_mask_cpu],
        )
        _accumulate_stat_block(
            totals["teacher_distribution_stats"],
            "nonblank_prob",
            teacher_nonblank_probs_cpu[teacher_dist_mask_cpu],
        )

        student_argmax_cpu = argmax_ids_cpu
        teacher_argmax_cpu = teacher_topk_indices_cpu[..., 0].to(torch.long)
        agreement = (student_argmax_cpu == teacher_argmax_cpu) & teacher_dist_mask_cpu
        totals["teacher_distribution_stats"]["argmax_agreement_sum"] = float(
            totals["teacher_distribution_stats"].get("argmax_agreement_sum", 0.0) + float(agreement.sum().item())
        )
        totals["teacher_distribution_stats"]["argmax_agreement_count"] = float(
            totals["teacher_distribution_stats"].get("argmax_agreement_count", 0.0)
            + float(teacher_dist_mask_cpu.sum().item())
        )
        totals["teacher_distribution_stats"]["topk_overlap_at1_sum"] = float(
            totals["teacher_distribution_stats"].get("topk_overlap_at1_sum", 0.0) + float(agreement.sum().item())
        )
        totals["teacher_distribution_stats"]["topk_overlap_at1_count"] = float(
            totals["teacher_distribution_stats"].get("topk_overlap_at1_count", 0.0)
            + float(teacher_dist_mask_cpu.sum().item())
        )
        overlap_sum = 0.0
        overlap_count = 0.0
        compare_k = max(1, min(3, topk_limit, teacher_topk_limit))
        for batch_idx in range(int(teacher_dist_mask_cpu.shape[0])):
            valid_positions = torch.nonzero(teacher_dist_mask_cpu[batch_idx], as_tuple=False).flatten().tolist()
            for position in valid_positions:
                student_set = set(int(x) for x in topk_indices_cpu[batch_idx, position, :compare_k].tolist())
                teacher_set = set(int(x) for x in teacher_topk_indices_cpu[batch_idx, position, :compare_k].tolist())
                overlap_sum += float(len(student_set & teacher_set) / max(1, compare_k))
                overlap_count += 1.0
        totals["teacher_distribution_stats"]["topk_overlap_at3_sum"] = float(
            totals["teacher_distribution_stats"].get("topk_overlap_at3_sum", 0.0) + overlap_sum
        )
        totals["teacher_distribution_stats"]["topk_overlap_at3_count"] = float(
            totals["teacher_distribution_stats"].get("topk_overlap_at3_count", 0.0) + overlap_count
        )

    cosine_cpu = None
    if (
        student_features_cpu is not None
        and teacher_features_cpu is not None
        and teacher_feature_mask_cpu is not None
        and student_features_cpu.shape[-1] == teacher_features_cpu.shape[-1]
    ):
        cosine_cpu = F.cosine_similarity(student_features_cpu, teacher_features_cpu, dim=-1)
        mse = F.mse_loss(student_features_cpu, teacher_features_cpu, reduction="none").mean(dim=-1)
        student_norm = student_features_cpu.norm(dim=-1)
        teacher_norm = teacher_features_cpu.norm(dim=-1)
        norm_ratio = student_norm / teacher_norm.clamp_min(1e-8)
        _accumulate_stat_block(totals["teacher_feature_stats"], "cosine", cosine_cpu[teacher_feature_mask_cpu])
        _accumulate_stat_block(totals["teacher_feature_stats"], "mse", mse[teacher_feature_mask_cpu])
        _accumulate_stat_block(totals["teacher_feature_stats"], "norm_ratio", norm_ratio[teacher_feature_mask_cpu])

    max_len = int(argmax_ids_cpu.shape[1])
    for idx, seq_ids in enumerate(argmax_ids_cpu):
        out_len = int(max(0, min(max_len, int(out_lengths_cpu[idx].item()))))
        raw_ids = seq_ids[:out_len].tolist()
        pred = normalize_transcript(tokenizer.decode(raw_ids))
        ref = normalize_transcript(references[idx])
        predictions.append(pred)

        target_len = -1
        if target_lengths_cpu is not None:
            target_len = int(target_lengths_cpu[idx].item())
            totals["sum_target_lengths"] += float(target_len)
            totals["sum_length_margin"] += float(out_len - target_len)
            if out_len < target_len:
                totals["invalid_length_samples"] += 1.0

        collapsed_token_ids = _ctc_collapse_token_ids(raw_ids, blank_id=blank_id)
        pred_chars = pred.replace(" ", "")
        ref_chars = ref.replace(" ", "")
        blank_steps = sum(1 for token_id in raw_ids if int(token_id) == blank_id)
        raw_nonblank_ids = [int(token_id) for token_id in raw_ids if int(token_id) != blank_id]

        totals["decoded_steps"] += float(out_len)
        totals["blank_steps"] += float(blank_steps)
        totals["sum_out_lengths"] += float(out_len)
        totals["sum_pred_chars"] += float(len(pred_chars))
        totals["sum_ref_chars"] += float(len(ref_chars))
        totals["sum_adjacent_repeat_ratio"] += _adjacent_repeat_ratio(pred)
        totals["sum_dominant_char_ratio"] += _dominant_char_ratio(pred)
        totals["sum_unique_char_ratio"] += _unique_char_ratio(pred)
        totals["sum_collapsed_repeat_ratio"] += _adjacent_repeat_ratio(pred)
        totals["sum_collapsed_dominant_char_ratio"] += _dominant_char_ratio(pred)
        totals["sum_collapsed_unique_char_ratio"] += _unique_char_ratio(pred)
        totals["sum_collapse_compression_ratio"] += float(out_len / max(1, len(collapsed_token_ids)))
        totals["sum_raw_blank_argmax_ratio"] += float(blank_steps / max(1, out_len))
        totals["sum_raw_nonblank_argmax_ratio"] += float(len(raw_nonblank_ids) / max(1, out_len))
        totals["sum_raw_unique_token_ratio"] += _unique_ratio_from_ids(raw_ids)
        totals["sum_raw_dominant_token_ratio"] += _dominant_ratio_from_ids(raw_ids)
        totals["sum_raw_adjacent_repeat_ratio"] += _adjacent_repeat_ratio_from_ids(raw_ids)
        totals["sum_raw_token_switch_ratio"] += _token_switch_ratio_from_ids(raw_ids)
        _update_histogram(totals["raw_top1_token_histogram"], raw_ids)
        _update_histogram(totals["collapsed_token_histogram"], collapsed_token_ids)
        if not pred:
            totals["empty_predictions"] += 1.0
        if pred == ref:
            totals["exact_matches"] += 1.0

        sample_id = sample_ids[idx] if sample_ids is not None and idx < len(sample_ids) else f"sample_{idx}"
        counts = totals["sample_id_counts"]
        counts[str(sample_id)] = float(counts.get(str(sample_id), 0.0) + 1.0)

        if teacher_raw_lengths_cpu is not None and idx < len(teacher_raw_lengths_cpu):
            teacher_length = int(max(0, int(teacher_raw_lengths_cpu[idx].item())))
            student_length = int(out_len)
            alignment_stats = totals["alignment_stats"]
            alignment_stats["count"] = float(alignment_stats.get("count", 0.0) + 1.0)
            alignment_stats["student_time_steps_sum"] = float(
                alignment_stats.get("student_time_steps_sum", 0.0) + float(student_length)
            )
            alignment_stats["teacher_time_steps_sum"] = float(
                alignment_stats.get("teacher_time_steps_sum", 0.0) + float(teacher_length)
            )
            alignment_stats["teacher_to_student_time_ratio_sum"] = float(
                alignment_stats.get("teacher_to_student_time_ratio_sum", 0.0)
                + float(teacher_length / max(1, student_length))
            )
            interpolation_ratio = float(student_length / max(1, teacher_length))
            alignment_stats["alignment_interpolation_ratio_sum"] = float(
                alignment_stats.get("alignment_interpolation_ratio_sum", 0.0) + interpolation_ratio
            )
            alignment_stats["teacher_student_alignment_stretch_sum"] = float(
                alignment_stats.get("teacher_student_alignment_stretch_sum", 0.0) + interpolation_ratio
            )
            if waveform_lengths_cpu is not None and idx < len(waveform_lengths_cpu) and sample_rate:
                duration_sec = float(waveform_lengths_cpu[idx].item()) / max(1.0, float(sample_rate))
                if duration_sec > 0.0:
                    alignment_stats["student_fps_sum"] = float(
                        alignment_stats.get("student_fps_sum", 0.0) + (float(student_length) / duration_sec)
                    )
                    alignment_stats["teacher_fps_sum"] = float(
                        alignment_stats.get("teacher_fps_sum", 0.0) + (float(teacher_length) / duration_sec)
                    )
            student_valid = max(1, student_length)
            if teacher_dist_mask_cpu is not None:
                aligned_valid = float(teacher_dist_mask_cpu[idx].sum().item())
                alignment_stats["teacher_student_valid_frame_overlap_sum"] = float(
                    alignment_stats.get("teacher_student_valid_frame_overlap_sum", 0.0)
                    + float(aligned_valid / student_valid)
                )
                alignment_stats["teacher_student_valid_frame_overlap_count"] = float(
                    alignment_stats.get("teacher_student_valid_frame_overlap_count", 0.0) + 1.0
                )
            elif teacher_feature_mask_cpu is not None:
                aligned_valid = float(teacher_feature_mask_cpu[idx].sum().item())
                alignment_stats["teacher_student_valid_frame_overlap_sum"] = float(
                    alignment_stats.get("teacher_student_valid_frame_overlap_sum", 0.0)
                    + float(aligned_valid / student_valid)
                )
                alignment_stats["teacher_student_valid_frame_overlap_count"] = float(
                    alignment_stats.get("teacher_student_valid_frame_overlap_count", 0.0) + 1.0
                )

        if len(examples) < max_examples:
            frame_limit = max(0, min(int(max_frames), out_len))
            raw_top1_token_ids = [int(token_id) for token_id in raw_ids[:frame_limit]]
            raw_top1_token_probs = top1_probs_cpu[idx, :frame_limit].tolist()
            raw_topk_token_ids = topk_indices_cpu[idx, :frame_limit, :topk_limit].tolist()
            raw_topk_token_probs = topk_values_cpu[idx, :frame_limit, :topk_limit].tolist()
            frame_blank_series = blank_probs_cpu[idx, :frame_limit].tolist()
            frame_entropy_series = entropy_cpu[idx, :frame_limit].tolist()
            frame_top1_margin_series = top1_margin_cpu[idx, :frame_limit].tolist()
            example: Dict[str, Any] = {
                "index_in_batch": int(idx),
                "sample_id": str(sample_id),
                "source_dataset": None if source_datasets is None or idx >= len(source_datasets) else str(source_datasets[idx]),
                "source_split": None if source_splits is None or idx >= len(source_splits) else str(source_splits[idx]),
                "reference": ref,
                "prediction": pred,
                "out_length": int(out_len),
                "target_length": int(target_len),
                "decoded_token_ids": [int(token_id) for token_id in raw_ids],
                "collapsed_token_ids": [int(token_id) for token_id in collapsed_token_ids],
                "raw_top1_token_ids": raw_top1_token_ids,
                "raw_top1_token_probs": [float(v) for v in raw_top1_token_probs],
                "raw_topk_token_ids": [[int(v) for v in row] for row in raw_topk_token_ids],
                "raw_topk_token_probs": [[float(v) for v in row] for row in raw_topk_token_probs],
                "frame_blank_probs": [float(v) for v in frame_blank_series],
                "frame_entropy": [float(v) for v in frame_entropy_series],
                "frame_top1_margin": [float(v) for v in frame_top1_margin_series],
            }
            if teacher_topk_indices_cpu is not None and teacher_top1_probs_cpu is not None and teacher_dist_mask_cpu is not None:
                teacher_frame_limit = max(0, min(int(max_frames), int(teacher_dist_mask_cpu[idx].sum().item())))
                example["teacher_top1_token_ids"] = [
                    int(v) for v in teacher_topk_indices_cpu[idx, :teacher_frame_limit, 0].tolist()
                ]
                example["teacher_top1_token_probs"] = [
                    float(v) for v in teacher_top1_probs_cpu[idx, :teacher_frame_limit].tolist()
                ]
                match_limit = min(frame_limit, teacher_frame_limit)
                example["teacher_student_argmax_match"] = [
                    int(raw_top1_token_ids[pos] == int(teacher_topk_indices_cpu[idx, pos, 0].item()))
                    for pos in range(match_limit)
                ]
            if cosine_cpu is not None and teacher_feature_mask_cpu is not None:
                feature_frame_limit = max(0, min(int(max_frames), int(teacher_feature_mask_cpu[idx].sum().item())))
                example["teacher_student_feature_cosine_by_frame"] = [
                    float(v) for v in cosine_cpu[idx, :feature_frame_limit].tolist()
                ]
            examples.append(example)

    return predictions, totals, examples


def finalize_ctc_diagnostics(
    totals: Dict[str, Any],
    tokenizer: Optional[CharCTCTokenizer] = None,
    expected_num_unique_samples: Optional[int] = None,
) -> Dict[str, Any]:
    num_samples = max(1.0, float(totals.get("num_samples", 0.0)))
    decoded_steps = max(0.0, float(totals.get("decoded_steps", 0.0)))
    has_target_lengths = float(totals.get("sum_target_lengths", 0.0)) > 0.0

    diagnostics: Dict[str, Any] = {
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
    diagnostics["unique_char_ratio"] = float(totals.get("sum_unique_char_ratio", 0.0) / num_samples)
    diagnostics["collapsed_repeat_ratio"] = float(totals.get("sum_collapsed_repeat_ratio", 0.0) / num_samples)
    diagnostics["collapsed_dominant_char_ratio"] = float(
        totals.get("sum_collapsed_dominant_char_ratio", 0.0) / num_samples
    )
    diagnostics["collapsed_unique_char_ratio"] = float(
        totals.get("sum_collapsed_unique_char_ratio", 0.0) / num_samples
    )
    diagnostics["collapse_compression_ratio"] = float(
        totals.get("sum_collapse_compression_ratio", 0.0) / num_samples
    )
    diagnostics["raw_blank_argmax_ratio"] = float(totals.get("sum_raw_blank_argmax_ratio", 0.0) / num_samples)
    diagnostics["raw_nonblank_argmax_ratio"] = float(
        totals.get("sum_raw_nonblank_argmax_ratio", 0.0) / num_samples
    )
    diagnostics["raw_unique_token_ratio"] = float(totals.get("sum_raw_unique_token_ratio", 0.0) / num_samples)
    diagnostics["raw_dominant_token_ratio"] = float(
        totals.get("sum_raw_dominant_token_ratio", 0.0) / num_samples
    )
    diagnostics["raw_adjacent_repeat_ratio"] = float(
        totals.get("sum_raw_adjacent_repeat_ratio", 0.0) / num_samples
    )
    diagnostics["raw_token_switch_ratio"] = float(totals.get("sum_raw_token_switch_ratio", 0.0) / num_samples)
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

    student_frame_stats = totals.get("student_frame_stats", {})
    for metric_name in ("blank_prob", "nonblank_prob", "entropy", "top1_confidence", "top1_margin"):
        mean, std = _summarize_stat_block(student_frame_stats, metric_name)
        diagnostics[f"frame_{metric_name}_mean"] = float(mean)
        diagnostics[f"frame_{metric_name}_std"] = float(std)

    raw_hist = totals.get("raw_top1_token_histogram", {})
    collapsed_hist = totals.get("collapsed_token_histogram", {})
    diagnostics["raw_top1_token_histogram"] = _histogram_to_rows(raw_hist, tokenizer=tokenizer)
    diagnostics["collapsed_token_histogram"] = _histogram_to_rows(collapsed_hist, tokenizer=tokenizer)

    sample_id_counts = totals.get("sample_id_counts", {})
    unique_sample_ids = float(len(sample_id_counts))
    diagnostics["unique_sample_ids_seen"] = unique_sample_ids
    if expected_num_unique_samples is None:
        expected_num_unique_samples = int(unique_sample_ids)
    diagnostics["sample_coverage_ratio"] = float(unique_sample_ids / max(1, int(expected_num_unique_samples)))
    revisit_total = max(0.0, float(sum(float(v) for v in sample_id_counts.values())) - unique_sample_ids)
    diagnostics["sample_revisit_ratio"] = float(revisit_total / max(1.0, num_samples))

    alignment_stats = totals.get("alignment_stats", {})
    alignment_count = max(1.0, float(alignment_stats.get("count", 0.0)))
    diagnostics["student_time_steps_mean"] = float(
        alignment_stats.get("student_time_steps_sum", 0.0) / alignment_count
    )
    diagnostics["teacher_time_steps_mean"] = float(
        alignment_stats.get("teacher_time_steps_sum", 0.0) / alignment_count
    )
    diagnostics["teacher_to_student_time_ratio_mean"] = float(
        alignment_stats.get("teacher_to_student_time_ratio_sum", 0.0) / alignment_count
    )
    diagnostics["alignment_interpolation_ratio_mean"] = float(
        alignment_stats.get("alignment_interpolation_ratio_sum", 0.0) / alignment_count
    )
    diagnostics["student_frames_per_second_estimate"] = float(
        alignment_stats.get("student_fps_sum", 0.0) / alignment_count
    )
    diagnostics["teacher_frames_per_second_estimate"] = float(
        alignment_stats.get("teacher_fps_sum", 0.0) / alignment_count
    )
    overlap_count = max(1.0, float(alignment_stats.get("teacher_student_valid_frame_overlap_count", 0.0)))
    diagnostics["teacher_student_valid_frame_overlap_ratio"] = float(
        alignment_stats.get("teacher_student_valid_frame_overlap_sum", 0.0) / overlap_count
    )
    diagnostics["teacher_student_alignment_stretch_mean"] = float(
        alignment_stats.get("teacher_student_alignment_stretch_sum", 0.0) / alignment_count
    )

    teacher_distribution_stats = totals.get("teacher_distribution_stats", {})
    for metric_name in ("entropy", "top1_confidence", "top1_margin", "blank_prob", "nonblank_prob"):
        mean, std = _summarize_stat_block(teacher_distribution_stats, metric_name)
        diagnostics[f"teacher_{metric_name}_mean"] = float(mean)
        diagnostics[f"teacher_{metric_name}_std"] = float(std)
    argmax_count = max(1.0, float(teacher_distribution_stats.get("argmax_agreement_count", 0.0)))
    diagnostics["teacher_student_argmax_agreement"] = float(
        teacher_distribution_stats.get("argmax_agreement_sum", 0.0) / argmax_count
    )
    overlap1_count = max(1.0, float(teacher_distribution_stats.get("topk_overlap_at1_count", 0.0)))
    diagnostics["teacher_student_topk_overlap_at1"] = float(
        teacher_distribution_stats.get("topk_overlap_at1_sum", 0.0) / overlap1_count
    )
    overlap3_count = max(1.0, float(teacher_distribution_stats.get("topk_overlap_at3_count", 0.0)))
    diagnostics["teacher_student_topk_overlap_at3"] = float(
        teacher_distribution_stats.get("topk_overlap_at3_sum", 0.0) / overlap3_count
    )

    teacher_feature_stats = totals.get("teacher_feature_stats", {})
    cosine_mean, cosine_std = _summarize_stat_block(teacher_feature_stats, "cosine")
    mse_mean, _ = _summarize_stat_block(teacher_feature_stats, "mse")
    norm_ratio_mean, _ = _summarize_stat_block(teacher_feature_stats, "norm_ratio")
    diagnostics["teacher_student_feature_cosine_mean"] = float(cosine_mean)
    diagnostics["teacher_student_feature_cosine_std"] = float(cosine_std)
    diagnostics["teacher_student_feature_mse_mean"] = float(mse_mean)
    diagnostics["teacher_student_feature_norm_ratio_mean"] = float(norm_ratio_mean)

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
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
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
