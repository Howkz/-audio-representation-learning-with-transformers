from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from src.data.text import CharCTCTokenizer
from src.models.audio_transformer import AudioTransformerCTC, AudioTransformerEncoder


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    positions = torch.arange(max_len, device=lengths.device, dtype=torch.long).unsqueeze(0)
    return positions < lengths.unsqueeze(1)


def _build_encoder_from_sections(model_cfg: Dict[str, Any], audio_cfg: Dict[str, Any]) -> AudioTransformerEncoder:
    n_mels = int(audio_cfg["n_mels"])
    patch_time = int(model_cfg.get("patch_time", 1))
    patch_freq = int(model_cfg.get("patch_freq", n_mels))
    return AudioTransformerEncoder(
        n_mels=n_mels,
        dim=int(model_cfg["dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        patch_size=patch_time,
        max_len=int(model_cfg["max_len"]),
        pos_embed=str(model_cfg.get("pos_embed", "sinusoidal")),
        patch_strategy=str(model_cfg.get("patch_strategy", "time")),
        patch_freq=patch_freq,
    )


def _normalize_temperature(value: float) -> float:
    return max(1e-4, float(value))


@dataclass
class TeacherOutput:
    target_kind: str
    values: torch.Tensor
    lengths: torch.Tensor


class BaseTeacher:
    target_kind: str = "logits"
    requires_waveform: bool = False
    family: str = "unknown"

    def forward_teacher(
        self,
        *,
        x_logmel: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        waveforms: Optional[torch.Tensor],
        waveform_lengths: Optional[torch.Tensor],
        temperature: float,
    ) -> TeacherOutput:
        raise NotImplementedError


class ExternalHFTeacher(BaseTeacher):
    def __init__(
        self,
        *,
        family: str,
        model_name: str,
        tokenizer: CharCTCTokenizer,
        device: torch.device,
    ) -> None:
        self.family = str(family)
        self.model_name = str(model_name)
        self.tokenizer = tokenizer
        self.device = device

        try:
            from transformers import AutoModel, AutoModelForCTC, AutoProcessor
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'transformers'. Install requirements.txt before using external teachers."
            ) from exc

        self.processor = AutoProcessor.from_pretrained(self.model_name)
        if self.family == "wav2vec2_ctc":
            self.model = AutoModelForCTC.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.target_kind = "logits"
            self.requires_waveform = True
            self._vocab_mapping = self._build_vocab_mapping().to(self.device)
        elif self.family == "hubert_features":
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.target_kind = "hidden_states"
            self.requires_waveform = True
            self._vocab_mapping = None
        else:
            raise ValueError(f"Unsupported external teacher family '{self.family}'.")

    def _build_vocab_mapping(self) -> torch.Tensor:
        processor_tokenizer = getattr(self.processor, "tokenizer", None)
        if processor_tokenizer is None:
            raise ValueError("The selected teacher does not expose a tokenizer.")

        vocab = processor_tokenizer.get_vocab()
        if not vocab:
            raise ValueError("Teacher tokenizer vocabulary is empty.")

        vocab_size = max(int(idx) for idx in vocab.values()) + 1
        mapping = torch.full((vocab_size,), fill_value=self.tokenizer.blank_id, dtype=torch.long)
        pad_token = getattr(processor_tokenizer, "pad_token", None)
        word_delim = getattr(processor_tokenizer, "word_delimiter_token", "|")

        for token, idx in vocab.items():
            normalized = str(token)
            target_id = self.tokenizer.blank_id
            if pad_token is not None and normalized == str(pad_token):
                target_id = self.tokenizer.blank_id
            elif normalized == str(word_delim) or normalized == " ":
                target_id = self.tokenizer.char_to_id[" "]
            elif len(normalized) == 1:
                candidate = normalized.lower()
                if candidate in self.tokenizer.char_to_id:
                    target_id = self.tokenizer.char_to_id[candidate]
            mapping[int(idx)] = int(target_id)
        return mapping

    def _output_lengths(self, waveform_lengths: torch.Tensor, output_time_steps: int) -> torch.Tensor:
        if hasattr(self.model, "_get_feat_extract_output_lengths"):
            lengths = self.model._get_feat_extract_output_lengths(waveform_lengths.to(torch.long))
            return torch.clamp(lengths.to(torch.long), min=0, max=int(output_time_steps))
        return torch.full(
            (waveform_lengths.shape[0],),
            fill_value=int(output_time_steps),
            dtype=torch.long,
            device=waveform_lengths.device,
        )

    def _waveform_inputs(
        self,
        waveforms: Optional[torch.Tensor],
        waveform_lengths: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if waveforms is None or waveform_lengths is None:
            raise ValueError("External teachers require waveforms and waveform_lengths.")
        input_values = waveforms.to(self.device, dtype=torch.float32, non_blocking=True)
        input_lengths = waveform_lengths.to(self.device, dtype=torch.long, non_blocking=True)
        attention_mask = _lengths_to_mask(input_lengths, max_len=int(input_values.shape[1])).to(torch.long)
        return input_values, attention_mask

    def forward_teacher(
        self,
        *,
        x_logmel: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        waveforms: Optional[torch.Tensor],
        waveform_lengths: Optional[torch.Tensor],
        temperature: float,
    ) -> TeacherOutput:
        del x_logmel, lengths
        input_values, attention_mask = self._waveform_inputs(waveforms, waveform_lengths)
        with torch.no_grad():
            if self.family == "wav2vec2_ctc":
                outputs = self.model(input_values=input_values, attention_mask=attention_mask)
                logits = outputs.logits.float()
                teacher_lengths = self._output_lengths(
                    waveform_lengths=attention_mask.sum(dim=1),
                    output_time_steps=int(logits.shape[1]),
                ).to(logits.device)
                probs = torch.softmax(logits / _normalize_temperature(temperature), dim=-1)
                mapped = logits.new_zeros((logits.shape[0], logits.shape[1], self.tokenizer.vocab_size))
                mapping = self._vocab_mapping.view(1, 1, -1).expand(logits.shape[0], logits.shape[1], -1)
                mapped.scatter_add_(2, mapping, probs)
                mapped = mapped / mapped.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                return TeacherOutput(
                    target_kind="logits",
                    values=mapped.detach(),
                    lengths=teacher_lengths.detach(),
                )

            outputs = self.model(input_values=input_values, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state.float()
            teacher_lengths = self._output_lengths(
                waveform_lengths=attention_mask.sum(dim=1),
                output_time_steps=int(hidden.shape[1]),
            ).to(hidden.device)
            return TeacherOutput(
                target_kind="hidden_states",
                values=hidden.detach(),
                lengths=teacher_lengths.detach(),
            )


class CheckpointTeacher(BaseTeacher):
    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        fallback_cfg: Dict[str, Any],
        tokenizer: CharCTCTokenizer,
        device: torch.device,
    ) -> None:
        self.family = "checkpoint"
        self.target_kind = "logits"
        self.requires_waveform = False
        self.device = device
        self.tokenizer = tokenizer
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Teacher checkpoint not found: {self.checkpoint_path}")

        payload = torch.load(self.checkpoint_path, map_location=device, weights_only=False)
        student_cfg = payload.get("student_config")
        if not isinstance(student_cfg, dict):
            student_cfg = {
                "model": dict(fallback_cfg["model"]),
                "audio": {"n_mels": int(fallback_cfg["audio"]["n_mels"])},
            }

        encoder = _build_encoder_from_sections(student_cfg["model"], student_cfg["audio"])
        vocab_size = int(payload.get("vocab_size", tokenizer.vocab_size))
        self.model = AudioTransformerCTC(encoder=encoder, vocab_size=vocab_size).to(self.device)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def forward_teacher(
        self,
        *,
        x_logmel: Optional[torch.Tensor],
        lengths: Optional[torch.Tensor],
        waveforms: Optional[torch.Tensor],
        waveform_lengths: Optional[torch.Tensor],
        temperature: float,
    ) -> TeacherOutput:
        del waveforms, waveform_lengths
        if x_logmel is None or lengths is None:
            raise ValueError("Checkpoint teacher requires log-Mel inputs and frame lengths.")
        with torch.no_grad():
            logits, out_lengths = self.model(
                x_logmel.to(self.device, non_blocking=True),
                lengths.to(self.device, non_blocking=True),
            )
            probs = torch.softmax(logits.float() / _normalize_temperature(temperature), dim=-1)
        return TeacherOutput(
            target_kind="logits",
            values=probs.detach(),
            lengths=out_lengths.detach(),
        )


def build_teacher(
    cfg: Dict[str, Any],
    tokenizer: CharCTCTokenizer,
    device: torch.device,
) -> Optional[BaseTeacher]:
    distill_cfg = cfg.get("distillation", {})
    if not isinstance(distill_cfg, dict) or not bool(distill_cfg.get("enabled", False)):
        return None

    teacher_cfg = cfg.get("teacher", {})
    if not isinstance(teacher_cfg, dict):
        raise ValueError("teacher config must be a mapping when distillation is enabled.")

    source = str(teacher_cfg.get("source", "external"))
    if source == "external":
        family = str(teacher_cfg.get("family", "wav2vec2_ctc"))
        model_name = str(teacher_cfg.get("model_name", "facebook/wav2vec2-base-960h"))
        return ExternalHFTeacher(
            family=family,
            model_name=model_name,
            tokenizer=tokenizer,
            device=device,
        )
    if source == "checkpoint":
        checkpoint_path = teacher_cfg.get("checkpoint_path")
        if not checkpoint_path:
            raise ValueError("teacher.checkpoint_path is required when teacher.source=checkpoint.")
        return CheckpointTeacher(
            checkpoint_path=checkpoint_path,
            fallback_cfg=cfg,
            tokenizer=tokenizer,
            device=device,
        )
    raise ValueError(f"Unsupported teacher.source '{source}'.")
