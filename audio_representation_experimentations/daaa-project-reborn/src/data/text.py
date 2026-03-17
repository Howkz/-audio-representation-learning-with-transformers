from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List


def normalize_transcript(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@dataclass
class CharCTCTokenizer:
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        charset = list(" abcdefghijklmnopqrstuvwxyz'")
        # Ensure no duplicate apostrophes in charset list.
        charset = list(dict.fromkeys(charset))
        self.blank_id = 0
        self.id_to_char = [self.blank_token] + charset
        self.char_to_id = {char: idx for idx, char in enumerate(self.id_to_char)}

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_char)

    def encode(self, text: str) -> List[int]:
        normalized = normalize_transcript(text)
        if not normalized:
            return [self.char_to_id[" "]]
        ids = [self.char_to_id[ch] for ch in normalized if ch in self.char_to_id]
        return ids if ids else [self.char_to_id[" "]]

    def decode(self, token_ids: Iterable[int]) -> str:
        chars: List[str] = []
        previous = None
        for idx in token_ids:
            if idx == self.blank_id:
                previous = idx
                continue
            if idx == previous:
                continue
            if 0 <= idx < len(self.id_to_char):
                chars.append(self.id_to_char[idx])
            previous = idx
        text = "".join(chars).replace(self.blank_token, "")
        return re.sub(r"\s+", " ", text).strip()

