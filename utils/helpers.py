from __future__ import annotations
from pathlib import Path
from typing import Iterable, List
import random

ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def load_corpus(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    words = [w.strip().lower() for w in text.splitlines()]
    words = [w for w in words if w and w.isalpha()]
    return words


def filter_words_by_charset(words: Iterable[str], alphabet: str = ALPHABET) -> List[str]:
    keep = []
    aset = set(alphabet)
    for w in words:
        if set(w) <= aset:
            keep.append(w)
    return keep


def random_word(words: List[str]) -> str:
    return random.choice(words)
