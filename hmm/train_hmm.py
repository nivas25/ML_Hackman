from __future__ import annotations
from pathlib import Path
from typing import List

import numpy as np
try:
    from hmmlearn.hmm import MultinomialHMM
    HMMLEARN_AVAILABLE = True
except Exception:
    HMMLEARN_AVAILABLE = False

from utils.helpers import ALPHABET, load_corpus, filter_words_by_charset
import joblib


def words_to_sequences(words: List[str], alphabet: str = ALPHABET) -> List[np.ndarray]:
    idx = {ch: i for i, ch in enumerate(alphabet)}
    seqs = []
    for w in words:
        seqs.append(np.array([idx[ch] for ch in w], dtype=np.int32))
    return seqs


def pad_and_stack(seqs: List[np.ndarray]) -> np.ndarray:
    # Concatenate with lengths for MultinomialHMM.fit
    return np.concatenate(seqs)[:, None]


def lengths(seqs: List[np.ndarray]) -> List[int]:
    return [len(s) for s in seqs]


def train_and_save(corpus_path: Path, out_path: Path, n_components: int = 26 * 2, random_state: int = 42):
    if not HMMLEARN_AVAILABLE:
        print("hmmlearn not available. Please install requirements and retry.")
        return

    words = load_corpus(corpus_path)
    words = filter_words_by_charset(words)
    if not words:
        raise ValueError("Corpus is empty after filtering")

    seqs = words_to_sequences(words)
    X = pad_and_stack(seqs)
    lens = lengths(seqs)

    model = MultinomialHMM(n_components=n_components, n_iter=50, random_state=random_state)
    model.n_features = len(ALPHABET)
    model.fit(X, lengths=lens)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_path)
    print(f"Saved HMM model to {out_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    corpus = root / "data" / "corpus.txt"
    out = Path(__file__).parent / "hmm_model.pkl"
    train_and_save(corpus, out)
