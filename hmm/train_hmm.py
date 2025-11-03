"""
train_hmm.py
------------
Hidden Markov Model-like letter probability oracle for Hangman.

Purpose:
- Train on data/corpus.txt
- Generate a random masked word from HangmanEnvironment
- Output letter probability distribution from the HMM
"""

from pathlib import Path
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Any
import string
import random

# Import your environment
from env.hangman_env import HangmanEnvironment


# -----------------------------------------------------------
# Data Loading
# -----------------------------------------------------------
def load_corpus(path: str) -> List[str]:
    """Load and clean the corpus file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Corpus not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words if w.isalpha()]


# -----------------------------------------------------------
# Model Training
# -----------------------------------------------------------
def build_positional_frequencies(words: List[str]) -> Dict[int, List[Dict[str, float]]]:
    """Build positional letter probability tables."""
    grouped = defaultdict(list)
    for w in words:
        grouped[len(w)].append(w)

    model = {}
    for length, group in grouped.items():
        counters = [Counter() for _ in range(length)]
        for w in group:
            for i, ch in enumerate(w):
                counters[i][ch] += 1
        model[length] = [
            {ch: count / sum(c.values()) for ch, count in c.items()} for c in counters
        ]
    return model


def train_hmm(corpus_path="data/corpus.txt", save_path="hmm_model.pkl"):
    """Train the model and save it."""
    words = load_corpus(corpus_path)
    print(f"Loaded {len(words)} words from corpus.")
    model = build_positional_frequencies(words)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {save_path}")
    return model


# -----------------------------------------------------------
# Prediction
# -----------------------------------------------------------
def predict_letter_probs(masked_word: str, guessed: Set[str], model: Any) -> Dict[str, float]:
    """Return P(letter | masked_word, guessed_letters)."""
    masked_word = masked_word.lower()
    L = len(masked_word)
    alphabet = set(string.ascii_lowercase)
    available = alphabet - guessed

    if not available:
        return {}

    if L not in model:
        # fallback to overall frequencies
        flat_counts = Counter({
            ch: sum(d.get(ch, 0) for probs in model.values() for d in probs)
            for ch in alphabet
        })
        total = sum(flat_counts.values())
        return {ch: flat_counts[ch] / total for ch in available} if total else {ch: 1 / len(available) for ch in available}

    probs = {ch: 0.0 for ch in available}
    for i, sym in enumerate(masked_word):
        if sym == "_":
            for ch in available:
                probs[ch] += model[L][i].get(ch, 0)

    total = sum(probs.values())
    if total == 0:
        return {ch: 1 / len(available) for ch in available}
    return {ch: p / total for ch, p in probs.items()}


# -----------------------------------------------------------
# Random Environment Interaction
# -----------------------------------------------------------
def test_with_env(model):
    """Use HangmanEnvironment to generate a realistic masked word and print letter probabilities."""
    env = HangmanEnvironment("data/corpus.txt")
    state = env.reset()

    # Randomly guess a few letters — mix of vowels and consonants
    all_letters = list("abcdefghijklmnopqrstuvwxyz")
    random.shuffle(all_letters)

    for _ in range(random.randint(3, 6)):  # simulate 3–6 random guesses
        letter = all_letters.pop()
        state, _, done = env.step(letter)
        if done:
            break

    env.render()

    masked = state["masked_word"]
    guessed = env.guessed_letters

    print(f"\n🎯 Masked word: {masked}")
    print(f"Guessed letters: {sorted(guessed)}\n")

    probs = predict_letter_probs(masked, guessed, model)

    print("📊 HMM Probability distribution:")
    for ch, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{ch}: {p:.4f}")



# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    model_path = "hmm/hmm_model.pkl"
    corpus_path = "data/corpus.txt"

    if not Path(model_path).exists():
        model = train_hmm(corpus_path, model_path)
    else:
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Test the HMM on a random environment state
    test_with_env(model)
