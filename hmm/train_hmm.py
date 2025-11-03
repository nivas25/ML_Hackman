"""
train_hmm.py
------------
Hidden Markov Model-like letter probability oracle for Hangman.
"""

from pathlib import Path
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Any
import string
import random
import sys, os

# --- This block makes all imports work, regardless of where you run it ---
# It finds the root "ML_Hackathon" folder and adds it to the system path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ----------------------------------------------------------------------

# Now the import will always work
from env.hangman_env import HangmanEnvironment


# -----------------------------------------------------------
# Data Loading
# -----------------------------------------------------------
def load_corpus(path: Path) -> List[str]:
    """Load and clean the corpus file."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    # Use errors='ignore' for robust parsing of any weird characters
    with path.open("r", encoding="utf-8", errors='ignore') as f:
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
        
        # --- Robustness Fix ---
        # Handle cases where a position might have 0 letters (e.g., from bad data)
        # to prevent ZeroDivisionError.
        prob_list = []
        for c in counters:
            total = sum(c.values())
            if total > 0:
                prob_list.append({ch: count / total for ch, count in c.items()})
            else:
                prob_list.append({}) # Add an empty dict if no letters found
        model[length] = prob_list
        # ----------------------
        
    return model


def train_hmm(corpus_path: Path, save_path: Path):
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
        # Fallback to a flat probability
        return {ch: 1 / len(available) for ch in available}

    probs = {ch: 0.0 for ch in available}
    total_weight = 0.0 # Track if we find any probabilities
    
    for i, sym in enumerate(masked_word):
        if sym == "_":
            if i < len(model[L]): # Ensure position exists
                total_weight += 1.0
                for ch in available:
                    # model[Length][Position_Index].get(character, 0)
                    probs[ch] += model[L][i].get(ch, 0)

    total_prob_sum = sum(probs.values())
    if total_prob_sum == 0 or total_weight == 0:
        # Fallback if no letters in our model match the blanks
        return {ch: 1 / len(available) for ch in available}
        
    # Normalize the final probabilities
    return {ch: p / total_prob_sum for ch, p in probs.items()}


# -----------------------------------------------------------
# Random Environment Interaction
# -----------------------------------------------------------
def test_with_env(model, corpus_path_for_env: Path):
    """Use HangmanEnvironment to test the HMM model."""
    
    # We pass the *absolute path* to the env
    env = HangmanEnvironment(str(corpus_path_for_env))
    state = env.reset()

    all_letters = list("abcdefghijklmnopqrstuvwxyz")
    random.shuffle(all_letters)

    for _ in range(random.randint(3, 6)):
        if not all_letters: break
        letter = all_letters.pop()
        # Env expects uppercase letters
        state, _, done = env.step(letter.upper())
        if done:
            break

    env.render()

    masked = state["masked_word"]       # Uppercase
    guessed_set = state["guessed_letters"] # Uppercase

    # Convert to lowercase for the prediction function
    masked_lower = masked.lower()
    guessed_lower = {g.lower() for g in guessed_set}

    print(f"\n🎯 Masked word: {masked}")
    print(f"Guessed letters: {sorted(guessed_set)}\n")

    probs = predict_letter_probs(masked_lower, guessed_lower, model)

    print("📊 HMM Probability distribution (Top 10):")
    for ch, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ch}: {p:.4f}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    # --- This is the key fix ---
    # Use the 'ROOT' variable to build absolute paths.
    # This will now work no matter where you run the script from.
    model_path = ROOT / "hmm/hmm_model.pkl"
    corpus_path = ROOT / "data/corpus.txt"
    # ---------------------------

    # Ensure the 'hmm' directory exists
    os.makedirs(ROOT / "hmm", exist_ok=True)

    if not model_path.exists():
        print("Model not found. Training...")
        model = train_hmm(corpus_path, model_path)
    else:
        print(f"Loading existing model from {model_path}...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    # Test the HMM
    print("\n--- Running HMM Test ---")
    test_with_env(model, corpus_path)