"""
train_hmm_v2.py
---------------
Enhanced HMM with multiple improvements:
1. Pattern matching (use revealed letters to filter candidates)
2. Bigram/trigram letter context
3. Letter frequency smoothing
4. Word frequency weighting
"""

from pathlib import Path
import pickle
from collections import defaultdict, Counter
from typing import List, Dict, Set, Any, Tuple
import string
import random
import sys
import os
import re

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.hangman_env import HangmanEnvironment


# -----------------------------------------------------------
# Data Loading
# -----------------------------------------------------------
def load_corpus(path: Path) -> List[str]:
    """Load and clean the corpus file."""
    if not path.exists():
        raise FileNotFoundError(f"Corpus not found: {path}")
    with path.open("r", encoding="utf-8", errors='ignore') as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words if w.isalpha()]


# -----------------------------------------------------------
# Enhanced Model Training
# -----------------------------------------------------------
class EnhancedHMM:
    def __init__(self):
        self.positional_probs = {}  # {length: [{letter: prob}]}
        self.bigrams = defaultdict(Counter)  # {letter: Counter of next letters}
        self.trigrams = defaultdict(Counter)  # {(letter1, letter2): Counter of next letters}
        self.overall_freq = Counter()  # Global letter frequencies
        self.word_list_by_length = defaultdict(list)  # {length: [words]}
        self.alphabet = set(string.ascii_lowercase)
        
    def train(self, words: List[str]):
        """Train all model components."""
        print(f"Training Enhanced HMM on {len(words)} words...")
        
        # Build word lists by length
        grouped = defaultdict(list)
        for w in words:
            grouped[len(w)].append(w)
            self.word_list_by_length[len(w)].append(w)
        
        # 1. Build positional probabilities
        for length, group in grouped.items():
            counters = [Counter() for _ in range(length)]
            for w in group:
                for i, ch in enumerate(w):
                    counters[i][ch] += 1
                    self.overall_freq[ch] += 1
            
            prob_list = []
            for c in counters:
                total = sum(c.values())
                if total > 0:
                    prob_list.append({ch: count / total for ch, count in c.items()})
                else:
                    prob_list.append({})
            self.positional_probs[length] = prob_list
        
        # 2. Build bigrams and trigrams
        for w in words:
            for i in range(len(w) - 1):
                self.bigrams[w[i]][w[i+1]] += 1
            for i in range(len(w) - 2):
                self.trigrams[(w[i], w[i+1])][w[i+2]] += 1
        
        # Normalize bigrams
        for letter, counts in self.bigrams.items():
            total = sum(counts.values())
            if total > 0:
                self.bigrams[letter] = {ch: count / total for ch, count in counts.items()}
        
        # Normalize trigrams
        for pair, counts in self.trigrams.items():
            total = sum(counts.values())
            if total > 0:
                self.trigrams[pair] = {ch: count / total for ch, count in counts.items()}
        
        # Normalize overall frequencies
        total_freq = sum(self.overall_freq.values())
        if total_freq > 0:
            self.overall_freq = {ch: count / total_freq for ch, count in self.overall_freq.items()}
        
        print(f"✅ Trained on {len(self.positional_probs)} word lengths")
        print(f"✅ Built {len(self.bigrams)} bigram patterns")
        print(f"✅ Built {len(self.trigrams)} trigram patterns")
    
    def _pattern_to_regex(self, masked: str) -> re.Pattern:
        """Convert masked word to regex pattern (e.g., '_a_' -> '^.a.$')"""
        pattern = '^' + masked.replace('_', '.') + '$'
        return re.compile(pattern, re.IGNORECASE)
    
    def _filter_by_pattern(self, masked: str, candidates: List[str]) -> List[str]:
        """Filter word list by matching the revealed pattern."""
        pattern = self._pattern_to_regex(masked)
        return [w for w in candidates if pattern.match(w)]
    
    def predict(self, masked_word: str, guessed: Set[str]) -> Dict[str, float]:
        """
        Enhanced prediction combining multiple signals:
        1. Pattern-matched word list
        2. Positional probabilities
        3. Bigram/trigram context
        4. Overall frequency smoothing
        """
        masked_word = masked_word.lower()
        length = len(masked_word)
        available = self.alphabet - guessed
        
        if not available:
            return {}
        
        # Initialize scores
        scores = {ch: 0.0 for ch in available}
        
        # Strategy 1: Pattern matching on candidate words
        if length in self.word_list_by_length:
            candidates = self._filter_by_pattern(masked_word, self.word_list_by_length[length])
            
            if candidates:
                # Count remaining letters in matching words
                letter_counts = Counter()
                for word in candidates:
                    for i, ch in enumerate(word):
                        if masked_word[i] == '_' and ch in available:
                            letter_counts[ch] += 1
                
                if letter_counts:
                    total = sum(letter_counts.values())
                    for ch in available:
                        scores[ch] += 3.0 * (letter_counts.get(ch, 0) / total)  # Heavy weight
        
        # Strategy 2: Positional probabilities
        if length in self.positional_probs:
            for i, sym in enumerate(masked_word):
                if sym == "_" and i < len(self.positional_probs[length]):
                    for ch in available:
                        scores[ch] += 1.5 * self.positional_probs[length][i].get(ch, 0)
        
        # Strategy 3: Bigram context (look at neighbors)
        for i, sym in enumerate(masked_word):
            if sym == "_":
                # Check left neighbor
                if i > 0 and masked_word[i-1] != "_":
                    left = masked_word[i-1]
                    if left in self.bigrams:
                        for ch in available:
                            scores[ch] += 1.0 * self.bigrams[left].get(ch, 0)
                
                # Check right neighbor
                if i < length - 1 and masked_word[i+1] != "_":
                    right = masked_word[i+1]
                    # Use reverse bigrams (what comes before right)
                    for ch in available:
                        if ch in self.bigrams and right in self.bigrams[ch]:
                            scores[ch] += 1.0 * self.bigrams[ch][right]
                
                # Trigram context
                if i > 1 and masked_word[i-1] != "_" and masked_word[i-2] != "_":
                    pair = (masked_word[i-2], masked_word[i-1])
                    if pair in self.trigrams:
                        for ch in available:
                            scores[ch] += 0.5 * self.trigrams[pair].get(ch, 0)
        
        # Strategy 4: Overall frequency smoothing (fallback)
        for ch in available:
            scores[ch] += 0.3 * self.overall_freq.get(ch, 0)
        
        # Normalize
        total_score = sum(scores.values())
        if total_score == 0:
            return {ch: 1.0 / len(available) for ch in available}
        
        return {ch: score / total_score for ch, score in scores.items()}


def train_enhanced_hmm(corpus_path: Path, save_path: Path):
    """Train and save the enhanced HMM."""
    words = load_corpus(corpus_path)
    model = EnhancedHMM()
    model.train(words)
    
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Enhanced model saved to {save_path}")
    return model


def load_enhanced_hmm(model_path: Path) -> EnhancedHMM:
    """Load a saved enhanced HMM."""
    with open(model_path, "rb") as f:
        return pickle.load(f)


# -----------------------------------------------------------
# Interactive Test
# -----------------------------------------------------------
def test_with_env(model: EnhancedHMM, corpus_path: Path):
    """Test the enhanced HMM with a random game."""
    env = HangmanEnvironment(str(corpus_path))
    state = env.reset()
    
    all_letters = list("abcdefghijklmnopqrstuvwxyz")
    random.shuffle(all_letters)
    
    for _ in range(random.randint(3, 6)):
        if not all_letters:
            break
        letter = all_letters.pop()
        state, _, done = env.step(letter.upper())
        if done:
            break
    
    env.render()
    
    masked = state["masked_word"]
    guessed_set = state["guessed_letters"]
    
    masked_lower = masked.lower()
    guessed_lower = {g.lower() for g in guessed_set}
    
    print(f"\n🎯 Masked word: {masked}")
    print(f"Guessed letters: {sorted(guessed_set)}\n")
    
    probs = model.predict(masked_lower, guessed_lower)
    
    print("📊 Enhanced HMM Probability distribution (Top 10):")
    for ch, p in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ch}: {p:.4f}")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    model_path = ROOT / "hmm" / "hmm_model_v2.pkl"
    corpus_path = ROOT / "data" / "corpus.txt"
    
    os.makedirs(ROOT / "hmm", exist_ok=True)
    
    if not model_path.exists():
        print("Enhanced model not found. Training...")
        model = train_enhanced_hmm(corpus_path, model_path)
    else:
        print(f"Loading existing enhanced model from {model_path}...")
        model = load_enhanced_hmm(model_path)
    
    print("\n--- Running Enhanced HMM Test ---")
    test_with_env(model, corpus_path)
