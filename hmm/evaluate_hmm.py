"""
evaluate_hmm.py
---------------
Comprehensive evaluation script for HMM models.
Measures performance on predicting correct letters in Hangman.
"""

from pathlib import Path
import pickle
import sys
import random
from typing import List, Dict, Set, Tuple
from collections import defaultdict
import string

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.hangman_env import HangmanEnvironment

# Import EnhancedHMM class so pickle can deserialize it
try:
    from hmm.train_hmm_v2 import EnhancedHMM
except ImportError:
    # If import fails, try alternate path
    import sys
    sys.path.insert(0, str(ROOT))
    from train_hmm_v2 import EnhancedHMM


def load_corpus(path: Path) -> List[str]:
    """Load word list."""
    with path.open("r", encoding="utf-8", errors='ignore') as f:
        words = [w.strip().lower() for w in f if w.strip()]
    return [w for w in words if w and w.isalpha()]


def simulate_game_state(word: str, num_guesses: int = 0) -> Tuple[str, Set[str]]:
    """
    Simulate a partially played game by randomly guessing some letters.
    Returns (masked_word, guessed_letters)
    """
    word = word.lower()
    guessed = set()
    
    if num_guesses > 0:
        # Randomly select letters to guess (mix of correct and incorrect)
        all_letters = list(string.ascii_lowercase)
        random.shuffle(all_letters)
        guessed = set(all_letters[:num_guesses])
    
    # Build masked word
    masked = ''.join([ch if ch in guessed else '_' for ch in word])
    
    return masked, guessed


def evaluate_prediction_rank(model, word: str, masked: str, guessed: Set[str]) -> Dict[str, any]:
    """
    Evaluate how well the model predicts the next correct letter.
    Returns metrics about the prediction quality.
    """
    word = word.lower()
    remaining_letters = set(word) - guessed
    available_letters = set(string.ascii_lowercase) - guessed
    
    if not remaining_letters or not available_letters:
        return None
    
    # Get model predictions
    try:
        if hasattr(model, 'predict'):
            # Enhanced HMM
            probs = model.predict(masked, guessed)
        else:
            # Original HMM (functional API)
            probs = model(masked, guessed)
    except Exception as e:
        print(f"Error predicting for word '{word}': {e}")
        return None
    
    if not probs:
        return None
    
    # Sort by probability (descending)
    sorted_predictions = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    # Find ranks of correct letters
    ranks = []
    for letter in remaining_letters:
        for rank, (predicted_letter, _) in enumerate(sorted_predictions, 1):
            if predicted_letter == letter:
                ranks.append(rank)
                break
    
    if not ranks:
        return None
    
    best_rank = min(ranks)
    avg_rank = sum(ranks) / len(ranks)
    
    return {
        'best_rank': best_rank,
        'avg_rank': avg_rank,
        'num_remaining': len(remaining_letters),
        'top1_hit': best_rank == 1,
        'top3_hit': best_rank <= 3,
        'top5_hit': best_rank <= 5,
        'top10_hit': best_rank <= 10,
    }


def evaluate_hmm(model, words: List[str], num_samples: int = 1000, num_guesses_range: Tuple[int, int] = (0, 10)):
    """
    Comprehensive evaluation of HMM model.
    
    Args:
        model: The HMM model to evaluate
        words: List of words to test on
        num_samples: Number of random samples to test
        num_guesses_range: Range of partial game states to test (min_guesses, max_guesses)
    """
    print(f"\n{'='*70}")
    print(f"🔬 EVALUATING HMM MODEL")
    print(f"{'='*70}")
    print(f"Sample size: {num_samples} words")
    print(f"Game state range: {num_guesses_range[0]}-{num_guesses_range[1]} guesses\n")
    
    # Sample random words
    test_words = random.sample(words, min(num_samples, len(words)))
    
    # Metrics by number of guesses
    metrics_by_stage = defaultdict(lambda: {
        'total': 0,
        'top1': 0,
        'top3': 0,
        'top5': 0,
        'top10': 0,
        'ranks': [],
    })
    
    overall_metrics = {
        'total': 0,
        'top1': 0,
        'top3': 0,
        'top5': 0,
        'top10': 0,
        'ranks': [],
    }
    
    for word in test_words:
        # Test at different game stages
        for num_guesses in range(num_guesses_range[0], min(num_guesses_range[1], len(word))):
            masked, guessed = simulate_game_state(word, num_guesses)
            
            # Skip if word is fully revealed
            if '_' not in masked:
                continue
            
            result = evaluate_prediction_rank(model, word, masked, guessed)
            
            if result:
                stage = num_guesses
                metrics_by_stage[stage]['total'] += 1
                metrics_by_stage[stage]['top1'] += result['top1_hit']
                metrics_by_stage[stage]['top3'] += result['top3_hit']
                metrics_by_stage[stage]['top5'] += result['top5_hit']
                metrics_by_stage[stage]['top10'] += result['top10_hit']
                metrics_by_stage[stage]['ranks'].append(result['best_rank'])
                
                overall_metrics['total'] += 1
                overall_metrics['top1'] += result['top1_hit']
                overall_metrics['top3'] += result['top3_hit']
                overall_metrics['top5'] += result['top5_hit']
                overall_metrics['top10'] += result['top10_hit']
                overall_metrics['ranks'].append(result['best_rank'])
    
    # Print results
    print("📊 OVERALL PERFORMANCE")
    print("-" * 70)
    if overall_metrics['total'] > 0:
        print(f"Total predictions: {overall_metrics['total']}")
        print(f"Top-1 Accuracy:  {100 * overall_metrics['top1'] / overall_metrics['total']:.2f}%")
        print(f"Top-3 Accuracy:  {100 * overall_metrics['top3'] / overall_metrics['total']:.2f}%")
        print(f"Top-5 Accuracy:  {100 * overall_metrics['top5'] / overall_metrics['total']:.2f}%")
        print(f"Top-10 Accuracy: {100 * overall_metrics['top10'] / overall_metrics['total']:.2f}%")
        avg_rank = sum(overall_metrics['ranks']) / len(overall_metrics['ranks'])
        print(f"Average Best Rank: {avg_rank:.2f}")
    
    print(f"\n📈 PERFORMANCE BY GAME STAGE")
    print("-" * 70)
    print(f"{'Stage':<12} {'Samples':<10} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10} {'Avg Rank':<10}")
    print("-" * 70)
    
    for stage in sorted(metrics_by_stage.keys()):
        m = metrics_by_stage[stage]
        if m['total'] > 0:
            top1_pct = 100 * m['top1'] / m['total']
            top3_pct = 100 * m['top3'] / m['total']
            top5_pct = 100 * m['top5'] / m['total']
            avg_rank = sum(m['ranks']) / len(m['ranks'])
            print(f"{stage} guesses   {m['total']:<10} {top1_pct:<9.1f}% {top3_pct:<9.1f}% {top5_pct:<9.1f}% {avg_rank:<10.2f}")
    
    print(f"\n{'='*70}\n")
    
    return overall_metrics


def compare_models(model1, model1_name: str, model2, model2_name: str, words: List[str], num_samples: int = 500):
    """Compare two HMM models side by side."""
    print(f"\n{'='*70}")
    print(f"⚖️  MODEL COMPARISON: {model1_name} vs {model2_name}")
    print(f"{'='*70}\n")
    
    print(f"Evaluating {model1_name}...")
    metrics1 = evaluate_hmm(model1, words, num_samples, (0, 8))
    
    print(f"\nEvaluating {model2_name}...")
    metrics2 = evaluate_hmm(model2, words, num_samples, (0, 8))
    
    print(f"\n{'='*70}")
    print(f"📊 COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {model1_name:<20} {model2_name:<20} {'Winner':<15}")
    print("-" * 70)
    
    if metrics1['total'] > 0 and metrics2['total'] > 0:
        m1_top1 = 100 * metrics1['top1'] / metrics1['total']
        m2_top1 = 100 * metrics2['top1'] / metrics2['total']
        winner1 = model1_name if m1_top1 > m2_top1 else model2_name
        print(f"{'Top-1 Accuracy':<25} {m1_top1:<19.2f}% {m2_top1:<19.2f}% {winner1:<15}")
        
        m1_top3 = 100 * metrics1['top3'] / metrics1['total']
        m2_top3 = 100 * metrics2['top3'] / metrics2['total']
        winner3 = model1_name if m1_top3 > m2_top3 else model2_name
        print(f"{'Top-3 Accuracy':<25} {m1_top3:<19.2f}% {m2_top3:<19.2f}% {winner3:<15}")
        
        m1_top5 = 100 * metrics1['top5'] / metrics1['total']
        m2_top5 = 100 * metrics2['top5'] / metrics2['total']
        winner5 = model1_name if m1_top5 > m2_top5 else model2_name
        print(f"{'Top-5 Accuracy':<25} {m1_top5:<19.2f}% {m2_top5:<19.2f}% {winner5:<15}")
        
        m1_rank = sum(metrics1['ranks']) / len(metrics1['ranks'])
        m2_rank = sum(metrics2['ranks']) / len(metrics2['ranks'])
        winner_rank = model1_name if m1_rank < m2_rank else model2_name
        print(f"{'Average Rank':<25} {m1_rank:<19.2f}  {m2_rank:<19.2f}  {winner_rank:<15}")
    
    print(f"{'='*70}\n")


# -----------------------------------------------------------
# Main
# -----------------------------------------------------------
if __name__ == "__main__":
    corpus_path = ROOT / "data" / "corpus.txt"
    original_model_path = ROOT / "hmm" / "hmm_model.pkl"
    enhanced_model_path = ROOT / "hmm" / "hmm_model_v2.pkl"
    
    print("Loading corpus...")
    words = load_corpus(corpus_path)
    print(f"✅ Loaded {len(words)} words\n")
    
    # Check which models exist
    has_original = original_model_path.exists()
    has_enhanced = enhanced_model_path.exists()
    
    if not has_original and not has_enhanced:
        print("❌ No models found!")
        print("Please train a model first:")
        print("  - Original: python hmm/train_hmm.py")
        print("  - Enhanced: python hmm/train_hmm_v2.py")
        sys.exit(1)
    
    # Load and evaluate models
    if has_enhanced:
        print(f"Loading enhanced model from {enhanced_model_path}...")
        with open(enhanced_model_path, "rb") as f:
            enhanced_model = pickle.load(f)
        
        evaluate_hmm(enhanced_model, words, num_samples=1000, num_guesses_range=(0, 10))
    
    if has_original and has_enhanced:
        # Load original model for comparison
        print(f"\nLoading original model from {original_model_path}...")
        with open(original_model_path, "rb") as f:
            original_model_dict = pickle.load(f)
        
        # Wrap original model in a callable
        from hmm.train_hmm import predict_letter_probs
        
        class OriginalHMMWrapper:
            def __init__(self, model_dict):
                self.model = model_dict
            
            def predict(self, masked: str, guessed: Set[str]) -> Dict[str, float]:
                return predict_letter_probs(masked, guessed, self.model)
        
        original_model = OriginalHMMWrapper(original_model_dict)
        
        # Compare models
        compare_models(
            original_model, "Original HMM",
            enhanced_model, "Enhanced HMM",
            words, num_samples=500
        )
    
    elif has_original:
        print(f"Loading original model from {original_model_path}...")
        with open(original_model_path, "rb") as f:
            model_dict = pickle.load(f)
        
        from hmm.train_hmm import predict_letter_probs
        
        class OriginalHMMWrapper:
            def __init__(self, model_dict):
                self.model = model_dict
            
            def predict(self, masked: str, guessed: Set[str]) -> Dict[str, float]:
                return predict_letter_probs(masked, guessed, self.model)
        
        original_model = OriginalHMMWrapper(model_dict)
        evaluate_hmm(original_model, words, num_samples=1000, num_guesses_range=(0, 10))
