# HMM Model Improvements

## Overview

We've created an **Enhanced HMM** (`train_hmm_v2.py`) that is significantly more powerful than the original positional-frequency model. It uses multiple strategies to predict the best next letter.

## Key Improvements

### 1. **Pattern Matching** (Most Powerful)

- Filters the word list to only words matching the current revealed pattern
- Example: For `"_A_"`, only considers 3-letter words with 'A' in position 2
- Dramatically improves accuracy when some letters are revealed

### 2. **Bigram Context**

- Learns which letters commonly follow/precede each other
- Example: After 'Q', almost always predicts 'U'
- Helps with common letter combinations like 'TH', 'NG', 'CH'

### 3. **Trigram Context**

- Learns 3-letter patterns (e.g., 'ING', 'THE')
- More precise than bigrams for specific contexts

### 4. **Weighted Scoring**

- Pattern matching: **3.0x** weight (strongest signal)
- Positional probabilities: **1.5x** weight
- Bigram context: **1.0x** weight
- Trigram context: **0.5x** weight
- Overall frequency: **0.3x** weight (fallback/smoothing)

## Files

- **`train_hmm_v2.py`** - Enhanced model with all improvements
- **`evaluate_hmm.py`** - Comprehensive evaluation script
- **`train_hmm.py`** - Original positional-only model (baseline)

## Usage

### Train Enhanced Model

```bash
python hmm/train_hmm_v2.py
```

Creates: `hmm/hmm_model_v2.pkl`

### Evaluate Performance

```bash
python hmm/evaluate_hmm.py
```

This will:

- Test prediction accuracy on 1000+ random game states
- Show Top-1, Top-3, Top-5, Top-10 accuracy
- Break down performance by game stage (early vs late game)
- Compare original vs enhanced model (if both exist)

### Expected Performance

**Enhanced HMM (v2):**

- Top-1 Accuracy: **35-45%** (predicts correct letter as #1 choice)
- Top-3 Accuracy: **60-75%** (correct letter in top 3)
- Top-5 Accuracy: **75-85%** (correct letter in top 5)
- Average Rank: **3-5** (how high the correct letter ranks)

**Original HMM (baseline):**

- Top-1 Accuracy: **20-30%**
- Top-3 Accuracy: **45-60%**
- Top-5 Accuracy: **60-75%**
- Average Rank: **5-7**

Performance improves as more letters are revealed (pattern matching becomes more effective).

## Evaluation Metrics Explained

### Top-K Accuracy

- **Top-1**: Model's #1 prediction is correct
- **Top-3**: Correct letter is in top 3 predictions
- **Top-5**: Correct letter is in top 5 predictions

Higher is better. These measure how confident and accurate the model is.

### Average Rank

- Where does the correct letter typically appear in predictions?
- Lower is better (rank 1 = perfect)
- A rank of 3-5 means the correct letter is usually in top 5

### By Game Stage

- **Early game** (0-3 guesses): Harder, relies on frequency
- **Mid game** (4-7 guesses): Pattern matching kicks in
- **Late game** (8+ guesses): Very accurate with revealed pattern

## Integration with RL Agent

The enhanced HMM can be used in `rl/train_rl.py` by:

1. Loading `hmm_model_v2.pkl` instead of `hmm_model.pkl`
2. Calling `model.predict(masked, guessed)`
3. The RL agent uses these probabilities to guide exploration

## Further Improvements (Advanced)

- **Word frequency weighting**: Common words weighted higher
- **Letter position variance**: Adjust confidence by position entropy
- **Ensemble methods**: Combine multiple prediction strategies
- **Neural networks**: LSTM/Transformer for sequence modeling

## Quick Commands

```bash
# Train enhanced model
python hmm/train_hmm_v2.py

# Evaluate it
python hmm/evaluate_hmm.py

# Use in RL training (update train_rl.py to load hmm_model_v2.pkl)
python rl/train_rl.py
```
