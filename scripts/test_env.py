from __future__ import annotations
import sys
from pathlib import Path

# Ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from env.hangman_env import HangmanEnvironment  # noqa: E402

# --- Setup ---
CORPUS_PATH = str(ROOT / 'data' / 'corpus.txt')

print(f"Attempting to create environment with corpus: {CORPUS_PATH}")
env = HangmanEnvironment(corpus_path=CORPUS_PATH, max_lives=6)

print("\n--- Starting Test Game ---")
state = env.reset()
print("Game reset. Initial state:")
env.render()

print("\nTesting step with guess 'E'...")
new_state, reward, done = env.step('E')
print("After guess 'E':")
env.render()
print(f"Reward: {reward}, Done: {done}")

print("\nTesting step with guess 'Z'...")
new_state, reward, done = env.step('Z')
print("After guess 'Z':")
env.render()
print(f"Reward: {reward}, Done: {done}")

print(f"\nTest complete. The secret word was: {env.secret_word}")
