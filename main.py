from __future__ import annotations
from pathlib import Path
from typing import List

from utils.helpers import load_corpus, filter_words_by_charset
from env.hangman_env import HangmanEnv
from rl.agent import TabularQAgent

try:
    import joblib
    from hmmlearn.hmm import MultinomialHMM  # noqa: F401
    HMM_OK = True
except Exception:
    HMM_OK = False


def evaluate(n_games: int = 2000) -> float:
    root = Path(__file__).resolve().parent
    words: List[str] = filter_words_by_charset(load_corpus(root / 'data' / 'corpus.txt'))
    env = HangmanEnv(words)

    policy_path = root / 'rl' / 'policy.pkl'
    if policy_path.exists():
        agent = TabularQAgent.load(policy_path)
    else:
        print("Policy not found. Using untrained agent.")
        agent = TabularQAgent()

    hmm_path = root / 'hmm' / 'hmm_model.pkl'
    hmm = None
    if HMM_OK and hmm_path.exists():
        try:
            hmm = joblib.load(hmm_path)
            print("Loaded HMM model.")
        except Exception as e:
            print(f"Failed to load HMM model: {e}")

    total_reward = 0.0
    wins = 0

    for _ in range(n_games):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            step = env.step(action)
            total_reward += step.reward
            obs = step.observation
            done = step.done
        if '_' not in obs['masked']:
            wins += 1

    win_rate = wins / n_games
    avg_reward = total_reward / n_games

    results_dir = root / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / 'logs').mkdir(parents=True, exist_ok=True)
    eval_path = results_dir / 'evaluation.txt'
    with eval_path.open('w', encoding='utf-8') as f:
        f.write(f"games={n_games}\nwin_rate={win_rate:.3f}\navg_reward={avg_reward:.3f}\n")

    print(f"Win rate: {win_rate:.3f} | Avg reward: {avg_reward:.3f}")
    print(f"Saved evaluation to {eval_path}")
    return win_rate


if __name__ == "__main__":
    evaluate()
