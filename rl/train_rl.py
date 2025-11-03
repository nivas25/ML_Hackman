from __future__ import annotations
from pathlib import Path
from typing import List
from tqdm import trange

from utils.helpers import load_corpus, filter_words_by_charset
from env.hangman_env import HangmanEnv
from rl.agent import TabularQAgent


def train(episodes: int = 5000, max_incorrect: int = 6, save_path: Path = Path("rl/policy.pkl")):
    root = Path(__file__).resolve().parents[1]
    words: List[str] = filter_words_by_charset(load_corpus(root / 'data' / 'corpus.txt'))
    env = HangmanEnv(words, max_incorrect=max_incorrect)
    agent = TabularQAgent()

    rewards = 0.0
    for _ in trange(episodes, desc="Training RL Agent"):
        obs = env.reset()
        done = False
        while not done:
            action = agent.act(obs)
            step = env.step(action)
            agent.learn(obs, action, step.reward, step.observation, step.done)
            obs = step.observation
            done = step.done
            rewards += step.reward

    save_path = (root / save_path).resolve()
    agent.save(save_path)
    print(f"Saved policy to {save_path}")
    return rewards / max(1, episodes)


if __name__ == "__main__":
    train()
