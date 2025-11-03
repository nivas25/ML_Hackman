from __future__ import annotations
from pathlib import Path
from typing import Dict, Optional
import pickle
import random

from utils.helpers import ALPHABET


class TabularQAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.95, epsilon: float = 0.1, alphabet: str = ALPHABET):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.alphabet = alphabet
        self.q: Dict[str, list] = {}  # state_key -> q-values list[len(alphabet)]

    def _ensure_state(self, state_key: str):
        if state_key not in self.q:
            self.q[state_key] = [0.0 for _ in self.alphabet]

    def _state_key(self, observation: dict) -> str:
        # Use masked pattern + guessed mask as a simple state key
        guessed_mask = ''.join(map(str, observation.get('guessed_mask', [])))
        return f"{observation.get('masked','')}|{guessed_mask}"

    def act(self, observation: dict) -> int:
        state = self._state_key(observation)
        self._ensure_state(state)
        if random.random() < self.epsilon:
            return random.randrange(len(self.alphabet))
        qvals = self.q[state]
        maxq = max(qvals)
        # Break ties randomly
        candidates = [i for i, v in enumerate(qvals) if v == maxq]
        return random.choice(candidates)

    def learn(self, obs: dict, action: int, reward: float, next_obs: dict, done: bool):
        s = self._state_key(obs)
        self._ensure_state(s)
        ns = self._state_key(next_obs)
        self._ensure_state(ns)
        best_next = 0.0 if done else max(self.q[ns])
        td_target = reward + self.gamma * best_next
        self.q[s][action] += self.alpha * (td_target - self.q[s][action])

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('wb') as f:
            pickle.dump({
                'alpha': self.alpha,
                'gamma': self.gamma,
                'epsilon': self.epsilon,
                'alphabet': self.alphabet,
                'q': self.q,
            }, f)

    @classmethod
    def load(cls, path: Path) -> 'TabularQAgent':
        with path.open('rb') as f:
            state = pickle.load(f)
        agent = cls(state['alpha'], state['gamma'], state['epsilon'], state['alphabet'])
        agent.q = state['q']
        return agent
