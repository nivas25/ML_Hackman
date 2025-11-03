from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union
import random


@dataclass
class StepResult:
    observation: Dict[str, Union[str, int, List[int]]]
    reward: float
    done: bool
    info: Dict[str, Union[str, int, float]] = field(default_factory=dict)


class HangmanEnv:
    """
    A simple Hangman environment suitable for tabular RL experiments.

    Contract
    - reset(target_word: Optional[str]) -> observation
    - step(action: Union[int, str]) -> StepResult
    - action space: 0..len(alphabet)-1 or a single lowercase letter
    - observation: dict with keys {masked, remaining, guessed_mask}
    """

    def __init__(self, words: List[str], max_incorrect: int = 6, alphabet: str = "abcdefghijklmnopqrstuvwxyz"):
        assert len(alphabet) == 26, "Default alphabet expected."
        self.words = [w.lower() for w in words if w and w.isalpha()]
        self.max_incorrect = max_incorrect
        self.alphabet = alphabet

        # Episode state
        self.target: str = ""
        self.masked: List[str] = []
        self.guessed: Set[str] = set()
        self.incorrect: int = 0
        self.done: bool = False

    def seed(self, seed: Optional[int] = None):
        random.seed(seed)

    def _sample_word(self) -> str:
        return random.choice(self.words)

    def reset(self, target_word: Optional[str] = None) -> Dict[str, Union[str, int, List[int]]]:
        self.target = (target_word or self._sample_word()).lower()
        self.masked = ["_" for _ in self.target]
        self.guessed = set()
        self.incorrect = 0
        self.done = False
        return self._observation()

    def _observation(self) -> Dict[str, Union[str, int, List[int]]]:
        guessed_mask = [1 if ch in self.guessed else 0 for ch in self.alphabet]
        return {
            "masked": "".join(self.masked),
            "remaining": self.max_incorrect - self.incorrect,
            "guessed_mask": guessed_mask,
        }

    def _apply_guess(self, letter: str) -> Tuple[bool, int]:
        correct = False
        newly_revealed = 0
        for i, ch in enumerate(self.target):
            if ch == letter and self.masked[i] == "_":
                self.masked[i] = ch
                correct = True
                newly_revealed += 1
        return correct, newly_revealed

    def step(self, action: Union[int, str]) -> StepResult:
        if self.done:
            return StepResult(self._observation(), 0.0, True, {"msg": "episode_done"})

        if isinstance(action, int):
            assert 0 <= action < len(self.alphabet), "Action index out of range"
            letter = self.alphabet[action]
        else:
            letter = str(action).lower()
            if letter not in self.alphabet:
                return StepResult(self._observation(), -0.5, False, {"msg": "invalid_action"})

        if letter in self.guessed:
            # Repeated guess small penalty
            return StepResult(self._observation(), -0.1, False, {"msg": "repeat"})

        self.guessed.add(letter)
        correct, newly = self._apply_guess(letter)

        reward = 0.0
        if correct:
            reward = 0.5 * newly
            if "_" not in self.masked:
                self.done = True
                reward += 5.0  # win bonus
        else:
            self.incorrect += 1
            reward = -1.0
            if self.incorrect >= self.max_incorrect:
                self.done = True
                reward -= 2.0  # lose penalty

        return StepResult(self._observation(), reward, self.done, {
            "target": self.target,
            "incorrect": self.incorrect,
        })

    # Utility for demos
    def render(self) -> str:
        return f"{''.join(self.masked)} | remaining={self.max_incorrect - self.incorrect} | guessed={sorted(self.guessed)}"
