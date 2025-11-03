# env/hangman_env.py

import random

class HangmanEnvironment:
    """
    A Hangman game environment for a Reinforcement Learning agent.
    Implements scoring per provided problem statement:
      - Repeated guess: -2
      - Correct letter: +1 (per step)
      - Wrong guess: -5
      - Win bonus: +50
      - Lose penalty: -50
    """

    def __init__(self, corpus_path: str, max_lives: int = 6):
        self.word_list = self._load_corpus(corpus_path)
        self.max_lives = max_lives
        self.alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Initialize game state variables
        self.secret_word: str = ""
        self.masked_word: str = ""
        self.lives_left: int = 0
        self.guessed_letters: set[str] = set()

    def _load_corpus(self, corpus_path: str):
        """Loads the corpus file into a list of words."""
        try:
            with open(corpus_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read all lines, strip whitespace, and convert to upper
                words = [line.strip().upper() for line in f if line.strip() and line.strip().isalpha()]
            print(f"HangmanEnv: Successfully loaded {len(words)} words from {corpus_path}")
            return words
        except FileNotFoundError:
            print(f"HangmanEnv: Error - Corpus file not found at {corpus_path}")
            return ["TESTWORD"]  # Fallback for safety

    def _get_state(self):
        """Returns the current state (observation) for the RL agent."""
        return {
            "masked_word": self.masked_word,
            "lives_left": self.lives_left,
            "guessed_letters": self.guessed_letters.copy(),
        }

    def reset(self):
        """Starts a new game of Hangman."""
        self.secret_word = random.choice(self.word_list)
        self.masked_word = "_" * len(self.secret_word)
        self.lives_left = self.max_lives
        self.guessed_letters = set()
        return self._get_state()

    def step(self, action_letter: str):
        """
        Takes one step in the environment by guessing a letter.
        Returns: (state, reward, done)
        """
        if not action_letter or len(action_letter) > 1:
             return self._get_state(), -10, False # Heavy penalty for bad action format

        action_letter = str(action_letter).upper()
        if action_letter not in self.alphabet:
             return self._get_state(), -10, False # Heavy penalty for non-alpha action
             
        reward = 0
        done = False

        # 1) Repeated guess penalty
        if action_letter in self.guessed_letters:
            reward = -2
            return self._get_state(), reward, done

        self.guessed_letters.add(action_letter)

        # 2) Correct guess reward
        if action_letter in self.secret_word:
            reward = +1

            new_masked = list(self.masked_word)
            for i, ch in enumerate(self.secret_word):
                if ch == action_letter:
                    new_masked[i] = action_letter
            self.masked_word = "".join(new_masked)

            # Win condition
            if "_" not in self.masked_word:
                reward = +50
                done = True

        else:
            # 3) Wrong guess penalty
            self.lives_left -= 1
            reward = -5

            # Lose condition
            if self.lives_left <= 0:
                reward = -50
                done = True

        return self._get_state(), reward, done

    def render(self):
        """Helper function to print the current game state."""
        print(f"  Board: {self.masked_word}   Lives: {self.lives_left}   Guessed: {sorted(self.guessed_letters)}")