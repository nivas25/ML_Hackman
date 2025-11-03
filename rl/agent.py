# rl/agent.py

import numpy as np
import pickle
import random
from collections import defaultdict

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.99, epsilon=1.0):
        self.q_table = defaultdict(lambda: np.zeros(26))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.alphabet_lower = "abcdefghijklmnopqrstuvwxyz"
        self.alphabet_upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.letter_to_int = {letter: i for i, letter in enumerate(self.alphabet_lower)}
        self.int_to_letter = {i: letter for i, letter in enumerate(self.alphabet_lower)}

    def get_simple_state(self, env_state):
        masked_word = env_state["masked_word"]
        lives = env_state["lives_left"]
        return (lives, masked_word) # The "smarter" Q-table state

    def choose_action(self, state, hmm_probs, guessed_letters_upper):
        if random.uniform(0, 1) < self.epsilon:
            available_actions = [l for l in self.alphabet_upper if l not in guessed_letters_upper]
            if not available_actions:
                return random.choice(self.alphabet_upper)
            return random.choice(available_actions)
        
        q_values = np.copy(self.q_table[state])
        
        hmm_scores = np.zeros(26)
        for letter_lower, prob in hmm_probs.items():
            if letter_lower in self.letter_to_int:
                hmm_scores[self.letter_to_int[letter_lower]] = prob

        # --- THIS IS THE KEY LINE TO TUNE ---
        hybrid_scores = q_values + (hmm_scores * 10.0)
        # ------------------------------------

        for i, letter_lower in enumerate(self.alphabet_lower):
            if letter_lower.upper() in guessed_letters_upper:
                hybrid_scores[i] = -np.inf

        best_action_int = np.argmax(hybrid_scores)
        return self.int_to_letter[best_action_int].upper()

    def update(self, state, action_upper, reward, next_state):
        action_int = self.letter_to_int[action_upper.lower()]
        
        old_value = self.q_table[state][action_int]
        next_max = np.max(self.q_table[next_state])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        
        self.q_table[state][action_int] = new_value

    def decay_epsilon(self, min_epsilon=0.01, decay_rate=0.999):
        if self.epsilon > min_epsilon:
            self.epsilon *= decay_rate

    def save_policy(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"✅ Q-table policy saved to {path}")

    def load_policy(self, path):
        with open(path, 'rb') as f:
            self.q_table = defaultdict(lambda: np.zeros(26), pickle.load(f))
        print(f"✅ Q-table policy loaded from {path}")