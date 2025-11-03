# main.py

import os
import pickle
import sys
import numpy as np
from tqdm import tqdm 

# --- This block makes local imports work ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(script_dir) # main.py is in the root
sys.path.append(project_root)
# -------------------------------------------

from env.hangman_env import HangmanEnvironment
from hmm.train_hmm import predict_letter_probs
from rl.agent import QLearningAgent

# --- 🛑 THIS IS THE FIX 🛑 ---
# Build paths *from the project root*
CORPUS_PATH = os.path.join(project_root, "data/corpus.txt")
HMM_MODEL_PATH = os.path.join(project_root, "hmm/hmm_model.pkl") 
POLICY_PATH = os.path.join(project_root, "rl/policy.pkl")
# -----------------------------

# --- Evaluation Parameters ---
NUM_GAMES = 2000

def evaluate():
    print("--- 🏆 Starting Final Evaluation ---")
    
    # 1. Load HMM
    print(f"Loading HMM model from {HMM_MODEL_PATH}...")
    with open(HMM_MODEL_PATH, 'rb') as f:
        hmm_model = pickle.load(f)
    print("HMM model loaded.")

    # 2. Load Trained Agent
    print(f"Loading trained RL policy from {POLICY_PATH}...")
    agent = QLearningAgent(epsilon=0.0) # Epsilon=0 for 100% exploitation
    agent.load_policy(POLICY_PATH)
    print("RL policy loaded.")

    # 3. Init Env
    env = HangmanEnvironment(corpus_path=CORPUS_PATH)
    
    # 4. Evaluation Loop
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    
    print(f"Running {NUM_GAMES} evaluation games...")
    for _ in tqdm(range(NUM_GAMES)):
        state_dict = env.reset()
        done = False
        
        while not done:
            state = agent.get_simple_state(state_dict)
            
            hmm_probs = predict_letter_probs(
                state_dict["masked_word"].lower(),
                {l.lower() for l in state_dict["guessed_letters"]},
                hmm_model
            )
            
            # Agent chooses best action
            action_letter = agent.choose_action(
                state, 
                hmm_probs, 
                state_dict["guessed_letters"]
            )
            
            # --- Track guesses BEFORE stepping ---
            if action_letter in state_dict["guessed_letters"]:
                total_repeated_guesses += 1
            elif action_letter not in env.secret_word:
                total_wrong_guesses += 1
            
            new_state_dict, reward, done = env.step(action_letter)
            state_dict = new_state_dict

        # Game over
        if state_dict["lives_left"] > 0:
            total_wins += 1
            
    # 5. Calculate Final Score
    print("\n--- 📊 Evaluation Results ---")
    
    success_rate = total_wins / NUM_GAMES
    avg_wrong = total_wrong_guesses / NUM_GAMES
    avg_repeated = total_repeated_guesses / NUM_GAMES
    
    # From the problem statement's formula:
    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

    print(f"Total Games:    {NUM_GAMES}")
    print(f"Total Wins:     {total_wins} (Success Rate: {success_rate:.2%})")
    print(f"Total Wrong:    {total_wrong_guesses} (Avg: {avg_wrong:.2f})")
    print(f"Total Repeated: {total_repeated_guesses} (Avg: {avg_repeated:.2f})")
    print("-----------------------------------")
    print(f"🏆 FINAL SCORE: {final_score}")
    print("-----------------------------------")

if __name__ == "__main__":
    evaluate()