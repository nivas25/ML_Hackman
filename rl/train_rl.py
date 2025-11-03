# rl/train_rl.py

import os
import pickle
import sys
import numpy as np

# --- This block makes local imports work ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# -------------------------------------------

from env.hangman_env import HangmanEnvironment
from hmm.train_hmm import predict_letter_probs # Using the original function
from rl.agent import QLearningAgent

# --- 🛑 THIS IS THE FIX 🛑 ---
# We build absolute paths from the project_root variable
CORPUS_PATH = os.path.join(project_root, "data/corpus.txt")
HMM_MODEL_PATH = os.path.join(project_root, "hmm/hmm_model.pkl") 
POLICY_SAVE_PATH = os.path.join(project_root, "rl/policy.pkl")
# -----------------------------

# --- THIS IS THE 30% SUCCESS RATE CONFIG ---
NUM_EPISODES = 500000  # 500k episodes
MIN_EPSILON = 0.01
EPSILON_DECAY = 0.999995 # Slower decay for 500k
# -------------------------------------------

def train():
    print("--- 🚀 Starting RL Agent Training (500k Episode Run) ---")
    
    # 1. Load HMM
    print(f"Loading HMM model from {HMM_MODEL_PATH}...")
    try:
        with open(HMM_MODEL_PATH, 'rb') as f:
            hmm_model = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ HMM Model not found at {HMM_MODEL_PATH}")
        print("Please run 'python hmm/train_hmm.py' from the root folder first!")
        return
    print("HMM model loaded.")

    # 2. Init Env and Agent
    env = HangmanEnvironment(corpus_path=CORPUS_PATH)
    agent = QLearningAgent(epsilon=1.0)
    
    print(f"Starting training for {NUM_EPISODES} episodes...")
    reward_history = []
    
    # 3. Training Loop
    for episode in range(NUM_EPISODES):
        state_dict = env.reset()
        state = agent.get_simple_state(state_dict)
        done = False
        total_reward = 0
        
        while not done:
            # Get HMM probabilities (all lowercase)
            hmm_probs = predict_letter_probs(
                state_dict["masked_word"].lower(),
                {l.lower() for l in state_dict["guessed_letters"]},
                hmm_model
            )
            
            # Agent chooses action (returns uppercase)
            action_letter = agent.choose_action(
                state, 
                hmm_probs, 
                state_dict["guessed_letters"]
            )
            
            # Env takes uppercase action
            new_state_dict, reward, done = env.step(action_letter)
            
            new_state = agent.get_simple_state(new_state_dict)
            
            # Agent learns
            agent.update(state, action_letter, reward, new_state)
            
            state = new_state
            state_dict = new_state_dict
            total_reward += reward
        
        agent.decay_epsilon(MIN_EPSILON, EPSILON_DECAY)
        reward_history.append(total_reward)
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(reward_history[-1000:])
            print(f"Episode {episode + 1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
            
    # 4. Save Final Policy
    agent.save_policy(POLICY_SAVE_PATH)
    print("--- 🎉 Training Complete ---")

if __name__ == "__main__":
    train()