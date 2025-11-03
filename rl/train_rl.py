# rl/train_rl.py

import os
import pickle
import sys
import numpy as np
from tqdm import tqdm 
import matplotlib.pyplot as plt

# --- This block makes local imports work ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(project_root)
# -------------------------------------------

from env.hangman_env import HangmanEnvironment
from hmm.train_hmm import predict_letter_probs 
from rl.agent import QLearningAgent

# --- File Paths (relative to project root) ---
CORPUS_PATH = os.path.join(project_root, "data/corpus.txt")
HMM_MODEL_PATH = os.path.join(project_root, "hmm/hmm_model.pkl") 
POLICY_SAVE_PATH = os.path.join(project_root, "rl/policy.pkl")

# --- 🛑 THE 30%+ SUCCESS RATE SETTINGS 🛑 ---
NUM_EPISODES = 500000      # 500k episodes
MIN_EPSILON = 0.01         # Explore down to 1%
EPSILON_DECAY = 0.99999    # Slow, steady decay to prevent plateau
CHECKPOINT_FREQ = 10000    # Save policy every 10,000 episodes
# -----------------------------------------

# --- Plotting Setup ---
PLOT_SAVE_DIR = os.path.join(project_root, "results")
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)
PLOT_SAVE_PATH = os.path.join(PLOT_SAVE_DIR, "learning_curve.png")
# ----------------------

def train():
    print(f"--- 🚀 Starting RL Agent Training ({NUM_EPISODES} Episode Run) ---")
    
    # 1. Load HMM
    print(f"Loading HMM model from {HMM_MODEL_PATH}...")
    with open(HMM_MODEL_PATH, 'rb') as f:
        hmm_model = pickle.load(f)
    print("HMM model loaded.")

    # 2. Init Env and Agent
    env = HangmanEnvironment(corpus_path=CORPUS_PATH)
    agent = QLearningAgent(epsilon=1.0)
    
    # --- Attempt to load a checkpoint if one exists ---
    if os.path.exists(POLICY_SAVE_PATH):
        print(f"Found existing policy at {POLICY_SAVE_PATH}. Resuming training...")
        agent.load_policy(POLICY_SAVE_PATH)
        agent.epsilon = 0.1 # Start with low-ish epsilon if resuming
    # --------------------------------------------------
    
    print(f"Starting training for {NUM_EPISODES} episodes...")
    
    reward_history = []
    plot_episodes = []
    plot_avg_rewards = []
    
    # 3. Training Loop
    for episode in tqdm(range(NUM_EPISODES)):
        state_dict = env.reset()
        state = agent.get_simple_state(state_dict) # Uses (lives, masked_word)
        done = False
        total_reward = 0
        
        while not done:
            hmm_probs = predict_letter_probs(
                state_dict["masked_word"].lower(),
                {l.lower() for l in state_dict["guessed_letters"]},
                hmm_model
            )
            
            action_letter = agent.choose_action(
                state, 
                hmm_probs, 
                state_dict["guessed_letters"]
            )
            
            new_state_dict, reward, done = env.step(action_letter)
            new_state = agent.get_simple_state(new_state_dict)
            
            agent.update(state, action_letter, reward, new_state)
            
            state = new_state
            state_dict = new_state_dict
            total_reward += reward
        
        agent.decay_epsilon(MIN_EPSILON, EPSILON_DECAY)
        reward_history.append(total_reward)
        
        # Log and save policy every 10k episodes
        if (episode + 1) % CHECKPOINT_FREQ == 0:
            avg_reward = np.mean(reward_history[-CHECKPOINT_FREQ:])
            tqdm.write(f"Episode {episode + 1}/{NUM_EPISODES} | Avg Reward: {avg_reward:.2f} | Epsilon: {agent.epsilon:.4f}")
            agent.save_policy(POLICY_SAVE_PATH) # Checkpoint save
            plot_episodes.append(episode + 1)
            plot_avg_rewards.append(avg_reward)
            
    # 4. Save Final Policy
    agent.save_policy(POLICY_SAVE_PATH)
    print("--- 🎉 Training Complete ---")

    # --- 5. GENERATE AND SAVE PLOT ---
    print(f"Generating learning curve plot...")
    plt.figure(figsize=(12, 6))
    plt.plot(plot_episodes, plot_avg_rewards, marker='o', linestyle='-', markersize=4, color='b', label='Avg. Reward')
    
    if len(plot_episodes) > 1:
        z = np.polyfit(plot_episodes, plot_avg_rewards, 2) 
        p = np.poly1d(z)
        plt.plot(plot_episodes, p(plot_episodes), "r--", label="Learning Trend")
        
    plt.title("Agent Learning Curve (Reward per Episode)", fontsize=16)
    plt.xlabel("Episodes", fontsize=12)
    plt.ylabel(f"Average Reward (per {CHECKPOINT_FREQ} Episodes)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(PLOT_SAVE_PATH)
    print(f"✅ Plot successfully saved to: {PLOT_SAVE_PATH}")

if __name__ == "__main__":
    try:
        import matplotlib
    except ImportError:
        print("Matplotlib not found. Installing...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
        import matplotlib.pyplot as plt
    try:
        import sklearn
    except ImportError:
        pass # We aren't using sklearn in this version
        
    train()