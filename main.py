# main.py

import os
import pickle
import sys
import numpy as np
from tqdm import tqdm 

# --- Import 'rich' components for beautiful output ---
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    RICH_INSTALLED = True
except ImportError:
    RICH_INSTALLED = False
# ---------------------------------------------------

# --- This block makes local imports work ---
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(script_dir) 
sys.path.append(project_root)
# -------------------------------------------

from env.hangman_env import HangmanEnvironment
from hmm.train_hmm import predict_letter_probs
from rl.agent import QLearningAgent

# --- File Paths ---
CORPUS_PATH = os.path.join(project_root, "data/corpus.txt")
HMM_MODEL_PATH = os.path.join(project_root, "hmm/hmm_model.pkl")
POLICY_PATH = os.path.join(project_root, "rl/policy.pkl")

# --- Evaluation Parameters ---
NUM_GAMES = 2000

def plain_print(text):
    """Fallback printer if 'rich' is not installed."""
    print(text)

def evaluate():
    if RICH_INSTALLED:
        console = Console()
        console.print(Panel(
            Text("Starting Final Agent Evaluation", style="bold white", justify="center"),
            border_style="blue"
        ))
        status_print = console.log
        header_print = lambda x: console.print(x, style="blue")
        title_print = lambda x: console.print(x, style="bold white", justify="center")
    else:
        console = None
        status_print = plain_print
        header_print = plain_print
        title_print = plain_print
    
    # 1. Load HMM
    status_print(f"Loading HMM model from {HMM_MODEL_PATH}...")
    try:
        with open(HMM_MODEL_PATH, 'rb') as f:
            hmm_model = pickle.load(f)
    except FileNotFoundError:
        print(f"❌ Error: HMM Model not found at {HMM_MODEL_PATH}. Run 'python hmm/train_hmm.py' first.")
        return
    status_print("HMM model loaded.")

    # 2. Load Trained Agent
    status_print(f"Loading trained RL policy from {POLICY_PATH}...")
    try:
        agent = QLearningAgent(epsilon=0.0) # Epsilon=0 for 100% exploitation
        agent.load_policy(POLICY_PATH)
    except FileNotFoundError:
        print(f"❌ Policy file not found at {POLICY_PATH}. Run 'python rl/train_rl.py' first.")
        return
    status_print("RL policy loaded.")

    # 3. Init Env
    status_print("Initializing environment...")
    env = HangmanEnvironment(corpus_path=CORPUS_PATH)
    status_print(f"Environment initialized with {len(env.word_list)} words.")
    
    # 4. Evaluation Loop
    total_wins = 0
    total_wrong_guesses = 0
    total_repeated_guesses = 0
    total_reward_sum = 0.0
    total_steps = 0
    total_steps_on_win = 0
    
    print("\n") # Add a space before the progress bar
    for _ in tqdm(range(NUM_GAMES), desc="Running Evaluation Games"):
        state_dict = env.reset()
        done = False
        
        game_reward = 0.0
        game_steps = 0
        
        while not done:
            # --- Hybrid Agent Decision Making ---
            # 1. Get the simple Q-table state
            state = agent.get_simple_state(state_dict)
            
            # 2. Get HMM probabilities
            hmm_probs = predict_letter_probs(
                state_dict["masked_word"].lower(),
                {l.lower() for l in state_dict["guessed_letters"]},
                hmm_model
            )
            
            # 3. Pass all 3 required arguments
            action_letter = agent.choose_action(
                state, 
                hmm_probs, 
                state_dict["guessed_letters"]
            )
            # ---------------------------
            
            if action_letter in state_dict["guessed_letters"]:
                total_repeated_guesses += 1
            elif action_letter not in env.secret_word:
                total_wrong_guesses += 1
            
            new_state_dict, reward, done = env.step(action_letter)
            state_dict = new_state_dict
            
            game_reward += reward
            game_steps += 1

        # --- Game Over: Update Metrics ---
        total_reward_sum += game_reward
        total_steps += game_steps
        
        if state_dict["lives_left"] > 0: # Check for a win
            total_wins += 1
            total_steps_on_win += game_steps
            
    # --- 5. Calculate Final Metrics ---
    success_rate = total_wins / NUM_GAMES
    final_score = (success_rate * 2000) - (total_wrong_guesses * 5) - (total_repeated_guesses * 2)

    avg_reward = total_reward_sum / NUM_GAMES
    avg_steps = total_steps / NUM_GAMES
    avg_wrong = total_wrong_guesses / NUM_GAMES
    avg_repeated = total_repeated_guesses / NUM_GAMES
    avg_steps_win = total_steps_on_win / total_wins if total_wins > 0 else 0 

    # --- 6. Build the Rich Output ---
    header_print("\n\n" + "="*50)
    title_print("HACKATHON EVALUATION RESULTS")
    header_print("="*50)

    if RICH_INSTALLED:
        # --- Create Table for Core Score ---
        score_table = Table(title="Core Scoring Metrics", border_style="green", show_header=True, header_style="bold magenta")
        score_table.add_column("Metric", style="dim", width=25)
        score_table.add_column("Value", style="bold")
        
        score_table.add_row("Total Games", f"{NUM_GAMES}")
        score_table.add_row("Total Wins", f"{total_wins}")
        score_table.add_row("Success Rate", f"{success_rate:.2%}")
        score_table.add_row("Total Wrong Guesses", f"{total_wrong_guesses}")
        score_table.add_row("Total Repeated Guesses", f"{total_repeated_guesses}")
        
        console.print(score_table)
        
        # --- Show Final Score Panel ---
        console.print(Panel(
            Text(f"{final_score:,.0f}", style="bold bright_green" if final_score > 0 else "bold bright_red", justify="center"),
            title="🏆 FINAL SCORE",
            border_style="bold green" if final_score > 0 else "bold red",
            padding=(1, 4)
        ))

        # --- Create Table for Diagnostics ---
        diag_table = Table(title="Detailed Agent Diagnostics", border_style="cyan", show_header=True, header_style="bold magenta")
        diag_table.add_column("Diagnostic", style="dim", width=25)
        diag_table.add_column("Average per Game", style="bold")

        diag_table.add_row("Avg. Reward / Game", f"{avg_reward:.2f}")
        diag_table.add_row("Avg. Steps / Game", f"{avg_steps:.2f}")
        diag_table.add_row("Avg. Steps on Win", f"{avg_steps_win:.2f}")
        diag_table.add_row("Avg. Wrong Guesses", f"{avg_wrong:.2f}")
        diag_table.add_row("Avg. Repeated Guesses", f"{avg_repeated:.2f}")
        
        console.print(diag_table)
        header_print("="*50)
    else:
        # --- Plain Text Fallback ---
        print("\n--- Core Scoring Metrics ---")
        print(f"  Total Wins:     {total_wins} / {NUM_GAMES}")
        print(f"  Success Rate:   {success_rate:.2%}")
        print(f"  Total Wrong:    {total_wrong_guesses}")
        print(f"  Total Repeated: {total_repeated_guesses}")
        print("-----------------------------------")
        print(f"  FINAL SCORE: {final_score:,.0f}")
        print("-----------------------------------")
        print("\n--- Detailed Agent Diagnostics ---")
        print(f"  Avg. Reward/Game:   {avg_reward:.2f}")
        print(f"  Avg. Steps/Game:    {avg_steps:.2f}")
        print(f"  Avg. Steps on Win:  {avg_steps_win:.2f}")
        print(f"  Avg. Wrong/Game:    {avg_wrong:.2f}")
        print(f"  Avg. Repeated/Game: {avg_repeated:.2f}")
        print("="*40)

if __name__ == "__main__":
    evaluate()