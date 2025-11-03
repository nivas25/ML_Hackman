# ML Hangman: HMM + RL

A small project that trains a Multinomial HMM on a word corpus and a tabular Q-learning agent to play Hangman.

## Structure

- `data/` — word corpus
- `env/` — Hangman environment
- `hmm/` — HMM training and saved model
- `rl/` — RL agent and training loop
- `utils/` — helpers (corpus loading, filters)
- `notebooks/` — HMM training, env demo, RL training
- `results/` — logs and evaluation output
- `main.py` — evaluates policy over multiple games

## Quickstart

```bash
# Windows PowerShell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train HMM (optional)
python .\hmm\train_hmm.py

# Train RL agent
python .\rl\train_rl.py

# Evaluate policy
python .\main.py
```

## Notes
- If `hmmlearn` isn’t installed, HMM training will be skipped.
- Replace `data/corpus.txt` with your own word list for better results.
- Notebooks include a top cell to set up `sys.path` for imports.
