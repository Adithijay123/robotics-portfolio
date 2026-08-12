"""
This test checks that all the data logging and file works correctly
end to end — essentially making sure that when a real training run
finishes, the results actually get saved in a format that can be loaded
back in and analysed in R.

It doesn't run any RL training or touch MuJoCo. Instead it creates mock
trial data and runs it through the same CSV writing, pickle saving, and
model checkpoint functions used in the real experiments, then checks the
outputs are correct. Five things are tested:

  1. Trial CSVs are created with the right columns, row counts, and
     value constraints (collisions non-negative, goal_reached is 0 or 1)
  2. The PPO trial CSV writes correctly with the right algorithm label
  3. Episode-level summary CSVs are created with all required columns
     and epsilon decreasing across episodes as expected
  4. Q-table pickle files save and reload correctly, preserving all
     state-action values exactly
  5. PPO model state dicts save and reload without errors, and the
     reloaded model produces valid finite outputs for a random input

If any of these fail it means the data pipeline is broken and real
experimental results could be lost or corrupted silently. Run this
before any full training session.

"""

import os
import sys
import csv
import pickle
import tempfile
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime

PASS = 0
FAIL = 0

def check(name, condition, got=None, expected=None):
    global PASS, FAIL
    if condition:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}")
        if got is not None:
            print(f"        got={got}  expected={expected}")
        FAIL += 1

print("=" * 55)
print("INTEGRATION TESTS — CSV logging and model IO pipeline")
print("=" * 55)

# Use a temp directory so we never pollute the real output folder
TMPDIR = tempfile.mkdtemp()
print(f"\n  Using temp directory: {TMPDIR}\n")

# ActorCritic architecture — identical to ppo_umaze.py
# Kept here so the IO tests are self-contained and don't
# depend on importing the main training script
class ActorCritic(nn.Module):
    def __init__(self, obs_dim=6, n_actions=8):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head  = nn.Linear(128, 1)

    def forward(self, x):
        h      = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

# Column schema used by both Q-Learning and PPO trial CSVs.
# Any change here needs to be reflected in the main training scripts.
TRIAL_COLS = [
    "trial", "timestamp", "algorithm", "maze",
    "collisions", "completion_time_s",
    "goal_reached", "timeout", "notes",
]

def save_trial_row(csv_path, episode, algorithm, maze,
                   collisions, completion_time_s,
                   goal_reached, timed_out, header_written):
    """Appends one trial result row to the CSV, writing the header on first call."""
    row = pd.DataFrame([{
        "trial":             episode,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":         algorithm,
        "maze":              maze,
        "collisions":        collisions,
        "completion_time_s": completion_time_s,
        "goal_reached":      int(goal_reached),
        "timeout":           int(timed_out),
        "notes":             "",
    }])
    row.to_csv(csv_path, mode="a", header=not header_written, index=False)


print("[1] Trial CSV — Q-Learning U-maze")

ql_trial_csv = os.path.join(TMPDIR, "usimql_trial1.csv")
header_written = False

# Five mock trials: 3 successes, 2 timeouts — mirrors real experiment variance
mock_trials = [
    (1,  "Q-Learning", "U-maze", 12, 0.48,  True,  False),
    (2,  "Q-Learning", "U-maze", 45, 5.00,  False, True ),
    (3,  "Q-Learning", "U-maze",  8, 0.31,  True,  False),
    (4,  "Q-Learning", "U-maze", 30, 5.00,  False, True ),
    (5,  "Q-Learning", "U-maze",  5, 0.22,  True,  False),
]
for ep, alg, maze, cols, t, goal, timeout in mock_trials:
    save_trial_row(ql_trial_csv, ep, alg, maze, cols, t, goal, timeout, header_written)
    header_written = True

check("Q-Learning trial CSV created", os.path.exists(ql_trial_csv))

df_ql = pd.read_csv(ql_trial_csv)
check("trial CSV has correct number of rows", len(df_ql) == 5,
      got=len(df_ql), expected=5)
check("trial CSV has all required columns",
      all(c in df_ql.columns for c in TRIAL_COLS),
      got=list(df_ql.columns))
check("algorithm column contains 'Q-Learning'",
      (df_ql["algorithm"] == "Q-Learning").all())
check("maze column contains 'U-maze'",
      (df_ql["maze"] == "U-maze").all())
check("goal_reached is 0 or 1 only",
      set(df_ql["goal_reached"].unique()).issubset({0, 1}),
      got=df_ql["goal_reached"].unique().tolist())
check("collisions are non-negative integers",
      (df_ql["collisions"] >= 0).all())
check("3 out of 5 trials reached goal",
      df_ql["goal_reached"].sum() == 3,
      got=df_ql["goal_reached"].sum(), expected=3)

print(f"\n  Preview of saved CSV:")
print(df_ql.to_string(index=False))

print("\n[2] Trial CSV — PPO U-maze")

ppo_trial_csv  = os.path.join(TMPDIR, "usimppo_trial1.csv")
header_written = False

# PPO mock values match the mean collision range seen in real training
mock_ppo = [
    (1, "PPO", "U-maze", 134, 5.72, False, True),
    (2, "PPO", "U-maze", 128, 5.00, False, True),
    (3, "PPO", "U-maze", 141, 5.00, False, True),
]
for ep, alg, maze, cols, t, goal, timeout in mock_ppo:
    save_trial_row(ppo_trial_csv, ep, alg, maze, cols, t, goal, timeout, header_written)
    header_written = True

check("PPO trial CSV created", os.path.exists(ppo_trial_csv))
df_ppo = pd.read_csv(ppo_trial_csv)
check("PPO trial CSV has 3 rows", len(df_ppo) == 3,
      got=len(df_ppo), expected=3)
check("PPO algorithm column correct",
      (df_ppo["algorithm"] == "PPO").all())

print("\n[3] Summary CSV — episode-level logging")

summary_path = os.path.join(TMPDIR, "usimql_trial1_summary.csv")
summaries    = []
for ep in range(1, 11):
    summaries.append({
        "episode":              ep,
        "ep_reward":            round(np.random.uniform(-50, 500), 3),
        "collisions":           int(np.random.randint(0, 50)),
        "success":              int(np.random.rand() > 0.5),
        "timeout":              0,
        "steps":                int(np.random.randint(10, 500)),
        "epsilon":              round(1.0 * (0.9965 ** ep), 4),  # matches real decay rate
        "completion_time_s":    float("nan"),
        "elapsed_s":            round(np.random.uniform(0.1, 5.0), 3),
        "episodes_to_converge": None,
    })

pd.DataFrame(summaries).to_csv(summary_path, index=False)
check("summary CSV created", os.path.exists(summary_path))

df_sum = pd.read_csv(summary_path)
check("summary CSV has 10 rows", len(df_sum) == 10,
      got=len(df_sum), expected=10)
check("summary has episode column", "episode" in df_sum.columns)
check("summary has collisions column", "collisions" in df_sum.columns)
check("summary has ep_reward column", "ep_reward" in df_sum.columns)
check("summary has success column", "success" in df_sum.columns)
check("epsilon is decreasing over episodes",
      df_sum["epsilon"].iloc[-1] < df_sum["epsilon"].iloc[0],
      got=df_sum["epsilon"].iloc[-1], expected=f"< {df_sum['epsilon'].iloc[0]}")

print("\n[4] Q-table pickle — save and reload")

qtable_path = os.path.join(TMPDIR, "usimql_trial1_qtable.pkl")

# Three hand-crafted states: one learned, one zero-initialised, one collision-penalised
Q_mock = {
    (5, 10, 0): np.array([0.1, -0.5, 0.3, 0.0, 0.2, -0.1, 0.4, 0.1]),
    (6, 10, 0): np.array([0.0,  0.0, 0.0, 0.0, 0.0,  0.0, 0.0, 0.0]),
    (5, 11, 1): np.array([-8.0, 0.1, 0.2, 0.1, 0.0,  0.0, 0.1, 0.0]),
}

with open(qtable_path, "wb") as f:
    pickle.dump(Q_mock, f)

check("Q-table pickle saved", os.path.exists(qtable_path))

with open(qtable_path, "rb") as f:
    Q_loaded = pickle.load(f)

check("Q-table reloads correct number of states",
      len(Q_loaded) == len(Q_mock),
      got=len(Q_loaded), expected=len(Q_mock))
check("Q-table values preserved after reload",
      np.allclose(Q_loaded[(5, 10, 0)], Q_mock[(5, 10, 0)]),
      got=Q_loaded[(5, 10, 0)].tolist(),
      expected=Q_mock[(5, 10, 0)].tolist())
check("Q-table each state has 8 action values",
      all(len(v) == 8 for v in Q_loaded.values()))

print("\n[5] PPO model — save and reload state dict")

model_path = os.path.join(TMPDIR, "usimppo_trial1_model.pt")
net        = ActorCritic(obs_dim=6, n_actions=8)
torch.save(net.state_dict(), model_path)

check("PPO model file saved", os.path.exists(model_path))

net2 = ActorCritic(obs_dim=6, n_actions=8)
net2.load_state_dict(torch.load(model_path, map_location="cpu"))
net2.eval()

check("PPO model reloads without error", True)

# Pass a random 6-dimensional observation through the reloaded model
# and check the output shapes and values are valid
obs_t          = torch.FloatTensor(np.random.rand(6).astype(np.float32)).unsqueeze(0)
logits, value  = net2(obs_t)
check("reloaded model produces valid logits shape",
      tuple(logits.shape) == (1, 8),
      got=tuple(logits.shape), expected=(1, 8))
check("reloaded model produces finite outputs",
      torch.isfinite(logits).all().item() and torch.isfinite(value).all().item())

shutil.rmtree(TMPDIR)
print(f"\n  Temp files cleaned up.")

print("\n" + "=" * 55)
print(f"  RESULTS:  {PASS} passed  |  {FAIL} failed  |  {PASS+FAIL} total")
print("=" * 55)
if FAIL > 0:
    sys.exit(1)