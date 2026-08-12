"""
data_prep.py
Reads all simulation summary CSVs and hardware trial CSVs.

Confirmed file structures:
  Hardcoded U-maze : trial_N.csv
    cols: trial, timestamp, algorithm, collisions, completion_time_s,
          wp2_time_s, wp3_time_s, goal_reached, timeout, notes
  Hardcoded Z-maze : trial_N.csv
    cols: trial, timestamp, algorithm, maze, collisions,
          completion_time_s, goal_reached, timeout, notes
  Sim episode files:
    usimppo_trialN.csv / usimql_trialN.csv
    zsimppo_trialN.csv / zsimql_trialN.csv
    cols: trial, timestamp, algorithm, maze, collisions,
          completion_time_s, goal_reached, timeout, notes
  Sim summary files:
    usimppo_trialN_summary.csv / usimql_trialN_summary.csv
    zsimppo_trialN_summary.csv / zsimql_trialN_summary.csv
    cols: episode, ep_reward, collisions, success, timeout, steps,
          epsilon, completion_time_s, elapsed_s, episodes_to_converge
Produces:
  sim_master.csv          (80 rows: 20 x 4 conditions)
  hardware_master.csv     (40 rows: 20 x 2 conditions)
  sim_episodes_clean.csv  (40000 rows: episode-level data for plots)
"""
import pandas as pd
import numpy as np
import glob
import os

#   PATHS                                    
BASE = r"C:\Users\adith\Desktop\Uni\Gituni\Y3\2306514_COMP303\Mujoco\Final Experiment"

U_PPO_DIR = os.path.join(BASE, "U_Maze", "U_sim_ppo_data")
U_QL_DIR  = os.path.join(BASE, "U_Maze", "U_sim_ql_data")
U_HW_DIR  = os.path.join(BASE, "U_Maze", "U_hardcode_data")

Z_PPO_DIR = os.path.join(BASE, "Z_Maze", "Z_sim_ppo_data")
Z_QL_DIR  = os.path.join(BASE, "Z_Maze", "Z_sim_ql_data")
Z_HW_DIR  = os.path.join(BASE, "Z_Maze", "Z_hardcode_data")

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


#   HELPER: SIMULATION SUMMARIES                        
def load_sim_summaries(folder, prefix, algorithm, maze):
    """
    Loads all _summary.csv files.
    Summary cols: episode, ep_reward, collisions, success, timeout,
                  steps, epsilon, completion_time_s, elapsed_s,
                  episodes_to_converge
    Returns one aggregated row per run.
    """
    pattern = os.path.join(folder, f"{prefix}*_summary.csv")
    files   = sorted(glob.glob(pattern))
    print(f"  Found {len(files)} summary files for {algorithm} {maze}")
    if len(files) == 0:
        print(f"  WARNING: no files matching {pattern}")

    rows = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, low_memory=False)

        mean_cols    = pd.to_numeric(df["collisions"],  errors="coerce").mean()
        mean_time    = pd.to_numeric(df["elapsed_s"],   errors="coerce").mean()
        success_rate = pd.to_numeric(df["success"],     errors="coerce").mean()

        conv_series = pd.to_numeric(df["episodes_to_converge"], errors="coerce").dropna()
        episodes_to_converge = int(conv_series.iloc[0]) if len(conv_series) > 0 else 500

        rows.append({
            "run":                  i + 1,
            "algorithm":            algorithm,
            "maze":                 maze,
            "platform":             "simulation",
            "mean_collisions":      round(mean_cols,    2),
            "mean_elapsed_s":       round(mean_time,    4),
            "success_rate":         round(success_rate, 4),
            "episodes_to_converge": episodes_to_converge,
        })

    return pd.DataFrame(rows)


#   HELPER: HARDWARE TRIALS                           
def load_hardware_trials(folder, maze):
    """
    Loads all trial_N.csv files.
    U-maze cols: trial, timestamp, algorithm, collisions,
                 completion_time_s, wp2_time_s, wp3_time_s,
                 goal_reached, timeout, notes
    Z-maze cols: trial, timestamp, algorithm, maze, collisions,
                 completion_time_s, goal_reached, timeout, notes
    Returns one row per trial, standardised.
    """
    pattern = os.path.join(folder, "trial_*.csv")
    files   = sorted(
        glob.glob(pattern),
        key=lambda x: int(
            os.path.basename(x).replace("trial_", "").replace(".csv", "")
        )
    )
    print(f"  Found {len(files)} hardware trial files for {maze}")
    if len(files) == 0:
        print(f"  WARNING: no files matching {pattern}")

    rows = []
    for f in files:
        df  = pd.read_csv(f, low_memory=False)
        row = df.iloc[0].to_dict()
        row["maze"]     = maze
        row["platform"] = "hardware"
        rows.append(row)

    return pd.DataFrame(rows)


#   HELPER: SIMULATION EPISODES                         
def load_sim_episodes(folder, prefix, algorithm, maze):
    """
    Loads all per-episode trial CSVs (NOT summary/qtable/model files).
    Episode cols: trial, timestamp, algorithm, maze, collisions,
                  completion_time_s, goal_reached, timeout, notes
    Returns all 500 episodes per run with run number stamped.
    """
    pattern = os.path.join(folder, f"{prefix}trial*.csv")
    files   = sorted([
        f for f in glob.glob(pattern)
        if "summary" not in os.path.basename(f)
        and "qtable"  not in os.path.basename(f)
        and "model"   not in os.path.basename(f)
    ])
    print(f"  Found {len(files)} episode files for {algorithm} {maze}")

    KEEP = ["trial", "timestamp", "algorithm", "maze",
            "collisions", "completion_time_s", "goal_reached",
            "timeout", "notes"]

    dfs = []
    for i, f in enumerate(files):
        df = pd.read_csv(f, low_memory=False)

        # Keep only columns we know about
        df = df[[c for c in KEEP if c in df.columns]]

        # Force numeric on key columns
        for col in ["trial", "collisions", "goal_reached", "timeout"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows where both trial and collisions are NaN
        # (handles any ghost/header rows without crashing)
        mask = pd.Series([True] * len(df))
        if "trial" in df.columns:
            mask = mask & df["trial"].notna()
        if "collisions" in df.columns:
            mask = mask & df["collisions"].notna()
        df = df[mask]

        # Remove any leftover header-ghost rows
        if "algorithm" in df.columns:
            df = df[df["algorithm"].isin(["PPO", "Q-Learning"])]

        # Stamp correct values
        df["run"]       = i + 1
        df["algorithm"] = algorithm
        df["maze"]      = maze

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


#   LOAD SIMULATION SUMMARIES                          
print("LOADING SIMULATION SUMMARIES")

u_ppo = load_sim_summaries(U_PPO_DIR, "usimppo_",  "PPO",        "U-maze")
u_ql  = load_sim_summaries(U_QL_DIR,  "usimql_",   "Q-Learning", "U-maze")
z_ppo = load_sim_summaries(Z_PPO_DIR, "zsimppo_",  "PPO",        "Z-maze")
z_ql  = load_sim_summaries(Z_QL_DIR,  "zsimql_",   "Q-Learning", "Z-maze")

sim_master = pd.concat([u_ppo, u_ql, z_ppo, z_ql], ignore_index=True)
sim_out    = os.path.join(OUT_DIR, "sim_master.csv")
sim_master.to_csv(sim_out, index=False)

print(f"\nsim_master.csv written ({len(sim_master)} rows)")
print(sim_master.groupby(["algorithm", "maze"])["run"].count().to_string())


#   LOAD HARDWARE TRIALS                            
print("LOADING HARDWARE TRIALS")

u_hw = load_hardware_trials(U_HW_DIR, "U-maze")
z_hw = load_hardware_trials(Z_HW_DIR, "Z-maze")

for df in [u_hw, z_hw]:
    df["algorithm"] = "Hardcoded"

hardware_master = pd.concat([u_hw, z_hw], ignore_index=True)
hardware_master = hardware_master.rename(columns={
    "collisions":        "mean_collisions",
    "completion_time_s": "mean_elapsed_s",
})
hardware_master["episodes_to_converge"] = np.nan
hardware_master["success_rate"]         = hardware_master["goal_reached"].astype(float)

hw_out = os.path.join(OUT_DIR, "hardware_master.csv")
hardware_master.to_csv(hw_out, index=False)

print(f"\nhardware_master.csv written ({len(hardware_master)} rows)")
print(hardware_master.groupby(["maze", "algorithm"])["mean_collisions"].count().to_string())


#   LOAD EPISODE-LEVEL DATA                           
print("LOADING EPISODE-LEVEL DATA")

u_ppo_ep = load_sim_episodes(U_PPO_DIR, "usimppo_",  "PPO",        "U-maze")
u_ql_ep  = load_sim_episodes(U_QL_DIR,  "usimql_",   "Q-Learning", "U-maze")
z_ppo_ep = load_sim_episodes(Z_PPO_DIR, "zsimppo_",  "PPO",        "Z-maze")
z_ql_ep  = load_sim_episodes(Z_QL_DIR,  "zsimql_",   "Q-Learning", "Z-maze")

sim_episodes = pd.concat([u_ppo_ep, u_ql_ep, z_ppo_ep, z_ql_ep], ignore_index=True)
ep_out       = os.path.join(OUT_DIR, "sim_episodes_clean.csv")
sim_episodes.to_csv(ep_out, index=False)

print(f"\nsim_episodes_clean.csv written ({len(sim_episodes)} rows)")
print(sim_episodes.groupby(["algorithm", "maze"])["run"].count().to_string())

print("DONE, files ready for R:")
print(f"  {sim_out}")
print(f"  {hw_out}")
print(f"  {ep_out}")