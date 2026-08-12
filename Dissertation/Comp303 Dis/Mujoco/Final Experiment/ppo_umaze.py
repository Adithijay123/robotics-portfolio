"""
PPO agent navigating UR5e through U-maze in MuJoCo.
Same maze, arm, start/goal, reward logic, and output format as ql_umaze.py
6-dim observation 
1500 episodes to give PPO fair learning time vs Q-Learning's 500
Entropy coefficient annealed to keep exploration alive early on
References:
  Schulman et al. (2017) "Proximal Policy Optimization Algorithms"
"""
import os
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import mujoco
import mujoco.viewer

 
#   PATHS  (identical to ql_umaze.py)
 
XML_PATH   = "umaze.xml"
OUTPUT_DIR = "U_sim_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

 
#   ENVIRONMENT CONSTANTS  (unchanged from ql_umaze.py)
 
EPISODES        = 500          # PPO needs more episodes than Q-Learning
MAX_STEPS       = 500
STEP_SIZE       = 0.005         # 5 mm per action
GOAL_THRESH     = 0.012         # 12 mm goal radius
SAVE_EVERY      = 100
CONVERGE_WINDOW    = 100
CONVERGE_THRESHOLD = 0.80

#   Real arm waypoints   
HOME_QPOS   = np.deg2rad([-93.84, -73.27, -141.51, -55.21, 89.94, -3.84])
HOME_TARGET = np.array([-0.154, -0.325, 0.16642])   # green / start
GOAL_TARGET = np.array([-0.026, -0.325, 0.16635])   # red   / goal

#   Reward values  (identical to ql_umaze.py)
REWARD_GOAL           =  500.0
REWARD_COLLISION      =   -8.0
REWARD_PROGRESS_SCALE =  400.0
REWARD_STEP_PENALTY   =   -0.05

#   8-directional action set   
ACTIONS = np.array([
    [ STEP_SIZE,  0.0       ],
    [-STEP_SIZE,  0.0       ],
    [ 0.0,        STEP_SIZE ],
    [ 0.0,       -STEP_SIZE ],
    [ STEP_SIZE,  STEP_SIZE ],
    [ STEP_SIZE, -STEP_SIZE ],
    [-STEP_SIZE,  STEP_SIZE ],
    [-STEP_SIZE, -STEP_SIZE ],
], dtype=np.float64)
N_ACTIONS = 8

 
#   PPO HYPERPARAMETERS
 
GAMMA          = 0.95      # same discount as ql_umaze.py
GAE_LAMBDA     = 0.95      # GAE-lambda (Schulman et al. 2016)
CLIP_EPS       = 0.2       # PPO clip (Schulman et al. 2017)
LR             = 3e-4
ENTROPY_START  = 0.05      # annealed -> ENTROPY_END over training
ENTROPY_END    = 0.005
VALUE_COEF     = 0.5
MAX_GRAD_NORM  = 0.5
UPDATE_EPOCHS  = 6
MINI_BATCH     = 64
ROLLOUT_LEN    = 32       # collect 32 steps then update (not once per ep)

# Observation: [x_norm, y_norm, dx_norm, dy_norm, dist_norm, prev_col] -> 6 floats
OBS_DIM = 6

 
#   LOAD MUJOCO MODEL
print(f"Loading model: {os.path.abspath(XML_PATH)}")
model    = mujoco.MjModel.from_xml_path(XML_PATH)
data     = mujoco.MjData(model)
N_JOINTS = 6

_PROBE_TIP_ID = model.site("probe_tip").id

_WALL_NAMES = [
    "u_left_outer",  "u_left_inner",  "u_left_cap",
    "u_base_outer",  "u_base_inner",
    "u_right_inner", "u_right_outer", "u_right_cap",
]
_WALL_IDS = set()
for name in _WALL_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
        _WALL_IDS.add(gid)
    else:
        print(f"  WARNING: geom '{name}' not found")

_PROBE_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")
print(f"  Wall geoms found : {len(_WALL_IDS)}")
print(f"  Probe geom ID   : {_PROBE_GEOM_ID}")

 
#   ENVIRONMENT HELPERS 
def get_tcp():
    return data.site_xpos[_PROBE_TIP_ID].copy()

def move_tip_to(target_xyz, max_iter=40, tol=0.001):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        err = target_xyz - data.site_xpos[_PROBE_TIP_ID]
        if np.linalg.norm(err) < tol:
            return True
        mujoco.mj_jacSite(model, data, jacp, jacr, _PROBE_TIP_ID)
        J  = jacp[:, :N_JOINTS]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)
        data.qpos[:N_JOINTS] = np.clip(
            data.qpos[:N_JOINTS] + dq * 0.5,
            model.jnt_range[:N_JOINTS, 0],
            model.jnt_range[:N_JOINTS, 1],
        )
    mujoco.mj_forward(model, data)
    return False

def reset_env():
    mujoco.mj_resetData(model, data)
    data.qpos[:N_JOINTS] = HOME_QPOS
    data.qvel[:N_JOINTS] = 0.0
    data.ctrl[:N_JOINTS] = HOME_QPOS
    mujoco.mj_forward(model, data)
    move_tip_to(HOME_TARGET, max_iter=60, tol=0.001)
    data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
    mujoco.mj_forward(model, data)

def check_collision():
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == _PROBE_GEOM_ID and c.geom2 in _WALL_IDS) or \
           (c.geom2 == _PROBE_GEOM_ID and c.geom1 in _WALL_IDS):
            return True
    return False

 
#   STARTUP
reset_env()
mujoco.mj_forward(model, data)
_HOME_TCP = get_tcp().copy()
LOCKED_Z  = _HOME_TCP[2]
_GOAL_POS = GOAL_TARGET.copy()

X_MIN, X_MAX = -0.175, -0.005
Y_MIN, Y_MAX = -0.400, -0.308
X_SPAN = X_MAX - X_MIN
Y_SPAN = Y_MAX - Y_MIN

print(f"\n  Home TCP  : {_HOME_TCP}")
print(f"  Goal pos  : {_GOAL_POS}")
print(f"  Locked Z  : {LOCKED_Z:.4f} m")
print(f"  Contacts at rest: {data.ncon}  (should be 0)\n")

 
#   OBSERVATION BUILDER 6-dim
#   [x_norm, y_norm, dx_to_goal_norm, dy_to_goal_norm, dist_norm, prev_col] 
#   the relative dx/dy vector makes sure it points towards the goal 
#   and isn't just a raw position which could be learned as a lookup table.
 
def build_obs(tcp, prev_col: bool) -> np.ndarray:
    x_norm    = float(np.clip((tcp[0] - X_MIN) / X_SPAN, 0.0, 1.0))
    y_norm    = float(np.clip((tcp[1] - Y_MIN) / Y_SPAN, 0.0, 1.0))
    dx        = _GOAL_POS[0] - tcp[0]
    dy        = _GOAL_POS[1] - tcp[1]
    dx_norm   = float(np.clip(dx / X_SPAN, -1.0, 1.0))
    dy_norm   = float(np.clip(dy / Y_SPAN, -1.0, 1.0))
    dist      = float(np.linalg.norm([dx, dy]))
    dist_norm = float(np.clip(dist / (X_SPAN + Y_SPAN), 0.0, 1.0))
    col_flag  = float(prev_col)
    return np.array([x_norm, y_norm, dx_norm, dy_norm, dist_norm, col_flag],
                    dtype=np.float32)

 
#  PPO ACTOR-CRITIC NETWORK
class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head  = nn.Linear(128, 1)

        # Orthogonal init -- common PPO best practice
        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, x):
        h      = self.shared(x)
        logits = self.policy_head(h)
        value  = self.value_head(h).squeeze(-1)
        return logits, value

    def act(self, obs_np: np.ndarray):
        obs_t         = torch.FloatTensor(obs_np).unsqueeze(0)
        logits, value = self(obs_t)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0)

    def evaluate(self, obs_t, actions_t):
        logits, values = self(obs_t)
        dist           = Categorical(logits=logits)
        return dist.log_prob(actions_t), values, dist.entropy()


ac_net    = ActorCritic(OBS_DIM, N_ACTIONS)
optimizer = optim.Adam(ac_net.parameters(), lr=LR, eps=1e-5)

 
# PPO UPDATE
def compute_gae(rewards, values, dones, last_value=0.0):
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0
    next_val   = last_value
    for t in reversed(range(T)):
        delta         = rewards[t] + GAMMA * next_val * (1 - dones[t]) - values[t]
        gae           = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
        advantages[t] = gae
        next_val      = values[t]
    returns = advantages + np.array(values, dtype=np.float32)
    return torch.FloatTensor(returns), torch.FloatTensor(advantages)


def ppo_update(buf, entropy_coef: float):
    obs        = torch.FloatTensor(np.array(buf["obs"]))
    actions    = torch.LongTensor(buf["actions"])
    lp_old     = torch.stack(buf["log_probs"])
    returns    = buf["returns"]
    advantages = buf["advantages"]

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    T = obs.shape[0]
    for _ in range(UPDATE_EPOCHS):
        idxs = torch.randperm(T)
        for start in range(0, T, MINI_BATCH):
            mb              = idxs[start: start + MINI_BATCH]
            lp_new, values, entropy = ac_net.evaluate(obs[mb], actions[mb])
            ratio           = torch.exp(lp_new - lp_old[mb])
            surr1           = ratio * advantages[mb]
            surr2           = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * advantages[mb]
            p_loss          = -torch.min(surr1, surr2).mean()
            v_loss          = nn.functional.mse_loss(values, returns[mb])
            loss            = p_loss + VALUE_COEF * v_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(ac_net.parameters(), MAX_GRAD_NORM)
            optimizer.step()

 
#   OUTPUT 
existing  = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("usimppo_trial")]
run_num   = len(existing) // 3 + 1
TRIAL_CSV      = os.path.join(OUTPUT_DIR, f"usimppo_trial{run_num}.csv")
SUMMARY_PATH   = os.path.join(OUTPUT_DIR, f"usimppo_trial{run_num}_summary.csv")
MODEL_PATH_OUT = os.path.join(OUTPUT_DIR, f"usimppo_trial{run_num}_model.pt")
summaries      = []
_trial_hdr     = False

def save_trial_row(episode, collisions, completion_time_s, goal_reached, timed_out):
    global _trial_hdr
    row = pd.DataFrame([{
        "trial":             episode,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":         "PPO",
        "maze":              "U-maze",
        "collisions":        collisions,
        "completion_time_s": completion_time_s,
        "goal_reached":      int(goal_reached),
        "timeout":           int(timed_out),
        "notes":             "",
    }])
    row.to_csv(TRIAL_CSV, mode="a", header=not _trial_hdr, index=False)
    _trial_hdr = True

episodes_to_converge = None
success_history      = []

def check_converged(ep):
    global episodes_to_converge
    if episodes_to_converge is not None:
        return
    if len(success_history) >= CONVERGE_WINDOW:
        if np.mean(success_history[-CONVERGE_WINDOW:]) >= CONVERGE_THRESHOLD:
            episodes_to_converge = ep + 1
            print(f"\n    CONVERGED at episode {episodes_to_converge}\n")

 
#   TRAINING LOOP 
print(f"PPO v2 | U-maze | {EPISODES} eps | step={STEP_SIZE*1000:.0f}mm | obs_dim={OBS_DIM}")
print(f"Rollout every {ROLLOUT_LEN} steps | {UPDATE_EPOCHS} epochs | mini-batch {MINI_BATCH}")
print(f"Output: {OUTPUT_DIR}\n")

best_reward = -np.inf
global_step = 0

# Rolling rollout buffer (spans episode boundaries,standard PPO)
buf = {"obs": [], "actions": [], "log_probs": [],
       "rewards": [], "values": [], "dones": []}

with mujoco.viewer.launch_passive(model, data) as viewer:

    for ep in range(EPISODES):

        if not viewer.is_running():
            print("Viewer closed.")
            break

        reset_env()
        tcp      = get_tcp()
        prev_col = False
        obs      = build_obs(tcp, prev_col)

        ep_reward    = 0.0
        ep_cols      = 0
        ep_ok        = False
        ep_timeout   = False
        t0           = time.time()
        completion_t = float("nan")

        # Entropy annealing: linear decay over all episodes
        frac         = ep / max(EPISODES - 1, 1)
        entropy_coef = ENTROPY_START + frac * (ENTROPY_END - ENTROPY_START)

        for step in range(MAX_STEPS):

            action_idx, log_prob, value = ac_net.act(obs)

            cur    = get_tcp()
            target = np.array([
                cur[0] + ACTIONS[action_idx, 0],
                cur[1] + ACTIONS[action_idx, 1],
                LOCKED_Z
            ])

            safe_qpos = data.qpos[:N_JOINTS].copy()
            move_tip_to(target, max_iter=20, tol=0.001)
            data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
            mujoco.mj_step(model, data)
            viewer.sync()

            tcp_new   = get_tcp()
            collision = check_collision()
            dist_new  = np.linalg.norm(tcp_new[:2] - _GOAL_POS[:2])
            dist_old  = np.linalg.norm(cur[:2]     - _GOAL_POS[:2])

            if collision:
                data.qpos[:N_JOINTS] = safe_qpos
                data.ctrl[:N_JOINTS] = safe_qpos
                mujoco.mj_forward(model, data)
                tcp_new  = get_tcp()
                dist_new = np.linalg.norm(tcp_new[:2] - _GOAL_POS[:2])
                ep_cols += 1

            # Reward identical to ql_umaze.py
            if dist_new < GOAL_THRESH:
                completion_t = round(time.time() - t0, 3)
                reward       = REWARD_GOAL
                ep_ok        = True
                done         = True
            elif collision:
                reward = REWARD_COLLISION
                done   = False
            else:
                progress = (dist_old - dist_new) * REWARD_PROGRESS_SCALE
                reward   = progress + REWARD_STEP_PENALTY
                done     = False

            ep_reward += reward

            buf["obs"].append(obs.copy())
            buf["actions"].append(action_idx)
            buf["log_probs"].append(log_prob.detach())
            buf["rewards"].append(reward)
            buf["values"].append(value.item())
            buf["dones"].append(float(done))

            global_step += 1
            obs      = build_obs(tcp_new, collision)
            prev_col = collision

            # PPO update every ROLLOUT_LEN steps
            if global_step % ROLLOUT_LEN == 0 and len(buf["obs"]) > 0:
                with torch.no_grad():
                    obs_t      = torch.FloatTensor(obs).unsqueeze(0)
                    _, last_v  = ac_net(obs_t)
                    last_val   = last_v.item() if not done else 0.0

                returns, advantages = compute_gae(
                    buf["rewards"], buf["values"], buf["dones"], last_val
                )
                buf["returns"]    = returns
                buf["advantages"] = advantages
                ppo_update(buf, entropy_coef)

                buf = {"obs": [], "actions": [], "log_probs": [],
                       "rewards": [], "values": [], "dones": []}

            if ep_ok:
                break

        else:
            ep_timeout = True

        elapsed = round(time.time() - t0, 3)
        success_history.append(int(ep_ok))
        check_converged(ep)

        save_trial_row(ep + 1, ep_cols,
                       completion_t if ep_ok else elapsed,
                       ep_ok, ep_timeout)

        summaries.append({
            "episode":              ep + 1,
            "ep_reward":            round(ep_reward, 3),
            "collisions":           ep_cols,
            "success":              int(ep_ok),
            "timeout":              int(ep_timeout),
            "steps":                step + 1,
            "epsilon":              float("nan"),
            "completion_time_s":    completion_t,
            "elapsed_s":            elapsed,
            "episodes_to_converge": episodes_to_converge,
        })

        if ep_reward > best_reward:
            best_reward = ep_reward

        if (ep + 1) % SAVE_EVERY == 0:
            recent   = summaries[-SAVE_EVERY:]
            wins     = sum(e["success"] for e in recent)
            avg_cols = np.mean([e["collisions"] for e in recent])
            print(f"Ep {ep+1:5d}/{EPISODES} | "
                  f"ent={entropy_coef:.4f} | "
                  f"reward={ep_reward:8.2f} | "
                  f"cols={ep_cols:3d} | "
                  f"ok={ep_ok} | "
                  f"wins={wins}/{SAVE_EVERY} | "
                  f"avg_cols={avg_cols:.1f} | "
                  f"t={elapsed:.2f}s")

 
#   SAVE
pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
torch.save(ac_net.state_dict(), MODEL_PATH_OUT)

wins = sum(e["success"] for e in summaries)
ct   = [e["completion_time_s"] for e in summaries
        if e["success"] and not np.isnan(e["completion_time_s"])]
ac_s = [e["collisions"] for e in summaries if e["success"]]

print(f"\n  PPO U-maze v2 -- Training Complete")
print(f"  Trial CSV           : {TRIAL_CSV}")
print(f"  Episode summary     : {SUMMARY_PATH}")
print(f"  Model weights       : {MODEL_PATH_OUT}")
print(f"  Total successes     : {wins}/{EPISODES}  ({100*wins/EPISODES:.1f}%)")
if wins > 0:
    print(f"  Mean completion     : {np.nanmean(ct):.2f}s +/- {np.nanstd(ct):.2f}s")
    print(f"  Mean collisions     : {np.mean(ac_s):.1f}")
print(f"  Episodes to converge: "
      f"{episodes_to_converge if episodes_to_converge else 'Not converged'}")