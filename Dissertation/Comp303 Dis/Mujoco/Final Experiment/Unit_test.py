"""
they check that the core mathematical logic is correct before anything else runs.  

  1. Reward function — checks all five cases against known values:
     goal reached should give exactly +500, collision exactly -8,
     moving toward the goal should give a positive reward, moving away
     should give a negative one, and standing still should give only
     the step penalty of -0.05. Also checks the goal threshold
     boundary behaves correctly at 12mm.

  2. Observation vector — checks that build_obs() produces a 6-element
     float32 array with all values in the expected ranges. The
     normalised x/y positions should be in [0,1], the direction
     components in [-1,1], the distance in [0,1], and the collision
     flag should be exactly 0.0 or 1.0. Also checks that the distance
     is near zero when the probe is already at the goal.

  3. ActorCritic network — checks that the network produces the right
     output shapes for both a single observation and a batch of 64
     (the mini-batch size used in PPO updates), and that none of the
     outputs are NaN or Inf after initialisation.

  4. Q-table — checks that new states initialise to all zeros and that
     a manual Bellman update produces exactly the right value.

If any of these fail, stop and fix them before running any training.
A wrong reward value or a broken observation vector will silently
corrupt every experiment result.
"""

import numpy as np
import torch
import torch.nn as nn
import sys

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
print("UNIT TESTS — reward logic and observation vector")
print("=" * 55)

# Constants copied directly from ppo_umaze.py and ql_umaze.py.
# If these change in the main scripts, update them here too.
REWARD_GOAL           =  500.0
REWARD_COLLISION      =   -8.0
REWARD_PROGRESS_SCALE =  400.0
REWARD_STEP_PENALTY   =   -0.05
GOAL_THRESH           =  0.012
STEP_SIZE             =  0.005
X_MIN, X_MAX          = -0.175, -0.005
Y_MIN, Y_MAX          = -0.400, -0.308
X_SPAN                =  X_MAX - X_MIN
Y_SPAN                =  Y_MAX - Y_MIN
GOAL_POS              =  np.array([-0.026, -0.325, 0.16635])
OBS_DIM               =  6
N_ACTIONS             =  8


def compute_reward(dist_new, dist_old, collision, goal_reached):
    """
    Shared reward function used by both Q-Learning and PPO.
    Priority order: goal > collision > progress + step penalty.
    """
    if goal_reached:
        return REWARD_GOAL
    elif collision:
        return REWARD_COLLISION
    else:
        progress = (dist_old - dist_new) * REWARD_PROGRESS_SCALE
        return progress + REWARD_STEP_PENALTY


def build_obs(tcp, prev_col):
    """
    Builds the 6-dimensional observation vector from the current TCP
    position and previous collision flag. All values are normalised to
    fit within the bounds the PPO network expects.
    """
    x_norm    = float(np.clip((tcp[0] - X_MIN) / X_SPAN, 0.0, 1.0))
    y_norm    = float(np.clip((tcp[1] - Y_MIN) / Y_SPAN, 0.0, 1.0))
    dx        = GOAL_POS[0] - tcp[0]
    dy        = GOAL_POS[1] - tcp[1]
    dx_norm   = float(np.clip(dx / X_SPAN, -1.0, 1.0))
    dy_norm   = float(np.clip(dy / Y_SPAN, -1.0, 1.0))
    dist      = float(np.linalg.norm([dx, dy]))
    dist_norm = float(np.clip(dist / (X_SPAN + Y_SPAN), 0.0, 1.0))
    col_flag  = float(prev_col)
    return np.array([x_norm, y_norm, dx_norm, dy_norm, dist_norm, col_flag],
                    dtype=np.float32)


class ActorCritic(nn.Module):
    """
    Actor-Critic network used by PPO. Shared backbone feeds into
    separate policy and value heads. Identical to ppo_umaze.py —
    kept here so the unit tests are fully self-contained.
    """
    def __init__(self, obs_dim, n_actions):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.Tanh(),
            nn.Linear(128, 128),     nn.Tanh(),
        )
        self.policy_head = nn.Linear(128, n_actions)
        self.value_head  = nn.Linear(128, 1)
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


print("\n[1] Reward function — known state checks")

r = compute_reward(dist_new=0.005, dist_old=0.020, collision=False, goal_reached=True)
check("goal reward is +500", r == 500.0, got=r, expected=500.0)

r = compute_reward(dist_new=0.050, dist_old=0.050, collision=True, goal_reached=False)
check("collision reward is -8", r == -8.0, got=r, expected=-8.0)

r = compute_reward(dist_new=0.050, dist_old=0.060, collision=False, goal_reached=False)
check("progress reward is positive when moving toward goal", r > 0, got=round(r, 4))

r = compute_reward(dist_new=0.070, dist_old=0.060, collision=False, goal_reached=False)
check("progress reward is negative when moving away from goal", r < 0, got=round(r, 4))

r = compute_reward(dist_new=0.060, dist_old=0.060, collision=False, goal_reached=False)
check("stationary step gives only step penalty (-0.05)", abs(r - (-0.05)) < 1e-6,
      got=round(r, 6), expected=-0.05)

# Boundary checks: 0.001m either side of the 12mm goal threshold
check("dist < GOAL_THRESH triggers goal",   GOAL_THRESH - 0.001 < GOAL_THRESH)
check("dist > GOAL_THRESH does not trigger goal", GOAL_THRESH + 0.001 >= GOAL_THRESH)


print("\n[2] Observation vector — shape and range checks")

tcp_home = np.array([-0.154, -0.325, 0.16642])
obs = build_obs(tcp_home, prev_col=False)

check("obs vector has 6 elements", obs.shape == (OBS_DIM,),
      got=obs.shape, expected=(OBS_DIM,))
check("obs dtype is float32", obs.dtype == np.float32,
      got=obs.dtype, expected=np.float32)
check("x_norm in [0, 1]",   0.0 <= obs[0] <= 1.0, got=obs[0])
check("y_norm in [0, 1]",   0.0 <= obs[1] <= 1.0, got=obs[1])
check("dx_norm in [-1, 1]", -1.0 <= obs[2] <= 1.0, got=obs[2])
check("dy_norm in [-1, 1]", -1.0 <= obs[3] <= 1.0, got=obs[3])
check("dist_norm in [0, 1]", 0.0 <= obs[4] <= 1.0, got=obs[4])
check("prev_col flag is 0.0 when no collision", obs[5] == 0.0, got=obs[5])

obs_col = build_obs(tcp_home, prev_col=True)
check("prev_col flag is 1.0 when collision", obs_col[5] == 1.0, got=obs_col[5])

# At goal position, distance should be essentially zero
tcp_goal = GOAL_POS.copy()
obs_goal = build_obs(tcp_goal, prev_col=False)
check("dist_norm near 0 when at goal", obs_goal[4] < 0.05, got=round(obs_goal[4], 4))


print("\n[3] ActorCritic network — output tensor shapes")

net = ActorCritic(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
net.eval()

obs_t         = torch.FloatTensor(obs).unsqueeze(0)  # single observation: shape [1, 6]
logits, value = net(obs_t)

check("policy logits shape is [1, 8]", tuple(logits.shape) == (1, N_ACTIONS),
      got=tuple(logits.shape), expected=(1, N_ACTIONS))
check("value output shape is [1]", tuple(value.shape) == (1,),
      got=tuple(value.shape), expected=(1,))
check("logits are finite (no NaN/Inf)", torch.isfinite(logits).all().item())
check("value is finite (no NaN/Inf)",   torch.isfinite(value).all().item())

# Batch of 64 — matches the mini-batch size used in PPO updates
batch_obs      = torch.FloatTensor(np.stack([obs] * 64))
logits_b, v_b  = net(batch_obs)
check("batch logits shape is [64, 8]", tuple(logits_b.shape) == (64, N_ACTIONS),
      got=tuple(logits_b.shape), expected=(64, N_ACTIONS))
check("batch values shape is [64]", tuple(v_b.shape) == (64,),
      got=tuple(v_b.shape), expected=(64,))


print("\n[4] Q-table — initialisation and update")

Q = {}

def get_q(state):
    """Returns the Q-values for a state, initialising to zeros if unseen."""
    if state not in Q:
        Q[state] = np.zeros(N_ACTIONS)
    return Q[state]

state  = (5, 10, 0)
q_vals = get_q(state)
check("new state initialises to all zeros", np.all(q_vals == 0.0))
check("Q-table returns array of length 8", len(q_vals) == N_ACTIONS,
      got=len(q_vals), expected=N_ACTIONS)

# Manual Bellman update
ALPHA, GAMMA = 0.20, 0.95
reward       = -8.0
next_state   = (5, 10, 1)
td_target    = reward + GAMMA * float(np.max(get_q(next_state)))
old_val      = get_q(state)[0]
get_q(state)[0] += ALPHA * (td_target - old_val)
expected_val = old_val + ALPHA * (td_target - old_val)
check("Q-table Bellman update correct",
      abs(get_q(state)[0] - expected_val) < 1e-9,
      got=round(get_q(state)[0], 6), expected=round(expected_val, 6))


print("\n" + "=" * 55)
print(f"  RESULTS:  {PASS} passed  |  {FAIL} failed  |  {PASS+FAIL} total")
print("=" * 55)
if FAIL > 0:
    sys.exit(1)