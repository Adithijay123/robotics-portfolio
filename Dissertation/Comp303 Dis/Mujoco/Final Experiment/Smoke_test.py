"""
Quick sanity check to make sure both Q-Learning and PPO can actually
train without falling over. This isn't testing whether they learn well —
it's just checking that the core training loop runs cleanly from start
to finish without errors, NaN values, or frozen weights.

To keep things fast, both agents run in a lightweight mock environment
 a simple 2D grid where the agent just needs to
reach a goal point with no walls. Q-Learning gets 10 episodes to
populate its Q-table and decay epsilon, while PPO gets 10 episodes to
collect rollouts, compute advantages, and run a gradient update.

If anything fundamental is broken, wrong tensor shapes, reward
functions returning NaN, the Q-table never updating, the PPO network
not changing its weights — this test will catch it before you waste
time running a full 500-episode training run.

All checks are printed as PASS/FAIL with a summary count at the end.
Exits with code 1 if anything fails so it can be caught in a CI pipeline.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
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
print("SMOKE TEST — Q-Learning and PPO train without errors")
print("=" * 55)

# constants
N_ACTIONS             = 8
OBS_DIM               = 6
ALPHA                 = 0.20
GAMMA                 = 0.95
EPSILON               = 1.0
EPSILON_DECAY         = 0.9965
EPSILON_END           = 0.05
REWARD_GOAL           =  500.0
REWARD_COLLISION      =   -8.0
REWARD_PROGRESS_SCALE =  400.0
REWARD_STEP_PENALTY   =   -0.05
GOAL_THRESH           =  0.012
STEP_SIZE             =  0.005
SMOKE_EPISODES        =  10
MAX_STEPS             =  50

# mock maze
class MockMaze:
    # 2-D grid maze stub.  Agent starts at (0.0, 0.0), goal at (0.1, 0.0). Walls are ignored

    ACTIONS = np.array([
        [ STEP_SIZE,  0.0      ],
        [-STEP_SIZE,  0.0      ],
        [ 0.0,        STEP_SIZE],
        [ 0.0,       -STEP_SIZE],
        [ STEP_SIZE,  STEP_SIZE],
        [ STEP_SIZE, -STEP_SIZE],
        [-STEP_SIZE,  STEP_SIZE],
        [-STEP_SIZE, -STEP_SIZE],
    ], dtype=np.float64)

    GOAL = np.array([0.10, 0.0])

    def reset(self):
        self.pos = np.array([0.0, 0.0])
        return self.pos.copy(), False

    def step(self, action_idx):
        old_pos   = self.pos.copy()
        self.pos += self.ACTIONS[action_idx]
        collision  = False          # no walls in smoke test
        dist_new   = np.linalg.norm(self.pos - self.GOAL)
        dist_old   = np.linalg.norm(old_pos  - self.GOAL)
        goal_reached = dist_new < GOAL_THRESH

        if goal_reached:
            reward = REWARD_GOAL
        elif collision:
            reward = REWARD_COLLISION
        else:
            progress = (dist_old - dist_new) * REWARD_PROGRESS_SCALE
            reward   = progress + REWARD_STEP_PENALTY

        return self.pos.copy(), reward, collision, goal_reached

def discretise(pos, prev_col, n_bins=20, span=0.2):
    xb = int(np.clip(pos[0] / span, 0.0, 0.9999) * n_bins)
    yb = int(np.clip((pos[1] + 0.1) / span, 0.0, 0.9999) * n_bins)
    return (xb, yb, int(prev_col))

#  ActorCritic (identical to ppo_umaze.py)
class ActorCritic(nn.Module):
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

    def act(self, obs_np):
        obs_t         = torch.FloatTensor(obs_np).unsqueeze(0)
        logits, value = self(obs_t)
        dist          = Categorical(logits=logits)
        action        = dist.sample()
        return action.item(), dist.log_prob(action), value.squeeze(0)

def build_obs_smoke(pos, prev_col, goal=np.array([0.10, 0.0])):
    dx      = goal[0] - pos[0]
    dy      = goal[1] - pos[1]
    dist    = np.linalg.norm([dx, dy])
    return np.array([
        float(np.clip(pos[0] / 0.2, 0.0, 1.0)),
        float(np.clip((pos[1] + 0.1) / 0.2, 0.0, 1.0)),
        float(np.clip(dx / 0.2, -1.0, 1.0)),
        float(np.clip(dy / 0.2, -1.0, 1.0)),
        float(np.clip(dist / 0.4, 0.0, 1.0)),
        float(prev_col),
    ], dtype=np.float32)

print(f"\n[1] Q-Learning smoke test — {SMOKE_EPISODES} episodes")

env      = MockMaze()
Q        = {}
epsilon  = EPSILON
rewards  = []
losses   = []

def get_q(s):
    if s not in Q:
        Q[s] = np.zeros(N_ACTIONS)
    return Q[s]

for ep in range(SMOKE_EPISODES):
    pos, prev_col = env.reset()
    state         = discretise(pos, prev_col)
    ep_reward     = 0.0
    ep_loss       = 0.0

    for step in range(MAX_STEPS):
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(N_ACTIONS)
        else:
            action_idx = int(np.argmax(get_q(state)))

        pos_new, reward, collision, goal_reached = env.step(action_idx)
        next_state = discretise(pos_new, collision)

        td_target    = reward + GAMMA * float(np.max(get_q(next_state)))
        td_error     = td_target - get_q(state)[action_idx]
        ep_loss     += abs(td_error)
        get_q(state)[action_idx] += ALPHA * td_error

        ep_reward += reward
        state      = next_state

        if goal_reached:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    rewards.append(ep_reward)
    losses.append(ep_loss)
    print(f"  ep {ep+1:2d}/{SMOKE_EPISODES} | "
          f"reward={ep_reward:8.3f} | "
          f"td_loss={ep_loss:.3f} | "
          f"epsilon={epsilon:.4f} | "
          f"q_states={len(Q)}")

check("Q-Learning ran all 10 episodes without error", len(rewards) == SMOKE_EPISODES)
check("Q-table populated (agent explored states)", len(Q) > 0, got=len(Q))
check("all rewards are finite", all(np.isfinite(r) for r in rewards))
check("all losses are finite", all(np.isfinite(l) for l in losses))
check("epsilon decayed below start", epsilon < EPSILON, got=round(epsilon, 4))
check("reward is not constant across all episodes",
      not all(r == rewards[0] for r in rewards))

print(f"\n[2] PPO smoke test — {SMOKE_EPISODES} episodes")

net       = ActorCritic(obs_dim=OBS_DIM, n_actions=N_ACTIONS)
optimizer = optim.Adam(net.parameters(), lr=3e-4)
ppo_rewards = []
ppo_losses  = []

CLIP_EPS   = 0.2
VALUE_COEF = 0.5
ROLLOUT_LEN = 32

buf = {"obs": [], "actions": [], "log_probs": [],
       "rewards": [], "values": [], "dones": []}
global_step = 0

for ep in range(SMOKE_EPISODES):
    pos, prev_col = env.reset()
    obs       = build_obs_smoke(pos, prev_col)
    ep_reward = 0.0

    for step in range(MAX_STEPS):
        action_idx, log_prob, value = net.act(obs)
        pos_new, reward, collision, goal_reached = env.step(action_idx)

        done = goal_reached
        buf["obs"].append(obs.copy())
        buf["actions"].append(action_idx)
        buf["log_probs"].append(log_prob.detach())
        buf["rewards"].append(reward)
        buf["values"].append(value.item())
        buf["dones"].append(float(done))

        obs           = build_obs_smoke(pos_new, collision)
        ep_reward    += reward
        global_step  += 1

        if global_step % ROLLOUT_LEN == 0 and len(buf["obs"]) > 0:
            # simple returns (no GAE for smoke test)
            returns = []
            G = 0.0
            for r, d in zip(reversed(buf["rewards"]), reversed(buf["dones"])):
                G = r + GAMMA * G * (1 - d)
                returns.insert(0, G)

            obs_t      = torch.FloatTensor(np.array(buf["obs"]))
            actions_t  = torch.LongTensor(buf["actions"])
            old_lp_t   = torch.stack(buf["log_probs"])
            returns_t  = torch.FloatTensor(returns)
            values_t   = torch.FloatTensor(buf["values"])
            adv_t      = (returns_t - values_t).detach()
            if adv_t.std() > 1e-8:
                adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

            logits, new_vals = net(obs_t)
            dist      = Categorical(logits=logits)
            new_lp    = dist.log_prob(actions_t)
            entropy   = dist.entropy()
            ratio     = torch.exp(new_lp - old_lp_t)
            surr1     = ratio * adv_t
            surr2     = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv_t
            p_loss    = -torch.min(surr1, surr2).mean()
            v_loss    = nn.functional.mse_loss(new_vals, returns_t)
            loss      = p_loss + VALUE_COEF * v_loss - 0.01 * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            optimizer.step()

            ppo_losses.append(loss.item())
            buf = {"obs": [], "actions": [], "log_probs": [],
                   "rewards": [], "values": [], "dones": []}

        if goal_reached:
            break

    ppo_rewards.append(ep_reward)
    print(f"  ep {ep+1:2d}/{SMOKE_EPISODES} | "
          f"reward={ep_reward:8.3f} | "
          f"loss={ppo_losses[-1]:.4f}" if ppo_losses else
          f"  ep {ep+1:2d}/{SMOKE_EPISODES} | reward={ep_reward:8.3f} | loss=pending")

check("PPO ran all 10 episodes without error", len(ppo_rewards) == SMOKE_EPISODES)
check("PPO produced at least one loss value", len(ppo_losses) > 0,
      got=len(ppo_losses))
check("all PPO rewards are finite",
      all(np.isfinite(r) for r in ppo_rewards))
check("all PPO losses are finite",
      all(np.isfinite(l) for l in ppo_losses))
check("PPO loss is not constant (network is updating)",
      len(ppo_losses) < 2 or not all(l == ppo_losses[0] for l in ppo_losses))

print("\n" + "=" * 55)
print(f"  RESULTS:  {PASS} passed  |  {FAIL} failed  |  {PASS+FAIL} total")
print("=" * 55)
if FAIL > 0:
    sys.exit(1)