"""
ql_zmaze.py
Q-Learning agent navigating UR5e through Z-maze in MuJoCo with a probe
to substitute for the piezoelectric sensor.

Uses the current zmaze.xml provided by the user:
- wall names: z_top_w1 ... z_bot_cap
- start site : [-0.130, -0.290, 0.244]
- goal site  : [-0.045, -0.430, 0.244]

"""

import mujoco
import mujoco.viewer     
import numpy as np
import pandas as pd
import pickle
import time
import os
from datetime import datetime


#  PATHS
XML_PATH   = "zmaze.xml"
OUTPUT_DIR = "Z_sim_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# HYPERPARAMETERS 
EPISODES            = 500
MAX_STEPS           = 500
STEP_SIZE           = 0.005
GOAL_THRESH         = 0.012
ALPHA               = 0.20
GAMMA               = 0.95
EPSILON_START       = 1.0
EPSILON_END         = 0.05
EPSILON_DECAY       = 0.9965
N_BINS              = 20
RENDER_EVERY        = 1
RENDER_SPEED        = 0.01
SAVE_EVERY          = 100
CONVERGE_WINDOW     = 100
CONVERGE_THRESHOLD  = 0.80


#  START / GOAL
HOME_QPOS   = np.deg2rad([-93.84, -73.27, -141.51, -55.21, 89.94, -3.84])
HOME_TARGET = np.array([-0.130, -0.290, 0.244])
GOAL_TARGET = np.array([-0.045, -0.430, 0.244])


#  LOAD MODEL 
print(f"Loading model: {os.path.abspath(XML_PATH)}")
model    = mujoco.MjModel.from_xml_path(XML_PATH)
data     = mujoco.MjData(model)
N_JOINTS = 6

_PROBE_TIP_ID  = model.site("probe_tip").id
_PROBE_GEOM_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe")


_WALL_NAMES = [
    "z_top_w1", "z_top_w2", "z_top_cap",
    "z_rd_rw",
    "z_mid_w1",
    "z_ld_lw", "z_ld_rw",
    "z_s2h_w1", "z_s2h_w2",
    "z_s3v_lw", "z_s3v_rw",
    "z_s3h_w1", "z_s3h_w2",
    "z_s4v_lw", "z_s4v_rw",
    "z_bot_w1", "z_bot_w2", "z_bot_cap",
]

_WALL_IDS = set()
for name in _WALL_NAMES:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid >= 0:
        _WALL_IDS.add(gid)
    else:
        print(f"  WARNING: geom '{name}' not found")

print(f"  Wall geoms found : {len(_WALL_IDS)}")
print(f"  Probe geom ID    : {_PROBE_GEOM_ID}")


# HELPERS 
def get_tcp():
    return data.site_xpos[_PROBE_TIP_ID].copy()


def check_collision():
    for i in range(data.ncon):
        c = data.contact[i]
        if (c.geom1 == _PROBE_GEOM_ID and c.geom2 in _WALL_IDS) or \
           (c.geom2 == _PROBE_GEOM_ID and c.geom1 in _WALL_IDS):
            return True
    return False


def move_tip_to(target_xyz, max_iter=25, tol=0.0008):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))

    for _ in range(max_iter):
        mujoco.mj_forward(model, data)
        err = target_xyz - data.site_xpos[_PROBE_TIP_ID]

        if np.linalg.norm(err) < tol:
            return True

        mujoco.mj_jacSite(model, data, jacp, jacr, _PROBE_TIP_ID)
        J = jacp[:, :N_JOINTS]
        dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(3), err)

        data.qpos[:N_JOINTS] = np.clip(
            data.qpos[:N_JOINTS] + dq * 0.35,
            model.jnt_range[:N_JOINTS, 0],
            model.jnt_range[:N_JOINTS, 1],
        )
        data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
        mujoco.mj_step(model, data)

    mujoco.mj_forward(model, data)
    return False


def execute_action_safely(delta_xy, substep=0.001):
    start_qpos = data.qpos[:N_JOINTS].copy()
    start_tcp  = get_tcp().copy()

    total_dist = float(np.linalg.norm(delta_xy))
    n_substeps = max(1, int(np.ceil(total_dist / substep)))

    for k in range(1, n_substeps + 1):
        alpha = k / n_substeps
        target = np.array([
            start_tcp[0] + alpha * delta_xy[0],
            start_tcp[1] + alpha * delta_xy[1],
            LOCKED_Z
        ])

        ok = move_tip_to(target, max_iter=12, tol=0.0008)
        data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
        mujoco.mj_step(model, data)

        if (not ok) or check_collision():
            data.qpos[:N_JOINTS] = start_qpos
            data.ctrl[:N_JOINTS] = start_qpos
            mujoco.mj_forward(model, data)
            return False, True

    return True, False


def reset_env():
    mujoco.mj_resetData(model, data)
    data.qpos[:N_JOINTS] = HOME_QPOS
    data.qvel[:N_JOINTS] = 0.0
    data.ctrl[:N_JOINTS] = HOME_QPOS
    mujoco.mj_forward(model, data)
    move_tip_to(HOME_TARGET, max_iter=60, tol=0.001)
    data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
    mujoco.mj_forward(model, data)


#  STARTUP 
reset_env()
mujoco.mj_forward(model, data)
_HOME_TCP = get_tcp().copy()
LOCKED_Z  = _HOME_TCP[2]
_GOAL_POS = GOAL_TARGET.copy()

# BOUNDING BOX — matched to your actual z_* maze geometry
X_MIN, X_MAX = -0.152, -0.018
Y_MIN, Y_MAX = -0.448, -0.272
X_SPAN = X_MAX - X_MIN
Y_SPAN = Y_MAX - Y_MIN

print(f"\n  Home TCP         : {_HOME_TCP}")
print(f"  Goal pos         : {_GOAL_POS}")
print(f"  Locked Z         : {LOCKED_Z:.4f} m")
print(f"  X span           : {X_SPAN:.3f} m | Y span: {Y_SPAN:.3f} m")
print(f"  Contacts at rest : {data.ncon}  (should be 0)\n")


# STATE DISCRETISATION 
def discretise(tcp, prev_col):
    xb = int(np.clip((tcp[0] - X_MIN) / X_SPAN, 0.0, 0.9999) * N_BINS)
    yb = int(np.clip((tcp[1] - Y_MIN) / Y_SPAN, 0.0, 0.9999) * N_BINS)
    return (xb, yb, int(prev_col))


#  Q-TABLE 
Q = {}

def get_q(state):
    if state not in Q:
        Q[state] = np.zeros(8)
    return Q[state]


#  ACTIONS 
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


#  OUTPUT 
existing = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("zsimql_trial")]
run_num  = len(existing) // 3 + 1
TRIAL_CSV    = os.path.join(OUTPUT_DIR, f"zsimql_trial{run_num}.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, f"zsimql_trial{run_num}_summary.csv")
QTABLE_PATH  = os.path.join(OUTPUT_DIR, f"zsimql_trial{run_num}_qtable.pkl")
summaries    = []
_trial_hdr   = False


def save_trial_row(episode, collisions, completion_time_s, goal_reached, timed_out):
    global _trial_hdr
    row = pd.DataFrame([{
        "trial":             episode,
        "timestamp":         datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm":         "Q-Learning",
        "maze":              "Z-maze",
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


# TRAINING LOOP
print(f"Q-Learning | Z-maze | {EPISODES} eps | step={STEP_SIZE*1000:.0f}mm | bins={N_BINS}x{N_BINS}")
print(f"Rendering  : ENABLED (every {RENDER_EVERY} episodes)")
print(f"Output     : {OUTPUT_DIR}\n")

epsilon     = EPSILON_START
best_reward = -np.inf

with mujoco.viewer.launch_passive(model, data) as viewer:

    for ep in range(EPISODES):

        if not viewer.is_running():
            print("Viewer closed.")
            break

        reset_env()

        tcp      = get_tcp()
        prev_col = False
        state    = discretise(tcp, prev_col)

        ep_reward    = 0.0
        ep_cols      = 0
        ep_ok        = False
        ep_timeout   = False
        t0           = time.time()
        completion_t = float("nan")

        do_render = ((ep + 1) % RENDER_EVERY == 0)

        for step in range(MAX_STEPS):

            if np.random.rand() < epsilon:
                action_idx = np.random.randint(N_ACTIONS)
            else:
                action_idx = int(np.argmax(get_q(state)))

            cur = get_tcp()

            delta_xy = ACTIONS[action_idx]
            moved, collision = execute_action_safely(delta_xy, substep=0.001)

            data.ctrl[:N_JOINTS] = data.qpos[:N_JOINTS]
            mujoco.mj_step(model, data)

            if do_render:
                viewer.sync()
                time.sleep(RENDER_SPEED)

            tcp_new  = get_tcp()
            dist_new = np.linalg.norm(tcp_new[:2] - _GOAL_POS[:2])
            dist_old = np.linalg.norm(cur[:2]     - _GOAL_POS[:2])

            if collision:
                ep_cols += 1

            if dist_new < GOAL_THRESH:
                completion_t = round(time.time() - t0, 3)
                reward       = 500.0
                ep_ok        = True
            elif collision:
                reward = -8.0
            else:
                progress = (dist_old - dist_new) * 400.0
                reward   = progress - 0.05

            ep_reward += reward

            next_state = discretise(tcp_new, collision)
            td_target  = reward + GAMMA * float(np.max(get_q(next_state)))
            get_q(state)[action_idx] += ALPHA * (td_target - get_q(state)[action_idx])
            state, prev_col = next_state, collision

            if ep_ok:
                break

        else:
            ep_timeout = True

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        elapsed = round(time.time() - t0, 3)

        success_history.append(int(ep_ok))
        check_converged(ep)

        save_trial_row(
            ep + 1,
            ep_cols,
            completion_t if ep_ok else elapsed,
            ep_ok,
            ep_timeout
        )

        summaries.append({
            "episode":              ep + 1,
            "ep_reward":            round(ep_reward, 3),
            "collisions":           ep_cols,
            "success":              int(ep_ok),
            "timeout":              int(ep_timeout),
            "steps":                step + 1,
            "epsilon":              round(epsilon, 4),
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
                  f"ε={epsilon:.3f} | "
                  f"reward={ep_reward:8.2f} | "
                  f"cols={ep_cols:3d} | "
                  f"ok={ep_ok} | "
                  f"wins={wins}/{SAVE_EVERY} | "
                  f"avg_cols={avg_cols:.1f} | "
                  f"t={elapsed:.2f}s")


# SAVE
pd.DataFrame(summaries).to_csv(SUMMARY_PATH, index=False)
with open(QTABLE_PATH, "wb") as f:
    pickle.dump(Q, f)

wins = sum(e["success"] for e in summaries)
ct   = [e["completion_time_s"] for e in summaries
        if e["success"] and not np.isnan(e["completion_time_s"])]
ac   = [e["collisions"] for e in summaries if e["success"]]

print(f"\n  Q-Learning Z-maze — Training Complete")
print(f"  Trial CSV           : {TRIAL_CSV}")
print(f"  Episode summary     : {SUMMARY_PATH}")
print(f"  Q-table states      : {len(Q)}")
print(f"  Total successes     : {wins}/{EPISODES}  ({100*wins/EPISODES:.1f}%)")
if wins > 0:
    print(f"  Mean completion     : {np.nanmean(ct):.2f}s ± {np.nanstd(ct):.2f}s")
    print(f"  Mean collisions     : {np.mean(ac):.1f}")
print(f"  Episodes to converge: "
      f"{episodes_to_converge if episodes_to_converge else 'Not converged'}")