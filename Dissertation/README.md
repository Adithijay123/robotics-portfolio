# PPO vs Q-Learning: Autonomous Navigation in Constrained Robotic Environments

**Name:** Adithi Jayaraman · **Student ID:** 2306514 · **Module:** COMP303

Comparative benchmarking of PPO and Q-Learning for UR5e arm navigation in MuJoCo simulation across two constrained maze geometries, with a deterministic hardware baseline on a physical UR7e arm instrumented with piezoelectric collision sensing.

---
## Quick Links
| Resource | Link |
|---|---|
| Dissertation (PDF) | `COMP303_2306514_Dissertation.pdf` |
| Development Log | [DevLog](https://www.notion.so/2780f7ecc39f8060a85ad674024fd7f9?v=2780f7ecc39f81c0b533000c046a9f48&source=copy_link) — also available in HTML format inside `Comp303 Dis/` |
| Presentation | [Canva Slides](https://canva.link/lxzgb0n9nt50f3c) |
| Demo Video (U-maze) | `Comp303 Dis/Media/U maze navigation working.mp4` |
| Demo Video (Z-maze) | `Comp303 Dis/Media/Z maze working.mp4` |
---

## Repository Structure

```
2306514_COMP302/
│
├── Comp303 Dis/                  # All dissertation deliverables
├── Development Log HTML
│   ├── CAD/                      # Fusion 360 files — end effector iterations + final
│   ├── Media/                    # Photos, videos, prototype images
│   ├── Mujoco/
│   │   └── Final Experiment/     # All training and testing code
│   │       ├── ppo_umaze.py          PPO agent — U-maze
│   │       ├── ppo_zmaze.py          PPO agent — Z-maze
│   │       ├── Ql_umaze.py           Q-Learning agent — U-maze
│   │       ├── Ql_zmaze.py           Q-Learning agent — Z-maze
│   │       ├── u_logger.py           Hardware trial logger — U-maze
│   │       ├── Z_logger.py           Hardware trial logger — Z-maze
│   │       ├── umaze.xml             MuJoCo U-maze environment
│   │       ├── zmaze.xml             MuJoCo Z-maze environment
│   │       ├── sim_master.csv        Simulation results (80 rows)
│   │       ├── sim_episodes_clean.csv Episode-level data (40,000 rows)
│   │       ├── Unit_test.py          26 unit tests
│   │       ├── Smoke_test.py         11 smoke tests
│   │       └── Integration_test.py   26 integration tests
│   ├── Testing/                  # Test evidence screenshots
│   ├── U_Maze_Data/              # Raw hardware trial CSVs — U-maze
│   ├── Z_Maze_Data/              # Raw hardware trial CSVs — Z-maze
│   ├── analysis.R                # Full statistical analysis (all hypotheses)
│   ├── graphs.R                  # Figure generation
│   ├── hardware_master.csv       # Hardware results (40 rows)
│   └── sim_master.csv            # Simulation results (80 rows)
│
├── Hardware_Pio_collision_logger/ # Arduino Nano piezo sensor firmware
└── IGNORE/                        # Initial errored sim trials and other simulation work (archived)
```

---

## Setup

### Prerequisites

| Tool | Version | Notes |
|---|---|---|
| Python | 3.11 | Use venv311 |
| MuJoCo | 3.35 | `pip install mujoco` |
| PyTorch | - | CPU build sufficient |
| R | 4.3 | For statistical analysis |
| Arduino IDE | Any | For piezo firmware only |

### Python Environment

```bash
# Create and activate virtual environment
python -m venv venv311
venv311\Scripts\activate           # Windows

# Install dependencies
pip install mujoco torch numpy pandas matplotlib
```

### R Dependencies

```r
install.packages(c("readr", "dplyr", "effsize"))
```

---

## Running the Experiments

All commands run from inside `Comp303 Dis/Mujoco/Final Experiment/` with venv311 activated.

### Simulation Training

```bash
# Q-Learning
python Ql_umaze.py       # U-maze — converges around episode 190
python Ql_zmaze.py       # Z-maze

# PPO
python ppo_umaze.py      # U-maze — 500 episodes
python ppo_zmaze.py      # Z-maze — 500 episodes
```

Each run produces a trial CSV and episode summary CSV in the same folder. Results are aggregated into `sim_master.csv` via `data_prep.py`.

### Hardware Baseline (UR7e)

Requires physical UR7e arm, LEGO maze fixture, and Arduino Nano with piezo sensor connected.

```bash
python u_logger.py       # U-maze — 20 trials
python Z_logger.py       # Z-maze — 20 trials
```

Results log to `U_Maze_Data/` and `Z_Maze_Data/` respectively.

### Statistical Analysis

Open `analysis.R` in RStudio and set the working directory to `Comp303 Dis/Mujoco/Final Experiment/`. Run the full script. Requires the three master CSVs to be present:

```
sim_master.csv             (80 rows — one per simulation run)
hardware_master.csv        (40 rows — one per hardware trial)
sim_episodes_clean.csv     (40,000 rows — one per episode)
```

---

## Running the Tests

```bash
# From Comp303 Dis/Mujoco/Final Experiment/ with venv311 activated

python Unit_test.py          # 26 tests — reward logic, obs vector, network shapes, Bellman update
python Smoke_test.py         # 11 tests — Q-Learning and PPO train without errors 
python Integration_test.py   # 26 tests — CSV logging, model save/reload 
```

---

## Hardware

The physical trials use a Universal Robots UR7e controlled via RTDE at 125 Hz. A custom 3D-printed end effector (CAD files in `Comp303 Dis/CAD/`) mounts a ceramic piezoelectric vibration sensor connected to an Arduino Nano. 
The Arduino firmware is in `Hardware_Pio_collision_logger/` and logs collision events in the format:

```
COLLISION #N | Time: X.XXs | Raw ADC: XXX
```

Collision threshold is ADC = 50, calibrated empirically. Baseline ADC at rest is 0–2.

Physical maze is constructed from standard LEGO bricks on a fixed baseplate. Two geometries:

- U-maze — single turn, 4 waypoints, 90s timeout
- Z-maze — three turns, 7 waypoints, 120s timeout

Waypoint joint coordinates are hard-coded in `u_logger.py` and `Z_logger.py` and documented in full in Appendix E of the dissertation.

---

## Key Results

| Algorithm | Maze | Success Rate | Mean Collisions | Converged |
|---|---|---|---|---|
| Q-Learning | U-maze | 78.2% | 37.19 | Episode 190.5 |
| PPO | U-maze | 1.1% | 134.05 | Never |
| Q-Learning | Z-maze | 0.3% | 50.56 | Never |
| PPO | Z-maze | 0.1% | 116.33 | Never |
| Hardcoded (UR7e) | U-maze | 100% | 3.15 | — |
| Hardcoded (UR7e) | Z-maze | 100% | 2.85 | — |

Q-Learning outperformed PPO on every metric within the 500-episode budget. Neither algorithm converged on the Z-maze. Full statistical analysis (MANOVA, Mann-Whitney U, Pearson's r, Cohen's d) is in `analysis.R` and Appendix A of the dissertation.

---
