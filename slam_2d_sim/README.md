# 2D Occupancy Grid SLAM Simulation

A self-contained Python simulation illustrating the core SLAM problem: a
simulated robot with a lidar sensor explores a 2D room, and two occupancy
grid maps are built in parallel, one using ground-truth pose, one using
only noisy odometry, to show exactly how pose drift corrupts mapping.

No ROS2, no hardware, no external dependencies beyond NumPy/Matplotlib

## Why this demo

Real SLAM (gmapping, cartographer, ORB-SLAM, etc.) exists to solve a
chicken-and-egg problem: you need an accurate map to localise well, but you
need accurate localisation to build a good map. This project isolates and
visualises that exact problem before tackling a full solution:

- `slam_2d_sim/robot.py` — simulates a robot with **noisy odometry**, so its
  own belief of its pose drifts from ground truth over time (exactly as a
  real robot's wheel encoders / IMU would drift).
- `slam_2d_sim/world.py` — a simple 2D room made of line-segment walls, with
  a simulated lidar that raycasts against them (stand-in for a real
  `/scan` topic).
- `slam_2d_sim/mapping.py` — log-odds **occupancy grid mapping**, the same
  core technique used in real SLAM stacks: each lidar ray marks free space
  along its path and an occupied cell at its endpoint, accumulated as
  log-odds and converted to probabilities.
- `demo.py` — drives the robot around the room via a simple waypoint
  controller, builds both maps simultaneously, and saves comparison figures.

## Setup

```bash
pip install numpy matplotlib
```

## Run

```bash
python demo.py
```

Outputs (in `output/`):
- `trajectory.png` — ground-truth path vs. the robot's drifted odometry estimate
- `final_maps_comparison.png` — the same environment mapped two ways:
  cleanly from ground-truth pose, and visibly smeared/doubled from raw
  odometry alone

## Example output

**Trajectory drift:**
Ground truth (solid green) vs. odometry estimate (dashed red) — the two
diverge steadily as small per-step noise accumulates, with no correction.

**Mapping quality:**
The left map (ground-truth pose) shows sharp, single-pixel-wide walls. The
right map (raw odometry pose) shows the same walls smeared and doubled,
because every lidar scan gets stamped into the map at a slightly wrong
pose. This is precisely the error that real SLAM back-ends (scan matching,
particle filters, pose-graph optimisation) exist to correct.

## Project layout

```
slam_2d_sim/
├── slam_2d_sim/
│   ├── world.py      # simulated environment + lidar raycasting
│   ├── robot.py       # simulated robot with noisy odometry + waypoint controller
│   └── mapping.py     # log-odds occupancy grid mapping
├── demo.py             # runs the simulation, saves comparison figures
└── output/             # generated figures (created on run)
```

## Natural next steps (not implemented here, but the obvious extensions)

- **Scan matching / ICP**: correct the odometry estimate by aligning
  consecutive lidar scans against the existing map, rather than trusting
  raw odometry blindly.
- **Particle filter (FastSLAM-style)**: maintain multiple pose hypotheses
  weighted by how well their implied map matches new scans.
- **Pose-graph optimisation**: treat each pose as a graph node, add
  constraints from odometry and loop closures, and optimise the whole
  trajectory globally (this is what Cartographer / g2o-based systems do).
- **Swap the simulated world for a real robot**: replace `World2D.lidar_scan`
  with a real `/scan` subscriber and `Robot2D`'s odometry with a real
  `/odom` subscriber — the mapping code in `mapping.py` doesn't need to
  change at all.
