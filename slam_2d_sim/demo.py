"""
Runs a simulated robot around a 2D room, building an occupancy grid map
two ways in parallel:

  1. Using ground-truth pose at each step ("mapping with known poses") —
     produces a clean map.
  2. Using only noisy odometry pose estimates — produces a map that visibly
     smears and doubles up walls as drift accumulates.
Usage:
    pip install numpy matplotlib
    python demo.py
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from slam_2d_sim.world import World2D
from slam_2d_sim.robot import Robot2D, waypoint_controller
from slam_2d_sim.mapping import OccupancyGridMap


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def run_simulation():
    np.random.seed(7)

    world = World2D.default_room()
    robot = Robot2D(x=-4.0, y=-4.0, heading=0.0, odom_noise_std=0.02, heading_noise_std=0.015)

    map_true_pose = OccupancyGridMap(world_size=world.size, resolution=0.05)
    map_odom_pose = OccupancyGridMap(world_size=world.size, resolution=0.05)

    waypoints = [(-4, 3), (0, 4), (3.5, 3.5), (3.5, -2.5), (-3, -4), (-4, -4)]

    true_traj, est_traj = [], []

    print("Simulating robot traversal and building occupancy maps")
    for i, (fwd, turn) in enumerate(waypoint_controller(robot, waypoints)):
        robot.move(fwd, turn)
        true_traj.append(robot.true_pose()[:2])
        est_traj.append(robot.estimated_pose()[:2])

        if i % 4 == 0:  # scan every few steps, like a real lidar rate vs motion rate
            tx, ty, th = robot.true_pose()
            scan = world.lidar_scan(tx, ty, th, num_beams=180)
            map_true_pose.integrate_scan(tx, ty, scan)

            ex, ey, eh = robot.estimated_pose()
            # Same physical scan (sensor doesn't know it's "wrong"), but
            # integrated at the drifted pose estimate instead.
            map_odom_pose.integrate_scan(ex, ey, scan)

    print(f"Done: {len(true_traj)} motion steps, "
          f"{len(true_traj) // 4} lidar scans integrated into each map.")
    return world, np.array(true_traj), np.array(est_traj), map_true_pose, map_odom_pose


def save_figures(world, true_traj, est_traj, map_true_pose, map_odom_pose):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    extent = [-world.size / 2, world.size / 2, -world.size / 2, world.size / 2]

    # --- Trajectory comparison ---
    fig, ax = plt.subplots(figsize=(6, 6))
    for x1, y1, x2, y2 in world.walls:
        ax.plot([x1, x2], [y1, y2], color="black", linewidth=2)
    ax.plot(true_traj[:, 0], true_traj[:, 1], label="Ground truth path", color="tab:green")
    ax.plot(est_traj[:, 0], est_traj[:, 1], label="Odometry estimate (drifted)", color="tab:red", linestyle="--")
    ax.set_title("Ground truth vs. drifted odometry trajectory")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "trajectory.png"), dpi=150)
    plt.close(fig)

    # --- Map comparison: mapping with known (true) pose vs. raw odometry ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(map_true_pose.probability_map(), origin="lower", cmap="Greys", extent=extent, vmin=0, vmax=1)
    axes[0].set_title("Mapped using ground-truth pose\n(clean map)")
    axes[0].set_aspect("equal")

    axes[1].imshow(map_odom_pose.probability_map(), origin="lower", cmap="Greys", extent=extent, vmin=0, vmax=1)
    axes[1].set_title("Mapped using raw odometry only\n(drift causes wall smearing/doubling)")
    axes[1].set_aspect("equal")

    fig.suptitle("The SLAM problem: pose error directly corrupts the map", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "final_maps_comparison.png"), dpi=150)
    plt.close(fig)

    print(f"Saved figures to {OUTPUT_DIR}/")


if __name__ == "__main__":
    world, true_traj, est_traj, map_true_pose, map_odom_pose = run_simulation()
    save_figures(world, true_traj, est_traj, map_true_pose, map_odom_pose)
