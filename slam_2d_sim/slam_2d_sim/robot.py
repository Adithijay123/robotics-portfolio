"""
A simple differential-drive-style robot moving through World2D, with noisy
odometry (so the robot's belief of its own pose drifts from ground truth). 
Motion is driven by a simple waypoint follower
so the demo runs unattended.
"""

import numpy as np


class Robot2D:
    def __init__(self, x: float = 0.0, y: float = 0.0, heading: float = 0.0,
                 odom_noise_std: float = 0.01, heading_noise_std: float = 0.01):
        # Ground truth pose (what the world "actually" is used only to
        # generate sensor readings, never given directly to the mapper).
        self.true_x = x
        self.true_y = y
        self.true_heading = heading

        # Estimated pose from (noisy) odometry — this is what SLAM has to
        # work with, and what drifts over time without correction.
        self.est_x = x
        self.est_y = y
        self.est_heading = heading

        self.odom_noise_std = odom_noise_std
        self.heading_noise_std = heading_noise_std

    def move(self, forward: float, turn: float):
        """Move forward by `forward` metres then turn by `turn` radians."""
        # Ground truth update (exact)
        self.true_heading += turn
        self.true_x += forward * np.cos(self.true_heading)
        self.true_y += forward * np.sin(self.true_heading)

        # Noisy odometry update — this is the robot's own (imperfect) belief
        noisy_forward = forward + np.random.normal(0, self.odom_noise_std)
        noisy_turn = turn + np.random.normal(0, self.heading_noise_std)
        self.est_heading += noisy_turn
        self.est_x += noisy_forward * np.cos(self.est_heading)
        self.est_y += noisy_forward * np.sin(self.est_heading)

    def true_pose(self):
        return self.true_x, self.true_y, self.true_heading

    def estimated_pose(self):
        return self.est_x, self.est_y, self.est_heading


def waypoint_controller(robot: Robot2D, waypoints: list[tuple[float, float]],
                         step_size: float = 0.15, angle_tol: float = 0.1):
    """
    Generator yielding (forward, turn) commands to drive `robot` through a
    list of waypoints in sequence, using ground truth pose (a real system
    would use the SLAM pose estimate + a proper controller,this is a
    stand-in kept deliberately simple so the demo is self-contained).
    """
    for wx, wy in waypoints:
        while True:
            dx = wx - robot.true_x
            dy = wy - robot.true_y
            dist = np.hypot(dx, dy)
            if dist < step_size:
                break

            target_heading = np.arctan2(dy, dx)
            heading_err = np.arctan2(
                np.sin(target_heading - robot.true_heading),
                np.cos(target_heading - robot.true_heading),
            )

            if abs(heading_err) > angle_tol:
                yield (0.0, np.clip(heading_err, -0.2, 0.2))
            else:
                yield (min(step_size, dist), heading_err * 0.3)
