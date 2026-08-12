"""
A simple 2D world made of line-segment walls, with a simulated lidar that
raycasts against them. Stands in for a real robot's environment + sensor
swap this for real /scan + /odom topics later and the mapping code
downstream doesn't change.
"""

import numpy as np


class World2D:
    def __init__(self, walls: list[tuple[float, float, float, float]], size: float = 10.0):
        """
        walls: list of (x1, y1, x2, y2) line segments forming the environment.
        size: world extends from -size/2 to +size/2 in both axes.
        """
        self.walls = np.array(walls, dtype=float)
        self.size = size

    @staticmethod
    def default_room() -> "World2D":
        """A simple rectangular room with a couple of interior obstacles."""
        s = 5.0
        walls = [
            # outer boundary
            (-s, -s, s, -s),
            (s, -s, s, s),
            (s, s, -s, s),
            (-s, s, -s, -s),
            # interior obstacles
            (-2, -2, -2, 1),
            (-2, 1, 0, 1),
            (1, -3, 1, -1),
            (1, -1, 3, -1),
            (2, 2, 4, 2),
        ]
        return World2D(walls, size=2 * s)

    def raycast(self, x: float, y: float, angle: float, max_range: float = 6.0) -> float:
        """Return distance to nearest wall along `angle` from (x, y), or max_range."""
        dx, dy = np.cos(angle), np.sin(angle)
        best = max_range

        for x1, y1, x2, y2 in self.walls:
            ex, ey = x2 - x1, y2 - y1
            denom = dx * ey - dy * ex
            if abs(denom) < 1e-9:
                continue  # parallel

            t = ((x1 - x) * ey - (y1 - y) * ex) / denom  # distance along ray
            u = ((x1 - x) * dy - (y1 - y) * dx) / denom  # position along segment

            if t > 1e-6 and 0.0 <= u <= 1.0 and t < best:
                best = t

        return best

    def lidar_scan(self, x: float, y: float, heading: float,
                    num_beams: int = 180, fov: float = 2 * np.pi,
                    max_range: float = 6.0, noise_std: float = 0.02) -> np.ndarray:
        """Simulate a full lidar sweep. Returns array of (angle, range) pairs, angle in world frame."""
        angles = heading + np.linspace(-fov / 2, fov / 2, num_beams, endpoint=False)
        ranges = np.array([self.raycast(x, y, a, max_range) for a in angles])
        if noise_std > 0:
            ranges = ranges + np.random.normal(0, noise_std, size=ranges.shape)
            ranges = np.clip(ranges, 0.0, max_range)
        return np.stack([angles, ranges], axis=1)
