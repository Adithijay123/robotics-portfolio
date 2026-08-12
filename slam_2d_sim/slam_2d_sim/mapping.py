"""
Log-odds occupancy grid mapping. Each cell stores a
log-odds value; free space along a lidar ray decreases it, the hit cell at
the end of the ray increases it. Converting log-odds -> probability gives
the occupancy map.

This module only builds the map from a given pose. Combined with Robot2D's odometry drift,
this demo shows the classic SLAM symptom: the map builds correctly if you
map from the true pose, but visibly smears/doubles-up walls if you (as many
naive systems do) map from raw odometry alone.
"""

import numpy as np


LOG_ODDS_FREE = -0.4
LOG_ODDS_OCC = 0.85
LOG_ODDS_MIN = -5.0
LOG_ODDS_MAX = 5.0


class OccupancyGridMap:
    def __init__(self, world_size: float = 10.0, resolution: float = 0.05):
        self.resolution = resolution
        self.world_size = world_size
        self.grid_n = int(world_size / resolution)
        self.log_odds = np.zeros((self.grid_n, self.grid_n), dtype=np.float32)

    def world_to_grid(self, x: float, y: float):
        gx = int((x + self.world_size / 2) / self.resolution)
        gy = int((y + self.world_size / 2) / self.resolution)
        return gx, gy

    def _bresenham(self, x0, y0, x1, y1):
        """Integer grid cells along the line from (x0,y0) to (x1,y1), exclusive of the endpoint."""
        cells = []
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while (x, y) != (x1, y1):
            cells.append((x, y))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    def integrate_scan(self, pose_x: float, pose_y: float, scan: np.ndarray, max_range: float = 6.0):
        """
        scan: array of (angle, range) pairs from World2D.lidar_scan, in world frame.
        Marks free space along each ray and the endpoint as occupied (if within range).
        """
        gx0, gy0 = self.world_to_grid(pose_x, pose_y)

        for angle, r in scan:
            hit = r < max_range - 1e-3  # did the ray actually hit something?
            end_x = pose_x + r * np.cos(angle)
            end_y = pose_y + r * np.sin(angle)
            gx1, gy1 = self.world_to_grid(end_x, end_y)

            if not (0 <= gx1 < self.grid_n and 0 <= gy1 < self.grid_n):
                continue

            for cx, cy in self._bresenham(gx0, gy0, gx1, gy1):
                if 0 <= cx < self.grid_n and 0 <= cy < self.grid_n:
                    self.log_odds[cy, cx] = np.clip(
                        self.log_odds[cy, cx] + LOG_ODDS_FREE, LOG_ODDS_MIN, LOG_ODDS_MAX
                    )

            if hit:
                self.log_odds[gy1, gx1] = np.clip(
                    self.log_odds[gy1, gx1] + LOG_ODDS_OCC, LOG_ODDS_MIN, LOG_ODDS_MAX
                )

    def probability_map(self) -> np.ndarray:
        """Return the map as occupancy probabilities in [0, 1] (0.5 = unknown)."""
        return 1.0 - 1.0 / (1.0 + np.exp(self.log_odds))
