"""Pinhole camera maths: pixels and depths to points in the world."""

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics of a rendered frame.

    ``fovy_deg`` is the vertical field of view, which is what MuJoCo stores.
    """

    width: int
    height: int
    fovy_deg: float

    @property
    def focal(self):
        """Focal length in pixels, identical on both axes (square pixels)."""
        return (self.height / 2.0) / math.tan(math.radians(self.fovy_deg) / 2.0)

    def rays(self, u, v):
        """Unit-less pixel offsets, divided by depth, in camera coordinates."""
        u = np.atleast_1d(np.asarray(u, dtype=float))
        v = np.atleast_1d(np.asarray(v, dtype=float))
        focal = self.focal
        return np.stack(
            [
                (u - self.width / 2.0) / focal,
                -(v - self.height / 2.0) / focal,
            ],
            axis=-1,
        )

    def to_camera(self, u, v, depth):
        """Camera-frame points for pixels ``(u, v)`` at ``depth`` metres."""
        depth = np.atleast_1d(np.asarray(depth, dtype=float))
        xy = self.rays(u, v) * depth.reshape(-1, 1)
        xyz = np.column_stack([xy, depth])
        return xyz[0] if np.isscalar(u) or np.ndim(u) == 0 else xyz


@dataclass(frozen=True)
class CameraPose:
    """World transform of a camera.

    ``rotation`` holds the camera axes as columns: right, up and back. The
    viewing direction is therefore ``-rotation[:, 2]``.
    """

    position: np.ndarray
    rotation: np.ndarray

    @property
    def right(self):
        return self.rotation[:, 0]

    @property
    def up(self):
        return self.rotation[:, 1]

    @property
    def back(self):
        return self.rotation[:, 2]

    def to_world(self, camera_points):
        """Map camera-frame points (N, 3) into world coordinates."""
        camera_points = np.atleast_2d(np.asarray(camera_points, dtype=float))
        x = camera_points[:, 0:1]
        y = camera_points[:, 1:2]
        z = camera_points[:, 2:3]
        return self.position + x * self.right + y * self.up - z * self.back


def unproject(pixels, depths, intrinsics, pose):
    """World points for pixel coordinates ``(u, v)`` seen at ``depths`` metres."""
    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    return pose.to_world(intrinsics.to_camera(pixels[:, 0], pixels[:, 1], depths))
