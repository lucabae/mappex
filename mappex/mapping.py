"""Accumulate observations into a 2D object map and draw it with OpenCV."""

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from mappex import scene

SCALE = 60.0
EXTENT = 6.0
MERGE_DISTANCE = 1.0
MIN_VIEWS = 30
DRAW_SEPARATION = 1.0

BACKGROUND = (30, 30, 30)
GRID = (52, 52, 52)
AXIS = (85, 85, 85)
LABEL = (110, 110, 110)
WALL = (140, 90, 60)
TRAIL = (150, 150, 60)
OBJECT = (60, 220, 60)
TEXT = (235, 235, 235)
DRONE = (0, 200, 255)


@dataclass
class Cluster:
    """A single observed object, refined as more views arrive."""

    x: float
    y: float
    views: int = 1

    def merge(self, x, y):
        total = self.views + 1
        self.x = (self.x * self.views + x) / total
        self.y = (self.y * self.views + y) / total
        self.views = total


def _overlaps(left, baseline, width, height, other):
    """Whether a text box collides with an already placed one."""
    ox, oy, ow, oh = other
    return (
        left < ox + ow + 4
        and left + width + 4 > ox
        and baseline - height < oy + 2
        and baseline + 2 > oy - oh
    )


@dataclass
class ObjectMap:
    """Objects seen so far, grouped by class and position."""

    merge_distance: float = MERGE_DISTANCE
    min_views: int = MIN_VIEWS
    draw_separation: float = DRAW_SEPARATION
    clusters: dict = field(default_factory=dict)

    def observe(self, class_name, x, y):
        """Record a sighting, folding it into the nearest cluster of its class.

        Repeated sightings of the same object are the norm: a single object is
        detected in many frames from slightly different viewpoints. Merging by
        distance keeps exactly one cluster per object instead of one per frame.
        """
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        bucket = self.clusters.setdefault(class_name, [])
        nearest = None
        best = self.merge_distance
        for cluster in bucket:
            distance = math.hypot(cluster.x - x, cluster.y - y)
            if distance <= best:
                nearest, best = cluster, distance
        if nearest is None:
            bucket.append(Cluster(x, y))
        else:
            nearest.merge(x, y)

    def visible(self):
        """Clusters worth showing, strongest first and de-duplicated by class.

        Two clusters of the same class that sit within ``draw_separation`` are
        almost certainly the same object seen from noisy angles; only the one
        with the most evidence is kept, which also stops a stray cluster from
        painting a second label next to a well-established object.
        """
        placed = []
        for class_name, bucket in sorted(self.clusters.items()):
            ranked = sorted(bucket, key=lambda c: c.views, reverse=True)
            accepted = []
            for cluster in ranked:
                if cluster.views < self.min_views:
                    continue
                if any(
                    math.hypot(cluster.x - other.x, cluster.y - other.y)
                    <= self.draw_separation
                    for other in accepted
                ):
                    continue
                accepted.append(cluster)
            placed.extend((class_name, c) for c in accepted)
        return placed

    def summary(self):
        """Human-readable lines, one per drawn object."""
        return [
            f"{name}: x={cluster.x:.2f} y={cluster.y:.2f} ({cluster.views} views)"
            for name, cluster in self.visible()
        ]

    def render(self, pose=None, path=None):
        """Draw the map: walls, objects with coordinates, trail and drone."""
        size = int(round(2 * EXTENT * SCALE)) + 1
        image = np.full((size, size, 3), BACKGROUND, np.uint8)
        self._draw_grid(image, size)
        for half in scene.WALL_HALVES:
            cv2.rectangle(
                image,
                self._pixel(-half, half, size),
                self._pixel(half, -half, size),
                WALL,
                2,
            )
        self._draw_objects(image, size)
        if path:
            self._draw_trail(image, path, size)
        if pose is not None:
            self._draw_drone(image, pose, size)
        cv2.putText(
            image, "x right, y up, grid 1 m", (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1,
        )
        return image

    def save(self, destination, pose=None, path=None):
        image = self.render(pose=pose, path=path)
        cv2.imwrite(str(destination), image)
        return image

    # -- drawing helpers ----------------------------------------------

    @staticmethod
    def _pixel(x, y, size):
        """World metres to image pixels; ``y`` grows upwards on screen."""
        centre = size // 2
        return int(round(centre + x * SCALE)), int(round(centre - y * SCALE))

    def _draw_grid(self, image, size):
        limit = int(EXTENT)
        for metres in range(-limit + 1, limit):
            offset = int(round(size // 2 + metres * SCALE))
            colour = AXIS if metres == 0 else GRID
            cv2.line(image, (offset, 0), (offset, size - 1), colour, 2 if metres == 0 else 1)
            cv2.line(image, (0, offset), (size - 1, offset), colour, 2 if metres == 0 else 1)
            if metres % 2 == 0:
                cv2.putText(image, str(metres), (offset + 2, size - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, LABEL, 1)
                cv2.putText(image, str(-metres), (4, offset - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, LABEL, 1)

    def _draw_objects(self, image, size):
        placed = []
        for class_name, cluster in self.visible():
            px, py = self._pixel(cluster.x, cluster.y, size)
            cv2.circle(image, (px, py), 4, OBJECT, -1)

            label = f"{class_name} {cluster.x:.2f}, {cluster.y:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_COMPLEX, 0.42, 1
            )
            tx = px + 7
            if tx + text_width > size - 4:
                tx = max(px - 7 - text_width, 2)
            ty = py - 6
            while any(
                _overlaps(tx, ty, text_width, text_height, box) for box in placed
            ):
                ty += text_height + 2
            placed.append((tx, ty, text_width, text_height))
            cv2.putText(image, label, (tx, ty),
                        cv2.FONT_HERSHEY_COMPLEX, 0.42, TEXT, 1)

    def _draw_trail(self, image, path, size):
        points = np.array([self._pixel(x, y, size) for x, y in path], np.int32)
        if len(points) > 1:
            cv2.polylines(image, [points], False, TRAIL, 1)

    def _draw_drone(self, image, pose, size):
        px, py = self._pixel(pose.position[0], pose.position[1], size)
        heading = math.atan2(-pose.rotation[1, 2], -pose.rotation[0, 2])
        tip = (px + 12 * math.cos(heading), py - 12 * math.sin(heading))
        left = (px + 7 * math.cos(heading + 2.5), py - 7 * math.sin(heading + 2.5))
        right = (px + 7 * math.cos(heading - 2.5), py - 7 * math.sin(heading - 2.5))
        cv2.fillPoly(image, [np.array([tip, left, right], np.int32)], DRONE)
