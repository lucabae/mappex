"""Generate a synthetic training set by photographing the scene from everywhere.

Each object body in the corridor gets its own folder of renders plus a
``labels.json`` holding the camera intrinsics and, per image, the projected
bounding box of every object visible in the frame.
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from mappex import scene

SKIP_BODIES = {"world", "x2"}
MAX_RANGE = 8.1
MIN_BOX_PIXELS = 4
MIN_OVERLAP = 0.65
SAMPLES_PER_ANGLE = 12


class ScenePhotographer:
    """Renders every object in the scene from a ring of valid viewpoints."""

    def __init__(self, model, data, width=640, height=480, seed=0):
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self.rng = np.random.default_rng(seed)
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.fovy = math.radians(scene.CAMERA_FOVY)
        self.focal = (height / 2.0) / math.tan(self.fovy / 2.0)
        self._sphere = _sphere_directions(128)
        self._ring = _ring_directions(64)
        self.bodies = self._collect_bodies()

    def close(self):
        self.renderer.close()

    # -- scene inspection ---------------------------------------------

    def _collect_bodies(self):
        bodies = {}
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if not name or name in SKIP_BODIES:
                continue
            geoms = [g for g in range(self.model.ngeom) if self.model.geom_bodyid[g] == body_id]
            if not geoms:
                continue
            centers = np.array([self.data.geom_xpos[g] for g in geoms])
            centre = centers.mean(axis=0)
            radius = max(
                float(np.linalg.norm(self.data.geom_xpos[g] - centre))
                + float(self.model.geom_rbound[g])
                for g in geoms
            )
            bodies[body_id] = {
                "name": name,
                "geoms": geoms,
                "centre": centre,
                "radius": max(radius, 0.05),
                "points": np.concatenate([self._geom_points(g) for g in geoms]),
            }
        return bodies

    def _geom_points(self, geom_id):
        """Surface samples of a geom, used to project its silhouette."""
        position = np.array(self.data.geom_xpos[geom_id], dtype=float)
        rotation = np.array(self.data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        gtype = int(self.model.geom_type[geom_id])
        size = np.array(self.model.geom_size[geom_id], dtype=float)

        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            signs = np.array(
                [[-1, -1, -1], [1, -1, -1], [-1, 1, -1], [1, 1, -1],
                 [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1]], dtype=float
            )
            return position + (signs * size[None, :3]) @ rotation.T
        if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            return position + size[0] * self._sphere
        if gtype == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
            return position + self._sphere * size[None, :3]
        if gtype in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
            ring = (self._ring @ rotation[:, :2].T) * size[0]
            axis = rotation[:, 2] * size[1]
            return np.concatenate([position + ring + axis, position + ring - axis])
        return position + float(self.model.geom_rbound[geom_id]) * self._sphere

    # -- viewpoint search ---------------------------------------------

    def _camera(self, lookat, distance, azimuth, elevation):
        self.camera.lookat[:] = lookat
        self.camera.distance = distance
        self.camera.azimuth = azimuth
        self.camera.elevation = elevation
        az, el = math.radians(azimuth), math.radians(elevation)
        forward = np.array([
            math.cos(el) * math.cos(az),
            math.cos(el) * math.sin(az),
            math.sin(el),
        ])
        up = np.array([
            -math.sin(el) * math.cos(az),
            -math.sin(el) * math.sin(az),
            math.cos(el),
        ])
        return np.asarray(lookat, dtype=float) - distance * forward, forward, up

    def _unoccluded(self, centre, body_id, position):
        delta = centre - position
        distance = float(np.linalg.norm(delta))
        if distance < 1e-6:
            return False
        if self._ray_hits(position, delta / distance, body_id):
            return True
        for geom in self.bodies[body_id]["geoms"]:
            vector = self.data.geom_xpos[geom] - position
            length = float(np.linalg.norm(vector))
            if length > 1e-6 and self._ray_hits(position, vector / length, body_id):
                return True
        return False

    def _ray_hits(self, origin, direction, body_id):
        geom_out = np.zeros(1, dtype=np.int32)
        mujoco.mj_ray(
            self.model, self.data, origin, direction, None, 1, -1, geom_out
        )
        geom = int(geom_out[0])
        return geom >= 0 and int(self.model.geom_bodyid[geom]) == body_id

    def view(self, body_id, azimuth, distance_fraction, elevation_fraction):
        """A rendered-and-projected view of ``body_id``, or ``None`` if unusable."""
        info = self.bodies[body_id]
        centre, radius = info["centre"], info["radius"]
        low = max(0.35, radius * 0.75 + 0.15)
        distance = low + (MAX_RANGE - low) * distance_fraction

        height = scene.HOVER_ALTITUDE + (elevation_fraction - 0.5) * 0.04
        sin_el = np.clip(-(height - centre[2]) / distance, -0.6, 0.35)
        elevation = math.degrees(math.asin(sin_el))

        position, forward, up = self._camera(centre, distance, azimuth, elevation)
        if not _inside_corridor(position):
            return None
        if not self._unoccluded(centre, body_id, position):
            return None

        right = np.cross(forward, up)
        norm = float(np.linalg.norm(right))
        if norm < 1e-6:
            return None
        right /= norm

        box = self._project(info["points"], position, forward, right, up)
        if box is None:
            return None

        offset = centre - position
        depth = float(np.dot(offset, forward))
        x_cam = float(np.dot(offset, right))
        y_cam = float(np.dot(offset, up))
        return {
            "position": position,
            "azimuth": azimuth,
            "elevation": elevation,
            "distance": distance,
            "camera_xyz": (x_cam, y_cam, depth),
            "pixel": (
                0.5 * self.width + self.focal * x_cam / depth,
                0.5 * self.height - self.focal * y_cam / depth,
            ),
            "box": box,
        }

    def _project(self, points, position, forward, right, up):
        """Clipped screen box of ``points``, or ``None`` if it is not usable."""
        offsets = points - position
        depths = offsets @ forward
        if float(depths.min()) <= 0.1:
            return None

        u = 0.5 * self.width + self.focal * (offsets @ right) / depths
        v = 0.5 * self.height - self.focal * (offsets @ up) / depths
        x_min, y_min = float(u.min()), float(v.min())
        box_w, box_h = float(u.max() - x_min), float(v.max() - y_min)
        if box_w < MIN_BOX_PIXELS or box_h < MIN_BOX_PIXELS:
            return None

        clipped = (
            max(0.0, x_min),
            max(0.0, y_min),
            min(float(self.width), x_min + box_w),
            min(float(self.height), y_min + box_h),
        )
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            return None
        overlap = (clipped[2] - clipped[0]) * (clipped[3] - clipped[1]) / (box_w * box_h)
        if overlap < MIN_OVERLAP:
            return None
        return (clipped[0], clipped[1], clipped[2] - clipped[0], clipped[3] - clipped[1])

    def azimuth_groups(self, body_id, groups, trials=10):
        """Pick ``groups`` well-separated azimuths from which the body is visible."""
        scores = {}
        for azimuth in range(0, 360, 3):
            hits = sum(
                self.view(body_id, azimuth, self.rng.uniform(0.05, 0.95), self.rng.uniform())
                is not None
                for _ in range(trials)
            )
            if hits:
                scores[float(azimuth)] = hits

        valid = [a for a, hits in scores.items() if hits >= 0.4 * trials] or list(scores)
        if len(valid) <= groups:
            return sorted(valid)

        chosen = [valid[0]]
        while len(chosen) < groups:
            best = max(valid, key=lambda a: min(_circular_distance(a, c) for c in chosen))
            if min(_circular_distance(best, c) for c in chosen) < 1e-6:
                break
            chosen.append(best)
        return sorted(chosen)

    # -- labelling -----------------------------------------------------

    def visible_objects(self, position, target_id, target_box):
        """Every object visible in the current render, labelled with its box."""
        self.renderer.enable_segmentation_rendering()
        self.renderer.update_scene(self.data, camera=self.camera)
        segmentation = self.renderer.render()
        self.renderer.disable_segmentation_rendering()

        geom_pixels = segmentation[:, :, 0]
        valid = (geom_pixels >= 0) & (geom_pixels < self.model.ngeom)
        body_pixels = np.where(valid, self.model.geom_bodyid[geom_pixels], -1)

        labels = []
        found_target = False
        order = sorted(self.bodies, key=lambda b: (b != target_id, b))
        for body_id in order:
            mask = body_pixels == body_id
            count = int(np.count_nonzero(mask))
            if count == 0:
                continue
            rows, columns = np.where(mask)
            x1, x2 = int(columns.min()), int(columns.max())
            y1, y2 = int(rows.min()), int(rows.max())
            box_w, box_h = x2 - x1 + 1, y2 - y1 + 1
            is_target = body_id == target_id
            if not is_target and (count < 25 or box_w < 4 or box_h < 4):
                continue
            if is_target:
                found_target = True
            labels.append(self._label(body_id, x1, y1, box_w, box_h, count, position))

        if target_id is not None and not found_target:
            x, y, w, h = target_box
            labels.insert(0, self._label(target_id, x, y, w, h, -1, position))
        return labels

    def _label(self, body_id, x, y, w, h, count, position):
        centre = self.bodies[body_id]["centre"]
        return {
            "class": self.bodies[body_id]["name"],
            "body_id": body_id,
            "bbox_xywh": [round(float(v), 3) for v in (x, y, w, h)],
            "bbox_xywh_normalized": [
                round(float(x / self.width), 5),
                round(float(y / self.height), 5),
                round(float(w / self.width), 5),
                round(float(h / self.height), 5),
            ],
            "pixel_count": count,
            "distance_m": round(float(np.linalg.norm(centre - position)), 4),
        }

    # -- dataset -------------------------------------------------------

    def generate(self, body_id, per_class, angles, out_dir):
        groups = self.azimuth_groups(body_id, angles)
        if not groups:
            return 0

        name = self.bodies[body_id]["name"]
        per_angle = math.ceil(per_class / len(groups))
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob(f"{name}_*.png"):
            stale.unlink()

        samples = []
        for group, azimuth_centre in enumerate(groups):
            made = 0
            for _ in range(4000):
                if len(samples) >= per_class or made >= per_angle:
                    break
                sample = self._capture(body_id, group, azimuth_centre, len(samples), out_dir)
                if sample is not None:
                    samples.append(sample)
                    made += 1

        self._write_labels(name, body_id, groups, samples, out_dir)
        return len(samples)

    def _capture(self, body_id, group, azimuth_centre, index, out_dir):
        """Render one accepted view to disk and return its label entry."""
        azimuth = float((azimuth_centre + self.rng.uniform(-6.0, 6.0)) % 360.0)
        view = self.view(body_id, azimuth, self.rng.uniform(), self.rng.uniform())
        if view is None:
            return None

        filename = f"{self.bodies[body_id]['name']}_{index:03d}.png"
        self.renderer.update_scene(self.data, camera=self.camera)
        image = self.renderer.render()
        cv2.imwrite(str(out_dir / filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        labels = self.visible_objects(view["position"], body_id, view["box"])
        return self._sample(filename, group, view, labels, self.bodies[body_id]["centre"])

    def _sample(self, filename, group, view, labels, centre):
        x_cam, y_cam, depth = view["camera_xyz"]
        pixel_x, pixel_y = view["pixel"]
        box_x, box_y, box_w, box_h = view["box"]
        return {
            "image": filename,
            "angle_group": group,
            "azimuth_deg": round(view["azimuth"], 3),
            "elevation_deg": round(view["elevation"], 3),
            "distance_m": round(view["distance"], 4),
            "camera_position_world": [round(float(v), 5) for v in view["position"]],
            "object_position_world": [round(float(v), 5) for v in centre],
            "object_position_camera_xyz": [
                round(x_cam, 5), round(y_cam, 5), round(depth, 5)
            ],
            "object_position_camera_xy": [
                round(x_cam / depth, 5), round(y_cam / depth, 5)
            ],
            "object_pixel_xy": [round(pixel_x, 3), round(pixel_y, 3)],
            "object_pixel_xy_normalized": [
                round(pixel_x / self.width, 5), round(pixel_y / self.height, 5)
            ],
            "object_bbox_xywh": [
                round(box_x, 3), round(box_y, 3), round(box_w, 3), round(box_h, 3)
            ],
            "object_bbox_xywh_normalized": [
                round(box_x / self.width, 5),
                round(box_y / self.height, 5),
                round(box_w / self.width, 5),
                round(box_h / self.height, 5),
            ],
            "object_size_px": [round(box_w, 3), round(box_h, 3)],
            "labels": labels,
            "objects": labels,
        }

    def _write_labels(self, name, body_id, groups, samples, out_dir):
        info = self.bodies[body_id]
        extents = info["points"].max(axis=0) - info["points"].min(axis=0)
        payload = {
            "class": name,
            "count": len(samples),
            "image_size": {"width": self.width, "height": self.height},
            "intrinsics": {
                "fovy_deg": round(math.degrees(self.fovy), 5),
                "focal_px": round(self.focal, 5),
                "cx": self.width / 2.0,
                "cy": self.height / 2.0,
            },
            "angle_groups_deg": groups,
            "object_center_world": [round(float(v), 5) for v in info["centre"]],
            "object_radius_m": round(info["radius"], 5),
            "object_size_m": [round(float(v), 5) for v in extents],
            "samples": samples,
        }
        with open(out_dir / "labels.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def _inside_corridor(position):
    """Whether a camera position is inside the walkable corridor."""
    x, y, z = position
    return -5.3 < x < 5.3 and -5.3 < y < 5.3 and 0.15 < z < 1.8


def _sphere_directions(count):
    index = np.arange(count) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / count)
    theta = np.pi * (1.0 + 5.0**0.5) * index
    return np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )


def _ring_directions(count):
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def _circular_distance(a, b):
    delta = abs(a - b) % 360.0
    return min(delta, 360.0 - delta)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--angles", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(argv)

    model = mujoco.MjModel.from_xml_path(str(scene.SCENE_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    photographer = ScenePhotographer(model, data, args.width, args.height, args.seed)
    try:
        selected = [
            body_id
            for body_id, info in photographer.bodies.items()
            if args.only is None or info["name"] in args.only
        ]
        total = 0
        for body_id in selected:
            name = photographer.bodies[body_id]["name"]
            produced = photographer.generate(
                body_id, args.per_class, args.angles, args.out / name
            )
            total += produced
            print(f"{name}: {produced} images")
        print(f"Done: {total} images in {args.out}")
    finally:
        photographer.close()


if __name__ == "__main__":
    main()
