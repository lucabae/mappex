"""Render the screenshots used in the README from the simulator.

Run this after training, once weights are available:

    python -m scripts.capture_docs

It writes ``docs/images/{scene,detection,depth,map}.png``.
"""

import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

from mappex import scene
from mappex.detector import Detector, locate
from mappex.mapping import ObjectMap
from mappex.simulation import Drone

OUT = Path(__file__).resolve().parent.parent / "docs" / "images"
POSE = ((3.17, 3.17), 180.0)


def scene_overview(mujoco_model):
    mujoco_model.vis.global_.offwidth = 1280
    mujoco_model.vis.global_.offheight = 1024
    data = mujoco.MjData(mujoco_model)
    mujoco.mj_forward(mujoco_model, data)
    renderer = mujoco.Renderer(mujoco_model, height=832, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.6]
    camera.distance = 13.0
    camera.elevation = -32.0
    camera.azimuth = 140.0
    renderer.update_scene(data, camera=camera)
    image = renderer.render()
    cv2.imwrite(str(OUT / "scene.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    renderer.close()


def capture_pose(detector):
    """One frame from a fixed pose: colour, detections and depth map."""
    (x, y), degrees = POSE
    with Drone(start=(x, y, scene.HOVER_ALTITUDE)) as drone:
        data = drone.data
        yaw = math.radians(degrees)
        data.qpos[0:3] = [x, y, scene.HOVER_ALTITUDE]
        data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        data.qvel[:] = 0
        mujoco.mj_forward(drone.model, data)

        frame = cv2.cvtColor(drone.frame(), cv2.COLOR_RGB2BGR)
        depth = drone.depth()
        pose = drone.camera_pose()
        intrinsics = drone.intrinsics

    mapping = ObjectMap()
    annotated = frame.copy()
    for name, box in detector.detect(frame):
        point = locate(depth, box, intrinsics, pose, 10.0)
        if point is not None:
            mapping.observe(name, float(point[0]), float(point[1]))
        _draw_box(annotated, box, name, point)

    scale = 1.5
    cv2.imwrite(str(OUT / "detection.png"), _upscale(annotated, scale))
    cv2.imwrite(str(OUT / "depth.png"), _depth_colour(depth, scale))
    return mapping


def _draw_box(image, box, name, point):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), (60, 220, 60), 2)
    label = name if point is None else f"{name} x={point[0]:.2f} y={point[1]:.2f}"
    (width, height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left = x1
    if left + width + 6 > image.shape[1]:
        left = max(image.shape[1] - width - 6, 0)
    cv2.rectangle(
        image, (left, y1 - height - 8), (left + width + 6, y1), (60, 220, 60), -1
    )
    cv2.putText(
        image, label, (left + 3, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (25, 25, 25), 1, cv2.LINE_AA,
    )


def _depth_colour(depth, scale):
    normalised = np.clip(depth / 6.0, 0.0, 1.0)
    grey = ((1.0 - normalised) * 255).astype("uint8")
    colour = cv2.applyColorMap(grey, cv2.COLORMAP_TURBO)
    return _upscale(colour, scale)


def _upscale(image, scale):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    detector = Detector(confidence=0.35)
    scene_overview(mujoco.MjModel.from_xml_path(str(scene.SCENE_PATH)))
    capture_pose(detector)

    from mappex.pipeline import MappingRun

    run = MappingRun(show=False, detector=detector)
    mapping = run.run()
    mapping.save(OUT / "map.png", pose=run.final_pose, path=run.trail)
    print(f"{len(mapping.visible())} objects mapped")
    print("Wrote screenshots to", OUT)


if __name__ == "__main__":
    main()
