"""The exploration loop: fly the corridor and build the object map."""

import argparse
from pathlib import Path

import cv2

from mappex import scene
from mappex.detector import Detector, locate
from mappex.mapping import ObjectMap
from mappex.simulation import Drone

MAX_DISTANCE = 10.0
CONFIDENCE = 0.35


class MappingRun:
    """Wires the drone, the detector and the map together.

    The drone emits colour and depth frames as it flies; every frame is turned
    into detections, each detection into a world position, and each position
    into an entry of the map. The map is redrawn as it grows.
    """

    def __init__(self, weights=None, show=True, save_path=None, detector=None):
        self.detector = detector or Detector(weights=weights, confidence=CONFIDENCE)
        self.map = ObjectMap()
        self.show = show
        self.save_path = Path(save_path) if save_path else None
        self.trail = []
        self.depth = None
        self.pose = None
        self.final_pose = None
        self.detections = []
        self._drone = None

    def run(self, side=None, corners=4):
        side = scene.SQUARE_SIDE if side is None else side
        with Drone() as drone:
            self._drone = drone
            drone.on_frame = self._on_frame
            drone.on_depth = self._on_depth
            drone.hover(1.0)
            for _ in range(corners):
                drone.rotate(90)
                drone.forward(side)
            drone.hover(1.0)
            self.final_pose = drone.camera_pose()

        image = self.map.render(pose=self.final_pose, path=self.trail)
        if self.save_path is not None:
            cv2.imwrite(str(self.save_path), image)
        if self.show:
            cv2.imshow("Map", image)
            cv2.waitKey(500)
        cv2.destroyAllWindows()
        return self.map

    def _on_depth(self, depth):
        self.depth = depth
        self.pose = self._drone.camera_pose()

    def _on_frame(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        position = self._drone.position
        self.trail.append((float(position[0]), float(position[1])))
        self.detections = []

        for class_name, box in self.detector.detect(image):
            point = None
            if self.depth is not None and self.pose is not None:
                point = locate(
                    self.depth,
                    box,
                    self._drone.intrinsics,
                    self.pose,
                    MAX_DISTANCE,
                )
            if point is not None:
                self.map.observe(class_name, float(point[0]), float(point[1]))
            self.detections.append((class_name, box, point))

        self._annotate(image)
        if self.show:
            cv2.imshow("Camera", image)
            cv2.waitKey(1)

    def _annotate(self, image):
        for class_name, (x1, y1, x2, y2), point in self.detections:
            cv2.rectangle(image, (x1, y1), (x2, y2), (60, 220, 60), 2)
            label = class_name
            if point is not None:
                label = f"{class_name} x={point[0]:.2f} y={point[1]:.2f}"
            cv2.putText(
                image,
                label,
                (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (60, 220, 60),
                1,
                cv2.LINE_AA,
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description="Map objects while flying the corridor.")
    parser.add_argument("--weights", type=Path, default=None, help="trained YOLO weights")
    parser.add_argument("--output", "-o", type=Path, default=Path("map.png"), help="map image")
    parser.add_argument("--side", type=float, default=None, help="square side in metres")
    parser.add_argument("--headless", action="store_true", help="do not open windows")
    args = parser.parse_args(argv)

    run = MappingRun(weights=args.weights, show=not args.headless, save_path=args.output)
    mapping = run.run(side=args.side)
    for line in mapping.summary():
        print(line)
    print(f"Map written to {args.output}")


if __name__ == "__main__":
    main()
