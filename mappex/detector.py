"""YOLO detection and resolution of the trained weights file."""

from pathlib import Path

import numpy as np

DEFAULT_WEIGHTS = "room_detection.pt"
WEIGHTS_ENV = "MAPPEX_WEIGHTS"
SEARCH_DIRS = ("runs", "weights")
LOCALISATION_PERCENTILE = 15
SURFACE_BAND = 0.35
MIN_PIXELS = 20


def resolve_weights(explicit=None):
    """Locate the trained weights, in order of preference.

    1. An explicit path, as given on the command line.
    2. ``$MAPPEX_WEIGHTS``.
    3. ``weights/room_detection.pt`` next to the project root.
    4. The newest ``best.pt`` produced by an Ultralytics run under ``runs/``.

    Nothing is hard-coded to a machine-specific absolute path, so a fresh clone
    works as soon as the training step has been run.
    """
    import os

    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get(WEIGHTS_ENV):
        candidates.append(Path(os.environ[WEIGHTS_ENV]).expanduser())
    root = Path(__file__).resolve().parent.parent
    candidates.append(root / "weights" / DEFAULT_WEIGHTS)
    candidates.extend(_trained_runs(root))

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "No trained weights found. Run `python -m mappex.train` first, then pass "
        f"--weights, set ${WEIGHTS_ENV}, or copy the file to weights/{DEFAULT_WEIGHTS}."
    )


def _trained_runs(root):
    runs = root / "runs"
    if not runs.is_dir():
        return []
    found = [p for p in runs.rglob("best.pt") if p.is_file()]
    return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)


class Detector:
    """Thin wrapper over Ultralytics that also estimates object positions."""

    def __init__(self, weights=None, confidence=0.25):
        from ultralytics import YOLO

        self.path = resolve_weights(weights)
        self.model = YOLO(str(self.path))
        self.confidence = confidence

    @property
    def names(self):
        return self.model.names

    def detect(self, image):
        """Yield ``(class_name, box)`` for each detection in a BGR frame."""
        result = self.model(image, conf=self.confidence, verbose=False)[0]
        for box in result.boxes:
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
            yield self.model.names[int(box.cls[0].item())], (x1, y1, x2, y2)


def locate(depth, box, intrinsics, pose, max_distance):
    """World centroid of the object inside ``box``, or ``None``.

    The centre pixel of a detection often lands on the surface behind the
    object, so the whole box is used: take the nearest depth percentile inside
    it and keep the pixels within a thin band of that value. That band is the
    object's front face, which is what the centroid should describe.
    """
    from mappex.depth import unproject

    x1, y1, x2, y2 = box
    height, width = depth.shape
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, width), min(y2, height)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    patch = depth[y1:y2, x1:x2]
    valid = (patch > 0.0) & (patch < max_distance)
    if valid.sum() < MIN_PIXELS:
        return None

    front = float(np.percentile(patch[valid], LOCALISATION_PERCENTILE))
    surface = valid & (patch <= front + SURFACE_BAND)
    if surface.sum() < MIN_PIXELS:
        return None

    rows, columns = np.nonzero(surface)
    pixels = np.column_stack([columns + x1, rows + y1])
    points = unproject(pixels, patch[surface], intrinsics, pose)
    return np.median(points, axis=0)
