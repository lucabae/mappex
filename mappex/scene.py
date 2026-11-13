"""Geometry and layout of the simulated corridor scene."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT / "mujoco" / "models" / "corridor.xml"

DRONE_SPAN = 0.62
CORRIDOR_WIDTH = 7.0 * DRONE_SPAN

# The scene is a square corridor around a solid inner block: a wall at +/-inner
# and a second one at +/-outer.
INNER_HALF = 1.0
CENTER_HALF = INNER_HALF + CORRIDOR_WIDTH / 2.0
OUTER_HALF = 2.0 * CENTER_HALF - INNER_HALF

HOVER_ALTITUDE = 1.0
START = (CENTER_HALF, -CENTER_HALF, HOVER_ALTITUDE)

# A closed square whose corners sit in the middle of the corridor.
SQUARE_SIDE = 2.0 * CENTER_HALF

CAMERA_FOVY = 45.0
FRAME_SIZE = (480, 640)
FRAME_FPS = 30.0

WALL_HALVES = (INNER_HALF, OUTER_HALF)
