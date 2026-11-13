"""The simulated drone: physics, flight primitives and camera streams."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from mappex import scene
from mappex.depth import CameraPose, Intrinsics

KP_POS = 6.0
KD_POS = 4.9
MAX_ACC = 3.0
KP_ATT = 12.0
KD_ATT = 1.7

MOVE_SPEED = 2.0
MAX_MOVE_ACC = 3.0
TURN_RATE = 60.0

BODY_NAME = "x2"
CAMERA_NAME = "fpv"


@dataclass
class _Motion:
    position: np.ndarray
    velocity: np.ndarray
    accel: np.ndarray
    yaw: float


class Drone:
    """A quadrotor in the corridor scene.

    Flight commands block until the manoeuvre finishes, emitting frames at a
    fixed rate so callers can process them as the drone moves.
    """

    def __init__(
        self,
        start=scene.START,
        scene_path=scene.SCENE_PATH,
        frame_size=scene.FRAME_SIZE,
        camera_fovy=scene.CAMERA_FOVY,
        frame_fps=scene.FRAME_FPS,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BODY_NAME)
        self.camera_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, CAMERA_NAME
        )
        if self.body_id < 0 or self.camera_id < 0:
            raise RuntimeError(f"Scene must contain '{BODY_NAME}' and '{CAMERA_NAME}'")

        self.model.cam_fovy[self.camera_id] = float(camera_fovy)
        self.height, self.width = frame_size
        self.intrinsics = Intrinsics(self.width, self.height, camera_fovy)
        self.frame_fps = float(frame_fps)
        self.on_frame = None
        self.on_depth = None

        self.mass = float(self.model.body_subtreemass[self.body_id])
        self._mixer = self._thrust_mixer()
        self._renderer = None
        self._depth_renderer = None
        self._yaw = 0.0

        self._reset(start)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    # -- state ---------------------------------------------------------

    @property
    def position(self):
        return self.data.qpos[0:3].copy()

    @property
    def yaw(self):
        return self._yaw

    def camera_pose(self):
        """World pose of the FPV camera, valid after the last physics step."""
        return CameraPose(
            position=self.data.cam_xpos[self.camera_id].copy(),
            rotation=self.data.cam_xmat[self.camera_id].reshape(3, 3).copy(),
        )

    # -- streams -------------------------------------------------------

    def frame(self):
        """Current FPV colour frame as uint8 ``(H, W, 3)``, RGB."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
        self._renderer.update_scene(self.data, camera=self.camera_id)
        return self._renderer.render()

    def depth(self):
        """Current FPV depth buffer in metres, perpendicular to the image plane."""
        if self._depth_renderer is None:
            self._depth_renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )
            self._depth_renderer.enable_depth_rendering()
        self._depth_renderer.update_scene(self.data, camera=self.camera_id)
        return self._depth_renderer.render()

    def sample(self):
        """Emit one colour frame and depth buffer to the callbacks."""
        if self.on_frame is not None:
            self.on_frame(self.frame())
        if self.on_depth is not None:
            self.on_depth(self.depth())

    # -- flight --------------------------------------------------------

    def hover(self, seconds=1.0):
        """Hold position and heading for ``seconds`` of simulated time."""
        return self._hold(self.position, self._yaw, seconds)

    def rotate(self, degrees, rate=TURN_RATE):
        """Turn in place by ``degrees``, keeping position and altitude."""
        target = self._yaw + math.radians(float(degrees))
        return self._rotate_to(target, rate)

    def forward(self, distance, speed=MOVE_SPEED):
        """Fly ``distance`` metres along the current heading."""
        heading = np.array([math.cos(self._yaw), math.sin(self._yaw), 0.0])
        return self._move_to(self.position + heading * float(distance), self._yaw, speed)

    def square(self, side, turns=4, speed=MOVE_SPEED):
        """Fly a closed square of ``side`` metres, turning left at each corner."""
        for _ in range(turns):
            self.rotate(90)
            self.forward(side, speed)

    def _move_to(self, target, yaw, speed):
        start = self.position
        delta = target - start
        distance = float(np.linalg.norm(delta))
        self._yaw = yaw
        if distance < 1e-4:
            return self._hold(target, yaw, 0.2)

        duration = max(
            1.875 * distance / max(float(speed), 1e-3),
            math.sqrt(5.7735 * distance / MAX_MOVE_ACC),
            0.8,
        )
        self._run(duration, lambda s: _Motion(start + delta * s[0], delta * s[1] / duration, delta * s[2] / (duration**2), yaw))
        return self._hold(target, yaw, 0.6)

    def _rotate_to(self, target_yaw, rate):
        start_yaw = self._yaw
        delta = target_yaw - start_yaw
        hold = self.position
        self._yaw = target_yaw
        if abs(delta) < 1e-4:
            return self._hold(hold, target_yaw, 0.2)

        angular_rate = math.radians(max(float(rate), 1.0))
        duration = max(1.875 * abs(delta) / angular_rate, 0.5)
        zeros = np.zeros(3)
        self._run(duration, lambda s: _Motion(hold, zeros, zeros, start_yaw + delta * s[0]))
        return self._hold(hold, target_yaw, 0.6)

    def _hold(self, target, yaw, seconds):
        if seconds > 0:
            zeros = np.zeros(3)
            self._run(seconds, lambda _: _Motion(target, zeros, zeros, yaw))
        return True

    def _run(self, duration, motion_at):
        """Step the simulation for ``duration``, tracking a smooth reference."""
        start_time = float(self.data.time)
        next_frame = start_time
        interval = 1.0 / max(self.frame_fps, 1.0)
        while float(self.data.time) - start_time < duration:
            progress = (float(self.data.time) - start_time) / duration
            motion = motion_at(_min_jerk(min(max(progress, 0.0), 1.0)))
            self._control(motion)
            mujoco.mj_step(self.model, self.data)
            if self.data.time >= next_frame:
                self.sample()
                next_frame = float(self.data.time) + interval
        self._yaw = motion_at(_min_jerk(1.0)).yaw

    # -- physics -------------------------------------------------------

    def _reset(self, start):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = np.asarray(start, dtype=float)
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._yaw = math.atan2(
            self.data.xmat[self.body_id].reshape(3, 3)[1, 0],
            self.data.xmat[self.body_id].reshape(3, 3)[0, 0],
        )

    def _thrust_mixer(self):
        """Inverse of the mapping from rotor thrusts to body wrench."""
        wrench = np.zeros((4, self.model.nu))
        for i in range(self.model.nu):
            site = self.model.actuator_trnid[i, 0]
            offset = self.model.site_pos[site]
            thrust = self.model.actuator_gear[i, 2]
            wrench[0, i] = thrust
            wrench[1, i] = offset[1] * thrust
            wrench[2, i] = -offset[0] * thrust
            wrench[3, i] = self.model.actuator_gear[i, 5]
        return np.linalg.inv(wrench)

    def _control(self, motion):
        rotation = self.data.xmat[self.body_id].reshape(3, 3)
        accel = (
            KP_POS * (motion.position - self.data.qpos[0:3])
            + KD_POS * (motion.velocity - self.data.qvel[0:3])
            + motion.accel
        )
        norm = np.linalg.norm(accel)
        if norm > MAX_ACC:
            accel *= MAX_ACC / norm

        thrust = self.mass * (accel + np.array([0.0, 0.0, 9.81]))
        body_z = thrust / np.linalg.norm(thrust)
        heading = np.array([math.cos(motion.yaw), math.sin(motion.yaw), 0.0])
        body_y = np.cross(body_z, heading)
        body_y /= np.linalg.norm(body_y)
        body_x = np.cross(body_y, body_z)
        desired = np.column_stack([body_x, body_y, body_z])

        error = 0.5 * _skew_vee(desired.T @ rotation - rotation.T @ desired)
        torque = -KP_ATT * error - KD_ATT * self.data.qvel[3:6]

        ctrl = self._mixer @ np.array([thrust @ rotation[:, 2], *torque])
        self.data.ctrl[:] = np.clip(
            ctrl,
            self.model.actuator_ctrlrange[:, 0],
            self.model.actuator_ctrlrange[:, 1],
        )

    def close(self):
        for renderer in (self._renderer, self._depth_renderer):
            if renderer is not None:
                renderer.close()
        self._renderer = None
        self._depth_renderer = None


def _min_jerk(s):
    """Position, velocity and acceleration scale factors for a smooth move."""
    return (
        10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5,
        30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4,
        60.0 * s - 180.0 * s**2 + 120.0 * s**3,
    )


def _skew_vee(matrix):
    """Axis-angle vector of the skew-symmetric part of ``matrix``."""
    return 0.5 * np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    )
