"""Tests for camera maths, depth localisation and map merging."""

import math

import numpy as np
import pytest

from mappex.depth import CameraPose, Intrinsics, unproject
from mappex.mapping import Cluster, ObjectMap


@pytest.fixture
def intrinsics():
    return Intrinsics(width=640, height=480, fovy_deg=45.0)


def _pose_at(position, yaw_radians):
    """A camera at ``position`` facing ``yaw``, with world up as its up axis."""
    heading = np.array([math.cos(yaw_radians), math.sin(yaw_radians), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(heading, up)
    return CameraPose(position=np.array(position, dtype=float), rotation=np.column_stack([right, up, -heading]))


def test_focal_matches_vertical_fov(intrinsics):
    assert intrinsics.focal == pytest.approx((480 / 2) / math.tan(math.radians(45) / 2))
    assert 570 < intrinsics.focal < 590


def test_centre_pixel_maps_to_optical_axis(intrinsics):
    pose = _pose_at((1.0, 2.0, 3.0), 0.0)
    point = unproject((320, 240), 5.0, intrinsics, pose)[0]
    # Centre pixel is straight ahead: five metres along the view direction.
    assert point == pytest.approx([6.0, 2.0, 3.0], abs=1e-6)


def test_projection_round_trip(intrinsics):
    """Projecting a world point and unprojecting it must return the point."""
    pose = _pose_at((0.0, 0.0, 1.0), math.radians(30))
    target = np.array([3.0, 2.0, 0.5])

    offset = target - pose.position
    depth = float(offset @ -pose.back)
    focal = intrinsics.focal
    u = intrinsics.width / 2 + focal * (offset @ pose.right) / depth
    v = intrinsics.height / 2 - focal * (offset @ pose.up) / depth

    recovered = unproject((u, v), depth, intrinsics, pose)[0]
    assert recovered == pytest.approx(target, abs=1e-6)


def test_unproject_rejects_nothing_but_keeps_depth(intrinsics):
    pose = _pose_at((0.0, 0.0, 1.0), 0.0)
    near = unproject((320, 240), 2.0, intrinsics, pose)[0]
    far = unproject((320, 240), 8.0, intrinsics, pose)[0]
    assert far[0] - pose.position[0] == pytest.approx(8.0)
    assert near[0] - pose.position[0] == pytest.approx(2.0)


def test_merge_folds_repeat_sightings_of_one_object():
    mapping = ObjectMap()
    for i in range(20):
        mapping.observe("chair", 2.0 + 0.01 * (i % 3), 3.0)
    bucket = mapping.clusters["chair"]
    assert len(bucket) == 1
    assert bucket[0].views == 20
    assert bucket[0].x == pytest.approx(2.01, abs=0.02)


def test_merge_keeps_two_objects_apart():
    mapping = ObjectMap()
    for _ in range(10):
        mapping.observe("chair", 0.0, 0.0)
        mapping.observe("chair", 4.0, 0.0)
    assert len(mapping.clusters["chair"]) == 2


def test_classes_do_not_mix():
    mapping = ObjectMap()
    for _ in range(10):
        mapping.observe("chair", 0.0, 0.0)
        mapping.observe("bench", 0.0, 0.0)
    assert set(mapping.clusters) == {"chair", "bench"}
    assert all(len(v) == 1 for v in mapping.clusters.values())


def test_weak_clusters_are_not_drawn():
    mapping = ObjectMap(min_views=5)
    mapping.clusters = {"tv": [Cluster(0.0, 0.0, views=4)]}
    assert mapping.visible() == []


def test_close_clusters_of_one_class_draw_once():
    """Two clusters of the same class near each other are the same object."""
    mapping = ObjectMap(min_views=1, draw_separation=1.0)
    mapping.clusters = {
        "tv": [Cluster(1.0, 1.0, views=100), Cluster(1.2, 1.1, views=90)]
    }
    visible = mapping.visible()
    assert len(visible) == 1
    assert visible[0][1].views == 100


def test_render_returns_an_image():
    mapping = ObjectMap(min_views=1)
    mapping.observe("chair", 1.0, 2.0)
    image = mapping.render()
    assert image.shape[2] == 3
    assert image.dtype == np.uint8
    assert image.shape[0] == image.shape[1]


def test_non_finite_observations_are_ignored():
    mapping = ObjectMap()
    mapping.observe("chair", float("nan"), 0.0)
    mapping.observe("chair", float("inf"), 1.0)
    assert "chair" not in mapping.clusters
