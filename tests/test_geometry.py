import numpy as np
import pytest
from pyquaternion import Quaternion

from recoverdrive.geometry import count_points_in_boxes, project_box, to_sensor_frame
from recoverdrive.types import Box3D, Pose

K = np.array([[1000.0, 0, 800], [0, 1000, 450], [0, 0, 1]])
IMAGE_WH = (1600, 900)
IDENTITY = Pose(np.array([1.0, 0, 0, 0]), np.zeros(3))


def make_box(center, size=(2.0, 4.0, 2.0), yaw=0.0):
    q = Quaternion(axis=[0, 0, 1], angle=yaw)
    return Box3D(center=np.array(center, float), size=np.array(size, float), rotation=np.array(q.elements),
                 category="car")


def test_to_sensor_frame_translation_and_rotation():
    ego = Pose(np.array(Quaternion(axis=[0, 0, 1], angle=np.pi / 2).elements), np.array([5.0, 0, 0]))
    box = to_sensor_frame(make_box([5, 10, 0]), ego, IDENTITY)
    # Ego at (5, 0) facing +y: a box 10 m along global +y is 10 m straight ahead (+x) in the ego frame.
    np.testing.assert_allclose(box.center, [10, 0, 0], atol=1e-9)


def test_to_sensor_frame_applies_extrinsic():
    ext = Pose(np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 2.0]))
    box = to_sensor_frame(make_box([11, 0, 2]), IDENTITY, ext)
    np.testing.assert_allclose(box.center, [10, 0, 0], atol=1e-9)


def test_project_box_in_front():
    # Camera frame: z forward. With identity rotation, length (4) is along x, width (2) along y.
    box = to_sensor_frame(make_box([0, 0, 10]), IDENTITY, IDENTITY)
    b2d = project_box(box, K, IMAGE_WH)
    np.testing.assert_allclose(b2d, [800 - 2000 / 9, 450 - 1000 / 9, 800 + 2000 / 9, 450 + 1000 / 9], atol=1e-6)


def test_project_box_behind_camera_is_none():
    box = to_sensor_frame(make_box([0, 0, -10]), IDENTITY, IDENTITY)
    assert project_box(box, K, IMAGE_WH) is None


def test_project_box_outside_image_is_none():
    box = to_sensor_frame(make_box([100, 0, 10]), IDENTITY, IDENTITY)
    assert project_box(box, K, IMAGE_WH) is None


def test_project_box_crossing_near_plane_is_clipped():
    box = to_sensor_frame(make_box([0, 0, 0.5], size=(2, 2, 4)), IDENTITY, IDENTITY)
    b2d = project_box(box, K, IMAGE_WH)
    assert b2d is not None
    assert 0 <= b2d[0] < b2d[2] <= IMAGE_WH[0] and 0 <= b2d[1] < b2d[3] <= IMAGE_WH[1]


def test_count_points_in_boxes():
    box = to_sensor_frame(make_box([0, 0, 0], size=(2, 2, 2)), IDENTITY, IDENTITY)
    points = np.array([[0, 0, 0, 1], [0.5, 0.5, 0.5, 1], [5, 0, 0, 1], [0, 0, 1.5, 1]], dtype=np.float32)
    assert count_points_in_boxes(points, [box]).tolist() == [2]


@pytest.mark.parametrize("points", [np.zeros((0, 4))])
def test_count_points_empty_cloud(points):
    box = to_sensor_frame(make_box([0, 0, 0]), IDENTITY, IDENTITY)
    assert count_points_in_boxes(points, [box]).tolist() == [0]
    assert count_points_in_boxes(points, []).tolist() == []
