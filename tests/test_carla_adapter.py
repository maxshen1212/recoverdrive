import numpy as np
from pyquaternion import Quaternion

from recoverdrive.adapters.carla_adapter import (
    DEFAULT_CAMERAS,
    actor_box,
    build_frame,
    camera_extrinsic,
    intrinsic_from_fov,
    ue_matrix_to_rh,
    ue_points_to_rh,
    ue_transform_matrix,
)
from recoverdrive.geometry import project_box, to_sensor_frame
from recoverdrive.types import Box3D


def test_ue_to_right_handed():
    m = ue_matrix_to_rh(ue_transform_matrix(10, 5, 1, 0, 0, 90))  # UE yaw +90 = turn right
    np.testing.assert_allclose(m[:3, 3], [10, -5, 1])
    np.testing.assert_allclose(m[:3, :3] @ [1, 0, 0], [0, -1, 0], atol=1e-9)  # forward now points to -y (right)
    np.testing.assert_allclose(ue_points_to_rh(np.array([[1, 2, 3, 0.5]]))[0], [1, -2, 3, 0.5])


def _box_at(x, y, z):
    return Box3D(center=np.array([x, y, z]), size=np.array([1.8, 4.5, 1.5]), rotation=np.array([1.0, 0, 0, 0]),
                 category="car")


def test_camera_extrinsic_projects_forward_point_to_center():
    front, back = DEFAULT_CAMERAS
    ego = build_frame("s", 0, 0, ue_transform_matrix(0, 0, 0, 0, 0, 0), {c.name: np.zeros((450, 800, 3), np.uint8)
                                                                          for c in DEFAULT_CAMERAS},
                      DEFAULT_CAMERAS, np.zeros((0, 4)), []).ego_pose
    k = intrinsic_from_fov(800, 450, 70)
    b = project_box(to_sensor_frame(_box_at(15, 0, 1.6), ego, camera_extrinsic(front.mount())), k, (800, 450))
    np.testing.assert_allclose([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], [400, 225], atol=1.0)
    assert project_box(to_sensor_frame(_box_at(-15, 0, 1.6), ego, camera_extrinsic(front.mount())), k,
                       (800, 450)) is None
    b = project_box(to_sensor_frame(_box_at(-15, 0, 1.6), ego, camera_extrinsic(back.mount())), k, (800, 450))
    np.testing.assert_allclose([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], [400, 225], atol=1.0)
    # A box to the left (+y, right-handed) appears in the left half of the front image.
    b = project_box(to_sensor_frame(_box_at(15, 3, 1.6), ego, camera_extrinsic(front.mount())), k, (800, 450))
    assert (b[0] + b[2]) / 2 < 400


def test_actor_box_global():
    box = actor_box(ue_transform_matrix(10, 5, 0, 0, 0, 90), (0.5, 0, 0.7), (2.0, 1.0, 0.75), "car")
    np.testing.assert_allclose(box.center, [10, -5.5, 0.7], atol=1e-9)  # local +x offset points to -y
    np.testing.assert_allclose(box.size, [2.0, 4.0, 1.5])
    assert abs(Quaternion(box.rotation).yaw_pitch_roll[0] + np.pi / 2) < 1e-6
