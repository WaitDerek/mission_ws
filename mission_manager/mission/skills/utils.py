import numpy as np
from scipy.spatial.transform import Rotation as R


def pose_2_tf_mat(pose):
    T = np.eye(4)
    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w
    ]
    T[:3,:3] = R.from_quat(q).as_matrix()
    T[:3,3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z
    ]
    return T

def dict_2_tf_mat(pos_dict):
    rot = R.from_quat(pos_dict['orientation']).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = pos_dict['position']
    return T

def dict_2_pose_array(pos_dict):
    pose_array = pos_dict['position'][:]
    pose_array.extend(pos_dict['orientation'])
    return pose_array

def mat_2_pose_array(T):
    xyz = T[:3,3]
    quat = R.from_matrix(
        T[:3,:3]
    ).as_quat()
    return [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),

        float(quat[0]),
        float(quat[1]),
        float(quat[2]),
        float(quat[3]), 
    ]

def euler_to_tf_mat(poses, order="xyz"):
    """
    Sequentially compose a list of [x, y, z, rx, ry, rz] poses.

    Args:
        poses: list of poses
            [
                [x, y, z, rx, ry, rz],
                [x, y, z, rx, ry, rz],
                ...
            ]

        order: Euler rotation order, e.g. "xyz", "zyx".

    Returns:
        np.ndarray: 4x4 transformation matrix list
    """
    T_list = []

    for pose in poses:
        x, y, z, rx, ry, rz = pose

        Ti = np.eye(4)
        Ti[:3, :3] = R.from_euler(
            order, [rx, ry, rz], degrees=True
        ).as_matrix()
        Ti[:3, 3] = [x, y, z]

        T_list.append(Ti)

    return T_list
