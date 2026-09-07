import math

from geometry_msgs.msg import Pose


class MissionError(RuntimeError):
    """Expected failure in the G1-D grasp sequence."""


class MissionCanceled(MissionError):
    """The client canceled the active grasp sequence."""


def pose_to_array(pose: Pose) -> list[float]:
    values = [
        float(pose.position.x),
        float(pose.position.y),
        float(pose.position.z),
        float(pose.orientation.x),
        float(pose.orientation.y),
        float(pose.orientation.z),
        float(pose.orientation.w),
    ]
    if not all(math.isfinite(value) for value in values):
        raise MissionError("pose contains NaN or Inf")
    norm = math.sqrt(sum(value * value for value in values[3:]))
    if norm < 1e-8:
        raise MissionError("pose quaternion has zero norm")
    values[3:] = [value / norm for value in values[3:]]
    return values


def quaternion_multiply(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def rotate_vector(
    vector: tuple[float, float, float],
    quaternion: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    qx, qy, qz, qw = quaternion
    vx, vy, vz = vector
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return (
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    )


def compose_poses(parent_pose: Pose, child_pose: Pose) -> Pose:
    """Compose T_reference_parent with T_parent_child."""
    parent_values = pose_to_array(parent_pose)
    child_values = pose_to_array(child_pose)
    parent_quaternion = tuple(parent_values[3:])
    child_position = rotate_vector(tuple(child_values[:3]), parent_quaternion)
    orientation = quaternion_multiply(parent_quaternion, tuple(child_values[3:]))
    orientation_norm = math.sqrt(sum(value * value for value in orientation))

    result = Pose()
    result.position.x = parent_values[0] + child_position[0]
    result.position.y = parent_values[1] + child_position[1]
    result.position.z = parent_values[2] + child_position[2]
    result.orientation.x = orientation[0] / orientation_norm
    result.orientation.y = orientation[1] / orientation_norm
    result.orientation.z = orientation[2] / orientation_norm
    result.orientation.w = orientation[3] / orientation_norm
    return result


def pose_from_transform(values: list[float]) -> Pose:
    """Convert a row-major homogeneous 4x4 matrix into a ROS pose."""
    if len(values) != 16:
        raise MissionError("transform must contain exactly 16 values")
    matrix = [float(value) for value in values]
    if not all(math.isfinite(value) for value in matrix):
        raise MissionError("transform contains NaN or Inf")
    if any(abs(value - expected) > 1e-6 for value, expected in zip(
        matrix[12:16], [0.0, 0.0, 0.0, 1.0]
    )):
        raise MissionError("transform last row must be [0, 0, 0, 1]")

    m00, m01, m02 = matrix[0:3]
    m10, m11, m12 = matrix[4:7]
    m20, m21, m22 = matrix[8:11]
    trace = m00 + m11 + m22
    if trace > 0.0:
        scale = 2.0 * math.sqrt(trace + 1.0)
        qw = 0.25 * scale
        qx = (m21 - m12) / scale
        qy = (m02 - m20) / scale
        qz = (m10 - m01) / scale
    elif m00 > m11 and m00 > m22:
        scale = 2.0 * math.sqrt(max(1.0 + m00 - m11 - m22, 0.0))
        qx = 0.25 * scale
        qy = (m01 + m10) / scale
        qz = (m02 + m20) / scale
        qw = (m21 - m12) / scale
    elif m11 > m22:
        scale = 2.0 * math.sqrt(max(1.0 + m11 - m00 - m22, 0.0))
        qx = (m01 + m10) / scale
        qy = 0.25 * scale
        qz = (m12 + m21) / scale
        qw = (m02 - m20) / scale
    else:
        scale = 2.0 * math.sqrt(max(1.0 + m22 - m00 - m11, 0.0))
        qx = (m02 + m20) / scale
        qy = (m12 + m21) / scale
        qz = 0.25 * scale
        qw = (m10 - m01) / scale

    quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if quaternion_norm < 1e-8:
        raise MissionError("transform rotation has zero norm")
    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = matrix[3], matrix[7], matrix[11]
    pose.orientation.x = qx / quaternion_norm
    pose.orientation.y = qy / quaternion_norm
    pose.orientation.z = qz / quaternion_norm
    pose.orientation.w = qw / quaternion_norm
    return pose
