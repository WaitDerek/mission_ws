"""Project the upright box model onto the robot horizontal plane."""
from copy import deepcopy
import math
import numpy as np
from scipy.spatial.transform import Rotation
from .common import MissionError

def constrain_box_horizontal(pose, footprint_in_source):
    """Local X is up, local Z is lateral. Preserve center and lateral heading."""
    q = pose.pose.orientation
    source_from_box = Rotation.from_quat([q.x, q.y, q.z, q.w])
    source_from_foot = Rotation.from_quat(footprint_in_source[1])
    foot_from_box = (source_from_foot.inv() * source_from_box).as_matrix()
    lateral = foot_from_box[:, 2].copy()
    lateral[2] = 0.
    norm = np.linalg.norm(lateral)
    if not math.isfinite(norm) or norm < 1e-6:
        raise MissionError("box horizontal constraint: lateral box Z has no horizontal heading")
    lateral /= norm
    up = np.array([0., 0., 1.])
    second = np.cross(lateral, up)
    projected = np.column_stack((up, second, lateral))
    rotation = source_from_foot * Rotation.from_matrix(projected)
    result = deepcopy(pose)
    out = rotation.as_quat()
    result.pose.orientation.x, result.pose.orientation.y, result.pose.orientation.z, result.pose.orientation.w = map(float, out)
    tilt = math.degrees(math.acos(float(np.clip(foot_from_box[2, 0], -1., 1.))))
    return result, tilt

def apply_box_horizontal(node, pose, *, drag_mode):
    prefix = "drag_box_tf" if drag_mode else "grasp_box_tf"
    if not node._boolean(f"{prefix}_horizontal_constraint_enabled"):
        return deepcopy(pose), ""
    frame = pose.header.frame_id.strip().lstrip("/")
    if not frame:
        raise MissionError("box horizontal constraint requires a pose frame")
    foot = (((0., 0., 0.), (0., 0., 0., 1.)) if frame == "base_footprint"
            else node._lookup_tf_carry_transform(
                frame, "base_footprint", parameter_prefix=f"{prefix}_body_home_carry"))
    result, tilt = constrain_box_horizontal(pose, foot)
    return result, (f"horizontal_constraint=enabled; workflow={prefix}; "
                    "up=box_X_to_base_footprint_Z; heading=projected_box_Z; "
                    f"removed_tilt_deg={tilt:.4f}; center=unchanged; fixture_calibration=preserved")
