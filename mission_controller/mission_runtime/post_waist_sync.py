"""Require settled hardware feedback and matching TF before final arm IK."""
import math
import time

import rclpy
from geometry_msgs.msg import Pose
from tf2_ros import TransformException

from .common import MissionError

# This gate only applies to TF grasp/drag after an actual waist move.
POSITION_TOLERANCE_RAD = 0.001
VELOCITY_TOLERANCE_RAD_SEC = 0.005
SETTLE_SEC = 0.30
MAX_AGE_SEC = 0.25
TF_TRANSLATION_TOLERANCE_M = 0.001
TF_ANGLE_TOLERANCE_RAD = 0.002


def transform_error(actual, expected):
    distance = math.sqrt(sum((a-b)**2 for a,b in zip(actual[0], expected[0])))
    qa, qb = actual[1], expected[1]
    norm = math.sqrt(sum(v*v for v in qa) * sum(v*v for v in qb))
    if norm < 1e-12:
        return distance, math.inf
    cosine = min(1., max(0., abs(sum(a*b for a,b in zip(qa,qb))) / norm))
    return distance, 2*math.acos(cosine)


def feedback_is_settled(positions, velocities, desired, age):
    return (len(positions) >= 4 and len(velocities) >= 4 and age <= MAX_AGE_SEC
            and all(v is not None and math.isfinite(v) for v in positions[:4]+velocities[:4])
            and all(abs(p-t) <= POSITION_TOLERANCE_RAD for p,t in zip(positions[:3],desired))
            and all(abs(v) <= VELOCITY_TOLERANCE_RAD_SEC for v in velocities[:4]))


def wait_for_settled_tf_targets(node, goal_handle, left_target, right_target, desired):
    desired = list(desired)
    if len(desired) != 3 or not all(math.isfinite(v) for v in desired):
        raise MissionError('post-waist synchronization requires three finite target angles')
    frame = left_target.header.frame_id.lstrip('/')
    if frame != right_target.header.frame_id.lstrip('/'):
        raise MissionError('post-waist targets must share the same frozen frame')
    deadline = time.monotonic() + node._float('grasp_box_tf_runtime_tf_timeout_sec')
    stable_since = None
    stable_samples = 0
    last_sequence = -1
    detail = 'waiting for settled hardware feedback'
    node._publish_box_grasp_feedback(goal_handle, 'WAITING_POST_WAIST_TF_SYNC',
                                    'waiting for waist error <= 0.001 rad, stable feedback for 0.30s, '
                                    'and same-time arm-base TF matching measured waist; no arm motion sent')
    while time.monotonic() < deadline:
        node._check_canceled(goal_handle, 'while synchronizing post-waist TF')
        positions, velocities, stamp, sequence = node._body_feedback_snapshot()
        age = time.monotonic()-stamp
        matches = feedback_is_settled(positions, velocities, desired, age)
        detail = f'body_rad={positions}; target_rad={desired}; velocity={velocities}; age={age:.3f}s'
        transforms = {}
        common_ns = 0
        if matches:
            try:
                names = {arm: node._string(f'{arm}_arm_base_frame').lstrip('/') for arm in ('left','right')}
                latest = {arm: node.tf_buffer.lookup_transform(frame, name, rclpy.time.Time())
                          for arm,name in names.items()}
                common_ns = min(t.header.stamp.sec*10**9+t.header.stamp.nanosec for t in latest.values())
                tf_age = (node.get_clock().now().nanoseconds-common_ns)/1e9
                if common_ns <= 0 or not 0 <= tf_age <= MAX_AGE_SEC:
                    matches = False
                common_time = rclpy.time.Time(nanoseconds=common_ns)
                # The configured FK has chest Joint4 at zero; include its measured angle.
                chest = node._compose_transform(node._joint123_chest_transform(positions[:3]),
                                                node._rotation_transform((0.,0.,1.),positions[3]))
                for arm, name in names.items():
                    msg = node.tf_buffer.lookup_transform(frame, name, common_time)
                    tr,qr = msg.transform.translation,msg.transform.rotation
                    actual = ((tr.x,tr.y,tr.z),(qr.x,qr.y,qr.z,qr.w))
                    expected = node._compose_transform(chest, node._configured_rpy_transform(
                        f'box_chest_to_{arm}_arm_base_xyz', f'box_chest_to_{arm}_arm_base_rpy'))
                    distance,angle = transform_error(actual,expected)
                    matches = matches and distance <= TF_TRANSLATION_TOLERANCE_M and angle <= TF_ANGLE_TOLERANCE_RAD
                    detail += f'; {arm}_tf_error={distance:.6f}m/{angle:.6f}rad'
                    transforms[arm] = actual
                detail += f'; tf_age={tf_age:.3f}s; common_tf_ns={common_ns}'
            except TransformException as exc:
                matches = False
                detail += f'; TF not ready: {exc}'
        if not matches:
            stable_since = None
            stable_samples = 0
        elif sequence != last_sequence:
            if stable_since is None:
                stable_since = time.monotonic()
            stable_samples += 1
        last_sequence = sequence
        if matches and stable_samples >= 3 and time.monotonic()-stable_since >= SETTLE_SEC:
            result = []
            for arm, target in [('left',left_target),('right',right_target)]:
                frozen = node._pose_stamped_to_transform(target)
                transformed = node._compose_transform(node._inverse_transform(transforms[arm]),frozen)
                pose = Pose()
                pose.position.x,pose.position.y,pose.position.z = transformed[0]
                pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w = transformed[1]
                result.append(pose)
            node._publish_box_grasp_feedback(goal_handle, 'POST_WAIST_TF_SYNCED', detail)
            return result
        time.sleep(.02)
    raise MissionError(f'post-waist feedback/TF synchronization timed out; no arm motion sent; {detail}')
