"""RViz target markers. All inputs are planner poses, never motion commands."""

import json
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path

from visualization_msgs.msg import Marker, MarkerArray

SNAPSHOT = Path('/rm_nvme/recordings/mission_logs/arm_targets/latest.json')
TARGET_TOPIC = '/mission/arm_target_markers'
ACTUAL_TOPIC = '/mission/arm_actual_markers'
COLORS = {'left': (0.1, 1.0, 0.25), 'right': (1.0, 0.45, 0.05)}


def pose_record(pose):
    p, q = pose.pose.position, pose.pose.orientation
    return {'frame': pose.header.frame_id, 'xyz': [p.x, p.y, p.z],
            'q': [q.x, q.y, q.z, q.w]}


def make_markers(record, *, actual=False):
    output = MarkerArray()
    clear = Marker()
    clear.action = Marker.DELETEALL
    output.markers.append(clear)
    namespace = 'actual' if actual else 'targets'

    def marker(pose, kind, color, size, text=''):
        m = Marker()
        m.header.frame_id = pose['frame']
        # Targets are frozen in a chassis frame, use current TF for display.
        m.ns = namespace
        m.id = len(output.markers)
        m.type = kind
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, pose['xyz'])
        m.pose.orientation.x, m.pose.orientation.y, m.pose.orientation.z, m.pose.orientation.w = map(float, pose['q'])
        m.color.r, m.color.g, m.color.b, m.color.a = map(float, color)
        m.scale.x = m.scale.y = m.scale.z = float(size)
        m.text = text
        if actual:
            m.lifetime.sec = 2
        output.markers.append(m)
        return m

    box = record.get('box')
    if box:
        marker(box, Marker.SPHERE, (1., 1., 1., 1.), .025)
        label = deepcopy(box)
        label['xyz'][2] += .13
        marker(label, Marker.TEXT_VIEW_FACING, (1., 1., 1., 1.), .028,
               record['phase'])
    for arm, entries in record['arms'].items():
        color = COLORS[arm]
        contact = entries['contact']
        marker(contact, Marker.SPHERE, (*color, 1.), .028 if actual else .045)
        label = deepcopy(contact)
        label['xyz'][2] += .06 if actual else .10
        marker(label, Marker.TEXT_VIEW_FACING, (*color, 1.), .025,
               f'{arm.upper()} {"ACTUAL" if actual else "AFTER"} contact')
        raw = entries.get('raw')
        if raw and not actual:
            marker(raw, Marker.SPHERE, (*color, .35), .03)
            before_label = deepcopy(raw)
            before_label['xyz'][2] += .045
            marker(before_label, Marker.TEXT_VIEW_FACING, (*color, .8), .022,
                   f'{arm.upper()} BEFORE contact')
    return output


def publish_targets(node, box, entries, phase):
    publisher = getattr(node, '_arm_target_marker_publisher', None)
    if publisher is None:
        return
    try:
        record = {'time': datetime.now().astimezone().isoformat(), 'phase': phase,
                  'box': pose_record(box), 'arms': entries}
        publisher.publish(make_markers(record))
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        temporary = SNAPSHOT.with_suffix(f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(record), encoding='utf-8')
        temporary.replace(SNAPSHOT)
    except Exception as exc:
        node.get_logger().warning(f'Arm target visualization failed: {exc}')


def publish_drag_join_targets(node, current_box, left_target, actual_right):
    if getattr(node, '_arm_target_marker_publisher', None) is None:
        return
    try:
        from geometry_msgs.msg import PoseStamped
        frame = node._string('grasp_box_tf_freeze_frame')
        base_to_left = node._lookup_tf_carry_transform(
            frame, node._string('left_arm_base_frame'),
            parameter_prefix='drag_box_tf_body_home_carry')
        left_pose = PoseStamped()
        left_pose.pose = left_target
        left_world = node._compose_transform(
            base_to_left, node._pose_stamped_to_transform(left_pose))
        entries = {}
        for arm, link in [('left', left_world), ('right', actual_right)]:
            center = node._compose_transform(link, (
                tuple(node._float_array(f'{arm}_fixture_center_in_link8_xyz')),
                (0., 0., 0., 1.)))
            entries[arm] = {
                name: {'frame': frame, 'xyz': list(tf[0]), 'q': list(tf[1])}
                for name, tf in [('link8', link), ('contact', center)]}
        box = PoseStamped()
        box.header.frame_id = frame
        box.pose = node._endpoint_sync_transform_to_pose(current_box)
        publish_targets(node, box, entries, 'Drag3 LEFT join target / RIGHT held')
    except Exception as exc:
        node.get_logger().warning(f'Drag join visualization failed: {exc}')
