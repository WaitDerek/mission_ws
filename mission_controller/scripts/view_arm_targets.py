"""Read-only marker relay: last planner snapshot plus live TF fixture centers."""
import json
import math
import time
from datetime import datetime
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener, TransformException
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import PoseStamped

from mission_runtime.arm_target_visualization import (
    SNAPSHOT, TARGET_TOPIC, ACTUAL_TOPIC, make_markers,
)
from mission_runtime.common import quaternion_multiply, rotate_vector

ROOT = Path('/rm_nvme/recordings/code/mission_ws/src/mission_controller')


def normalize(q):
    norm = math.sqrt(sum(v*v for v in q))
    return tuple(v/norm for v in q)


def replay_initial(parameters, observed=None):
    """Explicitly labelled reconstruction, not a record of measured arrival."""
    audit = Path('/rm_nvme/recordings/mission_logs/action_audit')
    files = sorted(audit.glob('*/mission_controller/execute*grasp_tf_*.jsonl'),
                   key=lambda p: p.stat().st_mtime, reverse=True)
    for path in files:
        records = [json.loads(line) for line in path.read_text().splitlines()]
        starts = [r for r in records if r.get('event') == 'start']
        results = [r for r in records if r.get('event') == 'result']
        if not starts or (not results and observed is None):
            continue
        result = (results[-1].get('result') or {}) if results else {}
        box = observed or result.get('box_pose') or {}
        if box.get('header', {}).get('frame_id') not in ('base_link', 'base_footprint'):
            continue
        if observed is not None:
            stamp = observed['header']['stamp']['sec']
            start = datetime.fromisoformat(starts[-1]['time']).timestamp()
            end = datetime.fromisoformat(results[-1]['time']).timestamp() if results else time.time()
            if not start <= stamp <= end:
                continue
        pose = box['pose']
        xyz = [pose['position'][a] for a in 'xyz']
        q = normalize(tuple(pose['orientation'][a] for a in 'xyzw'))
        goal = starts[-1]['goal']
        prefix = 'drag_box_tf' if 'drag' in starts[-1]['action'] else 'grasp_box_tf'
        model, layer = goal['box_type'], goal['box_layer']
        frame = box['header']['frame_id']
        output = {'phase': f'REPLAY initial {starts[-1]["time"][11:19]} (current config)',
                  'box': {'frame': frame, 'xyz': xyz, 'q': q}, 'arms': {}}
        for arm in ('left', 'right'):
            offset = parameters[f'{prefix}_direct_movel_{arm}_offset_xyz_{model}_layer{layer}']
            correction = parameters[f'{prefix}_joint123_{arm}_target_correction_pose_box_{model}_layer{layer}']
            d = rotate_vector(tuple(offset[i]+correction[i] for i in range(3)), q)
            raw = [xyz[i]+d[i] for i in range(3)]
            if prefix == 'drag_box_tf':
                raw[2] += parameters.get('drag_box_tf_contact_height_offset_m', 0.0)
            link_q = normalize(quaternion_multiply(
                quaternion_multiply(q, normalize(correction[3:])),
                normalize(parameters[f'direct_movel_{arm}_box_to_link8_orientation'])))
            contact = list(raw)
            k = parameters[f'{prefix}_{arm}_contact_forward_delta_scale']
            # Installed base_Link/base_link -> footprint is a pure Z translation.
            contact[1] = xyz[1] + k*(raw[1]-xyz[1])
            fixture = rotate_vector(tuple(parameters[f'{arm}_fixture_center_in_link8_xyz']), link_q)
            link = [contact[i]-fixture[i] for i in range(3)]
            output['arms'][arm] = {
                name: {'frame': frame, 'xyz': position, 'q': link_q}
                for name, position in [('raw', raw), ('contact', contact), ('link8', link)]}
        return output
    return None


class Viewer(Node):
    def __init__(self):
        super().__init__('mission_arm_target_viewer')
        self.parameters = {}
        for filename in ('core', 'direct_motion', 'grasp_tf', 'drag'):
            data = yaml.safe_load((ROOT/'config/mission'/f'{filename}.yaml').read_text())
            self.parameters.update(data['mission_controller']['ros__parameters'])
        qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                         reliability=ReliabilityPolicy.RELIABLE)
        self.target_pub = self.create_publisher(MarkerArray, TARGET_TOPIC, qos)
        self.actual_pub = self.create_publisher(MarkerArray, ACTUAL_TOPIC, qos)
        self.buffer = Buffer()
        self.listener = TransformListener(self.buffer, self)
        self.fallback = replay_initial(self.parameters)
        self.observation = self.create_subscription(
            PoseStamped, '/mission/box_object_pose_raw', self.observed, qos)
        self.create_timer(.5, self.tick)
        self.get_logger().info(f'Read-only visualization: {TARGET_TOPIC}, {ACTUAL_TOPIC}')

    def observed(self, message):
        p, q = message.pose.position, message.pose.orientation
        box = {'header': {'frame_id': message.header.frame_id,
                          'stamp': {'sec': message.header.stamp.sec}},
               'pose': {'position': dict(zip('xyz', (p.x,p.y,p.z))),
                        'orientation': dict(zip('xyzw', (q.x,q.y,q.z,q.w)))}}
        latest = replay_initial(self.parameters, observed=box)
        if latest:
            self.fallback = latest

    def tick(self):
        try:
            target = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else self.fallback
            if target:
                self.target_pub.publish(make_markers(target))
            actual = {'phase': 'ACTUAL live TF', 'arms': {}}
            for arm in ('left', 'right'):
                try:
                    tf = self.buffer.lookup_transform('base_footprint',
                        self.parameters[f'{arm}_link8_frame'], Time())
                except TransformException:
                    continue
                stamp = Time.from_msg(tf.header.stamp)
                if (self.get_clock().now() - stamp).nanoseconds > 2_000_000_000:
                    continue
                t, q = tf.transform.translation, tf.transform.rotation
                rotation = (q.x, q.y, q.z, q.w)
                d = rotate_vector(tuple(self.parameters[f'{arm}_fixture_center_in_link8_xyz']), rotation)
                actual['arms'][arm] = {
                    'link8': {'frame': 'base_footprint', 'xyz': [t.x,t.y,t.z], 'q': rotation},
                    'contact': {'frame': 'base_footprint', 'xyz': [t.x+d[0],t.y+d[1],t.z+d[2]], 'q': rotation}}
            self.actual_pub.publish(make_markers(actual, actual=True))
        except Exception as exc:
            self.get_logger().warning(f'Visualization refresh: {exc}')


def main():
    rclpy.init()
    node = Viewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
