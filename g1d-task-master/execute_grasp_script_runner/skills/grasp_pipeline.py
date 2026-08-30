#!/usr/bin/env python3
import os
import time
import json
import rclpy
import numpy as np
from utils.tools import *
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from object_pose_interfaces.action import EstimateObjectPose
from task_interfaces.action import MoveArmJoints, MoveArmPose


CONFIG_DIR = 'config'
GRIP_CONFIG_FILE = 'grip_config.json'


class GripBadgePipeline(Node):

    def __init__(self, config_path, sensor_node, gripper_node):
        super().__init__("grip_car_badge_pipeline")

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.server_timeout_sec = self.config['server_timeout_sec']
        self.sensor = sensor_node
        self.gripper = gripper_node

        self.suction_force_threshold = self.config['suction_force_threshold']
        self.suction_contact_detected = False
        self.suction_initial_fz = None

        self.base_T_left_ee = None
        self.cam_T_obj = None
        self.left_ee_T_cam = dict_2_tf_mat(self.config['left_ee_T_cam'])
        self.obj_T_down_target, self.obj_T_up_target = self.get_obj_T_target()
    
        self.left_ee_pose_sub = self.create_subscription(
            PoseStamped,
            "/left_ee_pose",
            self.left_ee_pose_callback,
            10
        )

        self.object_client = ActionClient(
            self,
            EstimateObjectPose,
            "/object_pose/estimate"
        )

        self.move_j_client = ActionClient(
            self,
            MoveArmJoints,
            '/move_arm_j'
        )

        self.move_p_client = ActionClient(
            self,
            MoveArmPose,
            "/move_arm_p"
        )

        self.move_l_client = ActionClient(
            self,
            MoveArmPose,
            "/move_arm_l"
        )
        
    def left_ee_pose_callback(self, msg):
        self.base_T_left_ee = pose_2_tf_mat(msg.pose)

    def move_j(self, left_joint_states, right_joint_states):
        if not self.move_j_client.wait_for_server(timeout_sec=self.server_timeout_sec):
            self.get_logger().error("/move_arm_j server not available")
            return False
        
        move_j_goal = MoveArmJoints.Goal()
        move_j_goal.left_joints = left_joint_states
        move_j_goal.right_joints = right_joint_states
        move_j_goal.dry_run = False
        move_j_goal.duration = 0.0
    
        send_future = self.move_j_client.send_goal_async(move_j_goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
    
        if not goal_handle.accepted:
            self.get_logger().error("/move_arm_j goal rejected")
            return False
    
        self.get_logger().info("/move_arm_j goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if not result.success:
            self.get_logger().error(
                f"move_arm_j failed: error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False
    
        self.get_logger().info(
            "/move_arm_j result: "
            f"success={result.success}, error_code={result.error_code}, "
            f"message={result.message}"
        )
        return True
    
    def move_p(self, target_left_pose, target_right_pose):
        if not self.move_p_client.wait_for_server(timeout_sec=self.server_timeout_sec):
            self.get_logger().error("/move_arm_p server not available")
            return False

        move_p_goal = MoveArmPose.Goal()
        move_p_goal.left_pose = target_left_pose
        move_p_goal.right_pose = target_right_pose
        move_p_goal.dry_run = False

        send_future = self.move_p_client.send_goal_async(move_p_goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()

        if not handle.accepted:
            self.get_logger().error("/move_arm_p goal rejected")
            return False

        self.get_logger().info("/move_arm_p goal accepted")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if not result.success:
            self.get_logger().error(
                f"/move_arm_p failed: error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False
    
        self.get_logger().info(
            "/move_arm_p result: "
            f"success={result.success}, error_code={result.error_code}, "
            f"message={result.message}"
        )
        return True

    def move_l(self, target_left_pose, target_right_pose):
        if not self.move_l_client.wait_for_server(timeout_sec=self.server_timeout_sec):
            self.get_logger().error("/move_arm_l server not available")
            return False

        move_l_goal = MoveArmPose.Goal()
        move_l_goal.left_pose = target_left_pose
        move_l_goal.right_pose = target_right_pose
        move_l_goal.dry_run = False

        send_future = self.move_l_client.send_goal_async(move_l_goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()

        if not handle.accepted:
            self.get_logger().error("/move_arm_l goal rejected")
            return False

        self.get_logger().info("/move_arm_l goal accepted")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result().result
        if not result.success:
            self.get_logger().error(
                f"/move_arm_l failed: error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False
    
        self.get_logger().info(
            "/move_arm_l result: "
            f"success={result.success}, error_code={result.error_code}, "
            f"message={result.message}"
        )
        return True

    def move_l_feedback_callback(self, feedback_msg):
        """
        Called periodically while /move_arm_l is executing.
        Use this callback to monitor the force sensor.
        """
        force = self.sensor.get_next_force()

        if force is None:
            self.get_logger().warn("Force sensor data unavailable")
            return

        fx, fy, fz = force

        delta_fz = abs(fz - self.suction_initial_fz)

        self.get_logger().info(
            f"Suction force: "
            f"Fx={fx:.4f}, "
            f"Fy={fy:.4f}, "
            f"Fz={fz:.4f}, "
            f"ΔFz={delta_fz:.4f}"
        )

        if delta_fz >= self.suction_force_threshold:
            self.get_logger().info(
                f"Contact detected: "
                f"ΔFz={delta_fz:.4f} >= "
                f"{self.suction_force_threshold:.4f}"
            )
            self.suction_contact_detected = True
    
    def get_obj_T_target(self):
        t1 = np.array([
            [  0.0, -1.0,  0.0,  0.0],
            [  1.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0,  1.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0],
        ])
        t2 = np.array([
            [  1.0,  0.0,  0.0,  0.0],
            [  0.0,  0.0, -1.0,  0.0],
            [  0.0,  1.0,  0.0,  0.0],
            [  0.0,  0.0,  0.0,  1.0],
        ])
        rot = t1 @ t2 @ np.array(self.config['correct_matrix'])

        t3 = np.eye(4)
        t4 = np.eye(4)
        t3[1, 3] = -self.config['bottom_to_left_ee']-self.config['down_offset']
        t4[1, 3] = -self.config['bottom_to_left_ee']-self.config['up_offset']

        obj_T_down_target = rot @ t3
        obj_T_up_target = rot @ t4

        return obj_T_down_target, obj_T_up_target

    def prepare(self):
        if not self.config['if_prepare']:
            return True

        left_traj = self.config['prepare_traj']['left']
        right_traj = self.config['prepare_traj']['right']
        if self.config['test_mode']:
            left_traj = self.config['test_traj']['prepare']['left']
            right_traj = self.config['test_traj']['prepare']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self.move_j(left_joint_states, right_joint_states):
                return False

        time.sleep(2)
        return True

    def get_base_T_left_ee(self):
        self.get_logger().info("Waiting for end-effector pose...")

        while self.base_T_left_ee is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info(f"Got left EE pose:\n{self.base_T_left_ee}")

    def get_cam_T_obj(self):    
        # Get the object pose in the camera Frame 
        if not self.object_client.wait_for_server(timeout_sec=self.server_timeout_sec):
            self.get_logger().error("Object pose action server not available")
            return False

        goal = EstimateObjectPose.Goal()
        goal.model_label = self.config['model_label']
        goal.instance_index = 0
        goal.confidence_threshold = 0.0

        send_future = self.object_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Object pose goal rejected")
            return  False

        self.get_logger().info("Object pose goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        self.cam_T_obj = pose_2_tf_mat(result.result.pose.pose)
        self.get_logger().info(f"cam_T_obj:\n{self.cam_T_obj}")

        return True

    def cal_base_T_target(self):
        base_T_obj = self.base_T_left_ee @ self.left_ee_T_cam @ self.cam_T_obj

        base_T_down_tar = base_T_obj @ self.obj_T_down_target
        base_T_up_tar = base_T_obj @ self.obj_T_up_target
        
        down_target_pose = (mat_2_pose_array(base_T_down_tar))
        up_target_pose = (mat_2_pose_array(base_T_up_tar))

        self.get_logger().info(
            f"Down target pose:\n{base_T_down_tar}"
        )
        self.get_logger().info(
            f"Up target pose:\n{base_T_up_tar}"
        )
        return down_target_pose, up_target_pose

    def suction(self):
        # Turn on vacuum
        if not self.gripper.turn_on('left'):
            return False

        # Get current EE pose
        self.get_base_T_left_ee()
        current_pose = self.base_T_left_ee.copy()

        # Parameters
        approach_distance = self.config['max_approach_distance']

        # Get initial force
        force = self.sensor.get_next_force()

        if force is None:
            self.get_logger().error(
                "No force sensor data received"
            )
            return False

        _, _, initial_fz = force

        self.suction_initial_fz = initial_fz
        self.suction_contact_detected = False

        self.get_logger().info(f"Initial force: Fz={initial_fz:.4f} kg")

        # Create a large target along local +Y
        left_ee_T_approach = np.eye(4)
        left_ee_T_approach[1, 3] = approach_distance

        target_pose = current_pose @ left_ee_T_approach

        self.get_logger().info(
            f"Starting suction approach: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )

        # Send ONE linear-motion goal
        if not self.move_l_client.wait_for_server(
            timeout_sec=self.server_timeout_sec
        ):
            self.get_logger().error(
                "/move_arm_l server not available while suctioning."
            )
            return False

        goal = MoveArmPose.Goal()
        goal.left_pose = mat_2_pose_array(target_pose)
        goal.right_pose = []
        goal.dry_run = False

        send_future = self.move_l_client.send_goal_async(
            goal,
            feedback_callback=self.move_l_feedback_callback
        )
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error( "/move_arm_l goal rejected")
            return False
        self.get_logger().info("/move_arm_l goal accepted")

        # Wait until movement finishes or contact is detected
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.01)
            if self.suction_contact_detected:
                self.get_logger().info("Suction force exceeds the threshold, stopping linear motion.")
                cancel_future = goal_handle.cancel_goal_async()
                rclpy.spin_until_future_complete(self, cancel_future)
                break

        # Get action result
        if not result_future.done():
            rclpy.spin_until_future_complete(self,result_future)

        result = result_future.result().result

        # Contact detected
        if self.suction_contact_detected:
            self.get_logger().info("Suction successful: contact detected.")
            return True

        # Motion completed without contact
        if result.success:
            self.get_logger().warn(
                "Maximum suction approach distance reached "
                "without detecting contact."
            )
            return True
        
        self.get_logger().error(
            f"/move_arm_l failed: "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )
        return False

    def withdraw(self, withdraw_target_pose):
        if not self.move_l_client.wait_for_server(
            timeout_sec=self.server_timeout_sec
        ):
            self.get_logger().error(
                "/move_arm_l server not available while withdrawing."
            )
            return False
                
        goal = MoveArmPose.Goal()
        goal.left_pose = withdraw_target_pose
        goal.right_pose = []
        goal.dry_run = False

        send_future = self.move_l_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)
        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error( "/move_arm_l goal rejected")
            return False

        # Wait until movement finishes or contact is detected
        self.get_logger().info("/move_arm_l goal accepted")
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        
        result = result_future.result().result
        if not result.success:
            self.get_logger().error(
                f"/move_arm_l failed: error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False
 
        self.get_logger().info(
            "/move_arm_l result: "
            f"success={result.success}, error_code={result.error_code}, "
            f"message={result.message}"
        )
        return True

    def run(self):
        if not self.prepare():
            return

        self.get_base_T_left_ee()

        if not self.get_cam_T_obj():
            return

        down_target_pose, up_target_pose = self.cal_base_T_target()

        if not self.move_p(down_target_pose, []):
            return

        if not self.suction():
            return 

        return self.withdraw(up_target_pose)

    def test_run(self):
        if not self.prepare():
            return

        time.sleep(10)

        if not self.suction():
            return

        left_traj = self.config['test_traj']['end']['left']
        right_traj = self.config['test_traj']['end']['right']

        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self.move_j(left_joint_states, right_joint_states):
                return False

        return True


def main():
    config_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), CONFIG_DIR)
    config_path = os.path.join(config_dir, GRIP_CONFIG_FILE)

    rclpy.init()
    node = GripBadgePipeline(config_path)    
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
