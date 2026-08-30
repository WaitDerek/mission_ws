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
GRIP_CONFIG_FILE = 'peel_config.json'


class PeelPipeline(Node):

    def __init__(self, config_path, sensor_node, gripper_node):
        super().__init__("peel_back_film_pipeline")

        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.server_timeout_sec = self.config['server_timeout_sec']
        self.left_ee_update_time = None
        self.right_ee_update_time = None
        self.cam_T_obj = None
        self.base_T_left_ee = None
        self.base_T_right_ee = None
        self.obj_T_above_target = self.get_obj_T_target()
        self.right_ee_T_cam = dict_2_tf_mat(self.config['right_ee_T_cam'])

        self.suction_force_threshold = self.config['suction_force_threshold']
        self.suction_initial_fz = None
        self.suction_contact_detected = None
        
        self.sensor = sensor_node
        self.gripper = gripper_node

        self.left_ee_pose_sub = self.create_subscription(
            PoseStamped,
            "/left_ee_pose",
            self.left_ee_pose_callback,
            10
        )
    
        self.right_ee_pose_sub = self.create_subscription(
            PoseStamped,
            "/right_ee_pose",
            self.right_ee_pose_callback,
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
        t3[1, 3] = -self.config['bottom_to_right_ee']-self.config['above_dist']

        obj_T_above_target = rot @ t3
        return obj_T_above_target

    def left_ee_pose_callback(self, msg):
        self.base_T_left_ee = pose_2_tf_mat(msg.pose)
        self.left_ee_update_time = self.get_clock().now()
    
    def right_ee_pose_callback(self, msg):
        self.base_T_right_ee = pose_2_tf_mat(msg.pose)
        self.right_ee_update_time = self.get_clock().now()

    def get_base_T_ee(self):
        self.get_logger().info("Waiting for new end-effector pose...")

        request_time = self.get_clock().now()
        while (
            self.left_ee_update_time is None
            or self.right_ee_update_time is None
            or self.left_ee_update_time <= request_time
            or self.right_ee_update_time <= request_time
        ):
            rclpy.spin_once(self, timeout_sec=0.01)
        
        self.get_logger().info(f"Got Left EE pose:\n{self.base_T_left_ee}")
        self.get_logger().info(f"Got Right EE pose:\n{self.base_T_right_ee}")

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

    def pose_prepare(self):
        theta = 5 / 180 * np.pi
        x = 0.12
        y = 0.20
        z = 0.04
        delta_l = [ 0.10, -0.04,  0.00]
        delta_r = [-0.02, -0.03, -0.03]
        
        rotate_x_right = np.eye(4)
        rotate_x_right[1:3, 1:3] = [
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ]
        left_mat = np.array([
            [ 0.0,  0.0, 1.0,  x+delta_l[0]],
            [ 0.0, -1.0, 0.0, +y+delta_l[1]],
            [ 1.0,  0.0, 0.0,  z+delta_l[2]],
            [ 0.0,  0.0, 0.0,  1],
        ])
        right_mat = np.array([
            [ 0.0,  0.0,  1.0,  x+delta_r[0]],
            [ 0.0,  1.0,  0.0, -y+delta_r[1]],
            [-1.0,  0.0,  0.0,  z+delta_r[2]],
            [ 0.0,  0.0,  0.0,  1],
        ])

        test_left_pose = mat_2_pose_array(left_mat)
        test_right_pose = mat_2_pose_array(right_mat @ rotate_x_right)
        return self.move_p(test_left_pose, test_right_pose)
    
    def joint_prepare(self):
        left_traj = self.config['prepare_traj']['left']
        right_traj = self.config['prepare_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self.move_j(left_joint_states, right_joint_states):
                return False
        time.sleep(2)
        return True

    def move_above(self):
        self.get_base_T_ee()

        if not self.get_cam_T_obj():
            return False

        base_T_above_target = self.base_T_right_ee @ self.right_ee_T_cam @ self.cam_T_obj @ self.obj_T_above_target
        return self.move_p([], mat_2_pose_array(base_T_above_target))

    def approach_forward(self):
        # Turn on vacuum
        if not self.gripper.turn_on('right'):
            return False

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

        # Get current EE pose and create a large target along local +Y
        self.get_base_T_ee()
        right_ee_T_approach = np.eye(4)
        right_ee_T_approach[1, 3] = self.config['max_approach_dist']
        target_right_pose = self.base_T_right_ee @ right_ee_T_approach
        self.get_logger().info(
            f"Starting suction approach: "
            f"{self.config['max_approach_dist'] * 1000:.1f} mm along local +Y"
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
        goal.left_pose = []
        goal.right_pose = mat_2_pose_array(target_right_pose)
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
    
    def peel_off(self):
        # Peel Trajectory
        l_ee_T_l_peel_target = np.eye(4)
        r_ee_T_r_peel_target = np.eye(4)

        l_ee_T_l_peel_target[:3, 3] = self.config['withdraw_dist_l']
        r_ee_T_r_peel_target[:3, 3] = self.config['withdraw_dist_r']
        theta = self.config['withdraw_theta'] / 180 * np.pi
        l_ee_T_l_peel_target[:2, :2] = [
            [ np.cos(theta),  np.sin(theta)],
            [-np.sin(theta),  np.cos(theta)],
        ]
        r_ee_T_r_peel_target[:2, :2] = [
            [ np.cos(theta), -np.sin(theta)],
            [ np.sin(theta),  np.cos(theta)],
        ]
        
        self.get_base_T_ee()
        base_T_left_peel_target = self.base_T_left_ee @ l_ee_T_l_peel_target
        base_T_right_peel_target = self.base_T_right_ee @ r_ee_T_r_peel_target
        if not self.move_p(
            mat_2_pose_array(base_T_left_peel_target),
            mat_2_pose_array(base_T_right_peel_target)):
            return False

        self.get_logger().info("Peel off Success")
        return True

    def detach(self):
        return self.gripper.turn_off('right')

    def run(self):
        if not self.joint_prepare():
            return False

        if not self.move_above():
            return False

        if not self.approach_forward():
            return False

        if not self.peel_off():
            return False

        return self.detach()