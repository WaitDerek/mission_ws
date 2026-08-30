#!/usr/bin/env python3
import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation
from task_interfaces.action import MoveArmPose, MoveArmJoints

"""
This script helps identify the x-, y-, and z-axes of the end effector.

Adjust the translation values in LEFT_EE_T_TARGET and observe the direction of the end-effector's motion.

The movement reveals the orientation of each axis.

  
when the joint configuration is set by INIT_JOINT_STATES:
  For the left_gripper_base_link:
    The positive x-axis points toward the torso link;
    The positive y-axis points outward from the wrist link and vertically downward toward the ground;
    The positive z-axis points horizontally from the back of the robot toward the front.

  For the right_gripper_base_link:
    The positive x-axis points from the torso link to outside;
    The positive y-axis points outward from the wrist link and vertically downward toward the ground;
    The positive z-axis points horizontally from the back of the robot toward the front.
"""

IF_PREPARE = 0

TARGET_POSE = {
    'left':[
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, 0.00],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ],
    'right':[
        [1.00, 0.00, 0.00, 0.00],
        [0.00, 1.00, 0.00, -0.04],
        [0.00, 0.00, 1.00, 0.00],
        [0.00, 0.00, 0.00, 1.00],
    ]
}

INIT_JOINT_STATES = {
    'left': [
       0.4629625976085663,
        0.2690458595752716,
        0.03353186324238777,
        0.2836785912513733,
        0.0665244311094284,
        0.8396867513656616,
        -0.31481361389160156,
    ],
    'right': [
        0.2632335126399994,
        -0.2975803017616272,
        0.14027535915374756,
        0.16541826725006104,
        0.09430386871099472,
        1.0945911407470703,
        0.2151767611503601,
    ]
}


class MoveEE(Node):

    def __init__(self):
        super().__init__('move_ee')
        self.if_prepare = IF_PREPARE

        self.init_left_joint_states = INIT_JOINT_STATES['left']
        self.init_right_joint_states = INIT_JOINT_STATES['right']

        self.base_T_left_ee = None
        self.base_T_right_ee = None

        self.left_ee_T_tar = np.array(TARGET_POSE['left'])
        self.right_ee_T_tar = np.array(TARGET_POSE['right'])
    
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

        self.move_p_client = ActionClient(
            self,
            MoveArmPose,
            "/move_arm_p"
        )

        self.move_j_client = ActionClient(
            self,
            MoveArmJoints,
            '/move_arm_j'
        )

    def pose_to_matrix(self, pose):
        T = np.eye(4)
        q = [
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w
        ]
        T[:3,:3] = Rotation.from_quat(q).as_matrix()
        T[:3,3] = [
            pose.position.x,
            pose.position.y,
            pose.position.z
        ]
        return T

    def matrix_to_pose_array(self, T):
        xyz = T[:3,3]
        quat = Rotation.from_matrix(
            T[:3,:3]
        ).as_quat()
        return [
            float(xyz[0]),
            float(xyz[1]),
            float(xyz[2]),
            float(quat[0]),
            float(quat[1]),
            float(quat[2]),
            float(quat[3])
        ]

    def prepare(self):
        if not self.if_prepare:
            return True
        
        def feedback_callback(feedback_msg):
            feedback = feedback_msg.feedback
            self.get_logger().info(
                "move_arm_j feedback: "
                f"stage={feedback.stage}, progress={feedback.progress:.2f}, "
                f"detail={feedback.detail}"
            )
        
        self.get_logger().info("Waiting for action server: /move_arm_j")
        if not self.move_j_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("/move_arm_j server not available")
            return False
        
        goal = MoveArmJoints.Goal()
        goal.left_joints = self.init_left_joint_states
        goal.right_joints = self.init_right_joint_states
        goal.dry_run = False
        goal.duration = 0.0
    
        
        send_future = self.move_j_client.send_goal_async(goal, feedback_callback=feedback_callback)
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

    def left_ee_pose_callback(self, msg):
        self.base_T_left_ee = self.pose_to_matrix(msg.pose)
    
    def right_ee_pose_callback(self, msg):
        self.base_T_right_ee = self.pose_to_matrix(msg.pose)

    def get_base_T_ee(self):
        self.get_logger().info("Waiting for end-effector pose...")

        while self.base_T_left_ee is None or self.base_T_right_ee is None:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.get_logger().info("Got End EE pose")   

    def cal_base_T_target(self):
        
        base_T_left_tar = self.base_T_left_ee @ self.left_ee_T_tar
        base_T_right_tar = self.base_T_right_ee @ self.right_ee_T_tar

        left_target_pose = (self.matrix_to_pose_array(base_T_left_tar))
        right_target_pose = (self.matrix_to_pose_array(base_T_right_tar))

        self.get_logger().info(
            f"Left Target pose:\n{left_target_pose}"
        )
        self.get_logger().info(
            f"Right Target pose:\n{right_target_pose}"
        )
        return left_target_pose, right_target_pose

    def move_to_target(self, left_target_pose, right_target_pose):
        if not self.move_p_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("move_arm_p server not available")
            return

        move_goal = MoveArmPose.Goal()
        move_goal.left_pose = left_target_pose
        move_goal.right_pose = right_target_pose
        move_goal.dry_run = False

        send_future = self.move_p_client.send_goal_async(move_goal)
        rclpy.spin_until_future_complete(self, send_future)
        handle = send_future.result()

        if not handle.accepted:
            self.get_logger().error("/move_arm_p goal rejected")
            return

        self.get_logger().info("/move_arm_p goal accepted")
        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        self.get_logger().info(
            f"/move_arm_p result: {result.result}"
        )

    def run(self):
        if not self.prepare():
            return
        
        self.get_base_T_ee()
        l_tar, r_tar = self.cal_base_T_target()
        self.move_to_target(l_tar, r_tar)


def main():
    rclpy.init()
    node = MoveEE()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
