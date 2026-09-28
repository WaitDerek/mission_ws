import os
import time
import json
import threading
import numpy as np
from .utils import *
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecuteAssembly
from rclpy.action import (
    ActionClient,
    ActionServer, 
    CancelResponse, 
    GoalResponse
)

ASSEMBLY_CONFIG_FILE = 'assembly_config.json'


class ExecutorAssembly:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):

        self.config_path = os.path.join(config_dir, ASSEMBLY_CONFIG_FILE)

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.wait_server_timeout = 5.0
        self.wait_accept_timeout = 5.0
        self.execution_timeout = 100.0

        self._cal_parameter()

        self.callback_group = ReentrantCallbackGroup()

        self._assemble_client = ActionClient(
            self._node,
            ExecuteAssembly,
            '/execute_assembly',
            callback_group=self.callback_group,
        )

        self._action_server = ActionServer(
            node,
            ExecuteAssembly,
            "execute_assembly",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        node.get_logger().info(
            "/execute_assembly action server started"
        )

    def _cal_parameter(self):
        return

    def _wait_for_future(self, future, timeout=2.0):
        """
        Wait for a ROS future without spinning the node.

        The ROS executor must already be running elsewhere, normally
        through a MultiThreadedExecutor in MissionManager.
        """

        done_event = threading.Event()

        def on_done(_):
            done_event.set()

        future.add_done_callback(on_done)

        if not done_event.wait(timeout):
            self._node.get_logger().error(
                'Timeout waiting for ROS future'
            )

            return None

        try:
            return future.result()

        except Exception as e:
            self._node.get_logger().error(
                f'ROS future failed: {e}'
            )

            return None
        
    def goal_callback(self, goal_request):
        """
        Accept an /execute_assembly  goal.

        The action has no goal fields, so every goal request
        represents a request to start the grasp workflow.
        """

        self._node.get_logger().info(
            "Received /execute_assembly goal"
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept cancellation requests.
        """

        self._node.get_logger().info(
            "Received /execute_assembly cancellation request"
        )

        return CancelResponse.ACCEPT

    def make_result(self, success, message):
        result = ExecuteAssembly.Result()
        result.success = success
        result.message = message
        return result

    def publish_feedback(
            self,
            goal_handle,
            stage,
            detail,
        ):
        """
        Publish /execute_assembly feedback.
        """
        feedback = ExecuteAssembly.Feedback()

        feedback.stage = stage
        feedback.detail = detail
    
        goal_handle.publish_feedback(feedback)

    def cancel_grasp(self, goal_handle):
        """
        Cancel the current grasp workflow.
        """
        goal_handle.canceled()
        return self.make_result(False, "Assemble badge workflow canceled")

    def destroy(self):
        self._action_server.destroy()

    def execute_callback(self, goal_handle):
        """
        Execute the assemble workflow.
        """
        self._node.get_logger().info("Start assembling car badge")
        self.publish_feedback(
            goal_handle,
            stage="start",
            detail="start assembling car badge",
        )

        if self.config['refresh_config']:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                self._cal_parameter()

        movement_i = 0
        
        if self.config['movement_switch'][movement_i][0]:
            if not self.prepare():
                goal_handle.abort()
                return self.make_result(False, "[Assemble Car Badge] Joint Preparation Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.assemble_move_t():
                goal_handle.abort()
                return self.make_result(False, "[Assemble Car Badge] Move Forward Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.withdraw_move_t():
                goal_handle.abort()
                return self.make_result(False, "[Assemble Car Badge] Withdraw Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.ending():
                goal_handle.abort()
                return self.make_result(False, "[Assemble Car Badge] Ending Failed")
        movement_i += 1

        self.publish_feedback(
            goal_handle,
            stage="end",
            detail="Assemble Car Badge Succeeds",
        )

        goal_handle.succeed()
        return self.make_result(True, "Peel Back Film Succeeds")
    
    def prepare(self):
        if not self._robot.set_torso_height(self.config['torso_height']):
            return False
        
        left_traj = self.config['prepare_traj']['left']
        right_traj = self.config['prepare_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False
        time.sleep(2)
        return True

    def assemble_move_t(self):
        left_pose = [0.0] * 6
        approach_distance = self.config['max_approach_distance']
        left_pose[1] = approach_distance
        self._node.get_logger().info(
            f"Left gripper starts approaching: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )

        return self._robot.move_t_force(
            force_threshold=self.config['suction_force_threshold'],
            left_pose=left_pose,
            right_pose=None,
            move_timeout=30.0,
            speed=0.1,
            velocity=0.1,
            disable_environment_collision=True,
        )

    def withdraw_move_t(self):
        if not self._gripper.turn_off('left'):
            return False

        left_pose = [0.0] * 6
        backward_dist = self.config['backward_dist']
        left_pose[1] = -backward_dist
        self._node.get_logger().info(
            f"Starting moving backward: "
            f"{backward_dist * 1000:.1f} mm along local +Y"
        )

        return self._robot.move_t(
            left_pose=left_pose,
            right_pose=None,
            move_timeout=20.0,
            speed=1.0,
            velocity=0.5,
            disable_environment_collision=True
        )

    def ending(self):
        if not self._gripper.turn_off('right'):
            return False
        
        left_traj = self.config['ending_joint_states']['left']
        right_traj = self.config['ending_joint_states']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False

        return True

    def client_request(self, workflow_goal_handle, **kwargs):
    
        self._node.get_logger().info(
            'Waiting for /execute_assembly action server...'
        )

        if not self._assemble_client.wait_for_server(
            timeout_sec=self.wait_server_timeout
        ):
            self._node.get_logger().error(
                '/execute_assembly action server not available'
            )

            return False

        # ---------------------------------------------------------
        # Create goal
        # ---------------------------------------------------------

        goal_msg = ExecuteAssembly.Goal()

        # Fill goal fields here if ExecuteGrasp has any.
        #
        # Example:
        # goal_msg.object_id = ...

        # ---------------------------------------------------------
        # Send goal
        # ---------------------------------------------------------

        self._node.get_logger().info(
            'Sending /execute_assembly goal'
        )

        send_goal_future = self._assemble_client.send_goal_async(
            goal_msg,
            feedback_callback=self.assembly_feedback_callback,
        )

        goal_handle = self._wait_for_future(
            send_goal_future,
            timeout=self.wait_accept_timeout,
        )

        if goal_handle is None:

            self._node.get_logger().error(
                'Timeout or exception while sending /execute_assembly goal'
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                '/execute_assembly goal rejected'
            )

            return False

        self._node.get_logger().info(
            '/execute_assembly goal accepted'
        )

        # ---------------------------------------------------------
        # Wait for result
        # ---------------------------------------------------------

        result_future = goal_handle.get_result_async()

        result_response = self._wait_for_future(
            result_future,
            timeout=self.execution_timeout,
        )

        if result_response is None:

            self._node.get_logger().error(
                'Timeout or exception while waiting for '
                '/execute_assembly result'
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info('/execute_assembly completed')
            return True
        else:
            self._node.get_logger().error(result.message)
            return False

    def assembly_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/execute_assembly feedback: {feedback}'
        )

