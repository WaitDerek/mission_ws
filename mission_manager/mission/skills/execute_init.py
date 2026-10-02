import os
import time
import json
import threading
import numpy as np
from .utils import *
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecuteInitialization
from rclpy.action import (
    ActionClient,
    ActionServer, 
    CancelResponse, 
    GoalResponse
)

INIT_CONFIG = 'init_config.json'


class ExecutorInitialization:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):
        
        self.init_config_path = os.path.join(config_dir, INIT_CONFIG)
        with open(self.init_config_path, 'r') as f:
            self.config = json.load(f)

        self.config_dir = config_dir

        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.wait_server_timeout = 5.0
        self.wait_accept_timeout = 5.0
        self.execution_timeout = 100.0

        self.callback_group = ReentrantCallbackGroup()

        self._init_client = ActionClient(
            self._node,
            ExecuteInitialization,
            '/execute_init',
            callback_group=self.callback_group,
        )

        self._action_server = ActionServer(
            node,
            ExecuteInitialization,
            "execute_init",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        node.get_logger().info(
            "/execute_init action server started"
        )

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
        Accept an /execute_init  goal.

        The action has no goal fields, so every goal request
        represents a request to start the initialization.
        """

        self._node.get_logger().info(
            "Received /execute_init goal"
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept cancellation requests.
        """

        self._node.get_logger().info(
            "Received /execute_init cancellation request"
        )

        return CancelResponse.ACCEPT

    def make_result(self, success, message):
        result = ExecuteInitialization.Result()
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
        Publish /execute_init feedback.
        """
        feedback = ExecuteInitialization.Feedback()

        feedback.stage = stage
        feedback.detail = detail
    
        goal_handle.publish_feedback(feedback)

    def cancel_grasp(self, goal_handle):
        """
        Cancel the initialization.
        """
        goal_handle.canceled()
        return self.make_result(False, "Initialization canceled")

    def destroy(self):
        self._action_server.destroy()

    def execute_callback(self, goal_handle):
        """
        Execute the initialization.
        """

        self._node.get_logger().info("Start initialization")
        movement_i = 0
        
        if self.config['movement_switch'][movement_i][0]:
            if not self.init_torso():
                goal_handle.abort()
                return self.make_result(False, "[Initialization] Initialize Torso Failed")
        movement_i += 1

        goal_handle.succeed()
        self._node.get_logger().info("Initialize Successfully")
        return self.make_result(True, "Initialize Successfully")
    
    def init_torso(self):
        return self._robot.initialize_torso()

    def client_request(self, workflow_goal_handle, **kwargs):
        
    
        self._node.get_logger().info(
            'Waiting for /execute_init action server...'
        )

        if not self._init_client.wait_for_server(
            timeout_sec=self.wait_server_timeout
        ):
            self._node.get_logger().error(
                '/execute_init action server not available'
            )

            return False

        # ---------------------------------------------------------
        # Create goal
        # ---------------------------------------------------------

        goal_msg = ExecuteInitialization.Goal()
        
        # ---------------------------------------------------------
        # Send goal
        # ---------------------------------------------------------

        self._node.get_logger().info(
            'Sending /execute_init goal'
        )

        send_goal_future = self._init_client.send_goal_async(
            goal_msg,
            feedback_callback=self.init_feedback_callback,
        )

        goal_handle = self._wait_for_future(
            send_goal_future,
            timeout=self.wait_accept_timeout,
        )

        if goal_handle is None:

            self._node.get_logger().error(
                'Timeout or exception while sending /execute_init goal'
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                '/execute_init goal rejected'
            )

            return False

        self._node.get_logger().info(
            '/execute_init goal accepted'
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
                '/execute_init result'
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info('/execute_init completed')
            return True
        else:
            self._node.get_logger().error(result.message)
            return False

    def init_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/execute_init feedback: {feedback}'
        )

