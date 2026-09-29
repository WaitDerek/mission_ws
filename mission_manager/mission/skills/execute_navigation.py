import os
import json
import threading
import numpy as np
from .utils import *
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import NavigateToPoint

NAVIGATION_CONFIG_FILE = 'navigation_config.json'


class ExecutorNavigation:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):

        self.config_path = os.path.join(config_dir, NAVIGATION_CONFIG_FILE)

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.wait_server_timeout = 5.0
        self.wait_accept_timeout = 5.0
        self.navigation_timeout = 100.0
        
        self._cal_parameter()
        self.callback_group = ReentrantCallbackGroup()

        self._navigate_client = ActionClient(
            self._node,
            NavigateToPoint,
            '/navigate_to_point',
            callback_group=self.callback_group,
        )

        node.get_logger().info(
            "/navigate_to_point client ready"
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

    def client_request(self, workflow_goal_handle, **kwargs):

        navi_goal = kwargs.get('navi_goal', False)
        if not navi_goal:
            self._node.get_logger().error(
                'Can not get navigation goal'
            )
            return False

        self._node.get_logger().info(
            'Waiting for /navigate_to_point action server...'
        )

        if not self._navigate_client.wait_for_server(
            timeout_sec=self.wait_server_timeout
        ):
            self._node.get_logger().error(
                '/navigate_to_point action server not available'
            )

            return False

        # ---------------------------------------------------------
        # Create goal
        # ---------------------------------------------------------

        goal_msg = NavigateToPoint.Goal()
        goal_msg.request_id = navi_goal['request_id']
        goal_msg.start = navi_goal['start']
        goal_msg.point_id = navi_goal['point_id']
        goal_msg.use_custom_pos = navi_goal['use_custom_pos']
        goal_msg.pos = navi_goal['pos']
        goal_msg.dry_run = navi_goal['dry_run']

        # ---------------------------------------------------------
        # Send goal
        # ---------------------------------------------------------

        self._node.get_logger().info(
            f"Sending /navigate_to_point {navi_goal['request_id']}"
        )

        send_goal_future = self._navigate_client.send_goal_async(
            goal_msg,
            feedback_callback=self.navigate_feedback_callback,
        )

        goal_handle = self._wait_for_future(
            send_goal_future,
            timeout=self.wait_accept_timeout,
        )

        if goal_handle is None:

            self._node.get_logger().error(
                f'Timeout or exception while sending '
                f"/navigate_to_point {navi_goal['request_id']}"
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                f"/navigate_to_point {navi_goal['request_id']} rejected"
            )

            return False

        self._node.get_logger().info(
            f"/navigate_to_point {navi_goal['request_id']} accepted"
        )

        # ---------------------------------------------------------
        # Wait for result
        # ---------------------------------------------------------

        result_future = goal_handle.get_result_async()

        result_response = self._wait_for_future(
            result_future,
            timeout=self.navigation_timeout,
        )

        if result_response is None:

            self._node.get_logger().error(
                f'Timeout or exception while waiting for '
                f"/navigate_to_point {navi_goal['request_id']} result"
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info(f"/navigate_to_point {navi_goal['request_id']} completed")
            return True
        else:
            self._node.get_logger().error(f"/navigate_to_point {navi_goal['request_id']} Failed")
            self._node.get_logger().error(result.message)
            return False

    def navigate_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/navigate_to_point feedback: {feedback}'
        )

