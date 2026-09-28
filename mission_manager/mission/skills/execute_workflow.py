import os
import json
import time
import threading

from rclpy.action import (
    ActionServer,
    CancelResponse,
    GoalResponse,
)

from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecuteWorkflow

WORKFLOW_CONFIG_FILE = 'workflow_config.json'


class ExecutorWorkflow:

    def __init__(
            self, 
            node,
            config_dir,
            executor_navigation,
            executor_grasp,
            executor_peel,
            executor_assembly,
        ):

        self._node = node

        self.config_path = os.path.join(
            config_dir,
            WORKFLOW_CONFIG_FILE,
        )

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self.wait_server_timeout = 5.0
        self.wait_accept_timeout = 5.0
        self.execution_timeout = 100.0
        self.navigation_timeout = 100.0

        self.executor_mapping = {
            'ExecutorNavigation': executor_navigation,
            'ExecutorGrasp': executor_grasp,
            'ExecutorPeel': executor_peel,
            'ExecutorAssembly': executor_assembly,
        }

        self.callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            self._node,
            ExecuteWorkflow,
            '/execute_workflow',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        self._node.get_logger().info(
            'ExecuteWorkflow action server started'
        )

    # =============================================================
    # ExecuteWorkflow action callbacks
    # =============================================================

    def goal_callback(self, goal_request):

        self._node.get_logger().info(
            'Received ExecuteWorkflow goal'
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):

        self._node.get_logger().info(
            'Received ExecuteWorkflow cancel request'
        )

        return CancelResponse.ACCEPT

    def make_workflow_result(self, success, msg):
        result = ExecuteWorkflow.Result()
        result.success = success
        result.message = msg
        return result

    def execute_callback(self, goal_handle):

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        for task in config['workflow']:
         
            if not task['if_exec']:
                continue

            if not self.executor_mapping[task['executor']].client_request(
                workflow_goal_handle=goal_handle,
                **task['params']
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f"task: [{task['description']}] failed"
                )

        self._node.get_logger().info('Workflow completed successfully')
        
        goal_handle.succeed()
        return self.make_workflow_result(
            True, 'Workflow completed successfully'
        )
