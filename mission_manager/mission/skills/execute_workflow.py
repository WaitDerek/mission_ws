import os
import json
import time
import threading

from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse,
)

from mission_manager_interfaces.action import (
    ExecuteGrasp,
    ExecutePeel,
    ExecuteWorkflow,
    ExecuteAssembly,
    NavigateToPoint,
)

from rclpy.callback_groups import ReentrantCallbackGroup


WORKFLOW_CONFIG_FILE = 'workflow_config.json'


class ExecuteWorkflowServer:

    def __init__(self, node, config_dir):

        # =========================================================
        # Basic members
        # =========================================================

        self._node = node

        # =========================================================
        # Load workflow configuration
        # =========================================================

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


        # =========================================================
        # Callback group
        # =========================================================

        self.callback_group = ReentrantCallbackGroup()

        # =========================================================
        # ExecuteWorkflow action server
        # =========================================================

        self._action_server = ActionServer(
            self._node,
            ExecuteWorkflow,
            '/execute_workflow',
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        # =========================================================
        # ExecuteGrasp action client
        # =========================================================

        self._grasp_client = ActionClient(
            self._node,
            ExecuteGrasp,
            '/execute_grasp',
            callback_group=self.callback_group,
        )

        # =========================================================
        # ExecutePeel action client
        # =========================================================

        self._peel_client = ActionClient(
            self._node,
            ExecutePeel,
            '/execute_peel',
            callback_group=self.callback_group,
        )

        # =========================================================
        # Assemble client
        # =========================================================

        self._assemble_client = ActionClient(
            self._node,
            ExecuteAssembly,
            '/execute_assembly',
            callback_group=self.callback_group,
        )

        # =========================================================
        # NavigateToPoint action client
        # =========================================================

        self._navigate_client = ActionClient(
            self._node,
            NavigateToPoint,
            '/navigate_to_point',
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

    # =============================================================
    # ExecuteGrasp
    # =============================================================

    def execute_grasp(self, workflow_goal_handle):

        self._node.get_logger().info(
            'Waiting for /execute_grasp action server...'
        )

        if not self._grasp_client.wait_for_server(
            timeout_sec=self.wait_server_timeout
        ):
            self._node.get_logger().error(
                '/execute_grasp action server not available'
            )

            return False

        # ---------------------------------------------------------
        # Create goal
        # ---------------------------------------------------------

        goal_msg = ExecuteGrasp.Goal()

        # Fill goal fields here if ExecuteGrasp has any.
        #
        # Example:
        # goal_msg.object_id = ...

        # ---------------------------------------------------------
        # Send goal
        # ---------------------------------------------------------

        self._node.get_logger().info(
            'Sending /execute_grasp goal'
        )

        send_goal_future = self._grasp_client.send_goal_async(
            goal_msg,
            feedback_callback=self.grasp_feedback_callback,
        )

        goal_handle = self._wait_for_future(
            send_goal_future,
            timeout=self.wait_accept_timeout,
        )

        if goal_handle is None:

            self._node.get_logger().error(
                'Timeout or exception while sending /execute_grasp goal'
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                '/execute_grasp goal rejected'
            )

            return False

        self._node.get_logger().info(
            '/execute_grasp goal accepted'
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
                '/execute_grasp result'
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info('/execute_grasp completed')
            return True
        else:
            self._node.get_logger().error(result.message)
            return False

    def grasp_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/execute_grasp feedback: {feedback}'
        )

    # =============================================================
    # ExecutePeel
    # =============================================================

    def execute_peel(self, workflow_goal_handle):

        self._node.get_logger().info(
            'Waiting for /execute_peel action server...'
        )

        if not self._peel_client.wait_for_server(
            timeout_sec=self.wait_server_timeout
        ):
            self._node.get_logger().error(
                '/execute_peel action server not available'
            )

            return False

        # ---------------------------------------------------------
        # Create goal
        # ---------------------------------------------------------

        goal_msg = ExecutePeel.Goal()

        # ExecutePeel has an empty goal in your current example:
        #
        # ros2 action send_goal --feedback /execute_peel \
        #     mission_manager_interfaces/action/ExecutePeel "{}"

        # ---------------------------------------------------------
        # Send goal
        # ---------------------------------------------------------

        self._node.get_logger().info(
            'Sending /execute_peel goal'
        )

        send_goal_future = self._peel_client.send_goal_async(
            goal_msg,
            feedback_callback=self.peel_feedback_callback,
        )

        goal_handle = self._wait_for_future(
            send_goal_future,
            timeout=self.wait_accept_timeout,
        )

        if goal_handle is None:

            self._node.get_logger().error(
                'Timeout or exception while sending /execute_peel goal'
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                '/execute_peel goal rejected'
            )

            return False

        self._node.get_logger().info(
            '/execute_peel goal accepted'
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
                '/execute_peel result'
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info('/execute_peel completed')
            return True
        else:
            self._node.get_logger().error(result.message)
            return False

    def peel_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/execute_peel feedback: {feedback}'
        )

    # =============================================================
    # ExecuteAssembly
    # =============================================================

    def execute_assembly(self, workflow_goal_handle):

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

    # =============================================================
    # NavigateToPoint
    # =============================================================

    def navigate_to_point(self, workflow_goal_handle, ponit_i, navi_goal):

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
            f'Sending /navigate_to_point goal-{ponit_i}'
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
                f'/navigate_to_point goal-{ponit_i}'
            )

            return False

        # ---------------------------------------------------------
        # Check acceptance
        # ---------------------------------------------------------

        if not goal_handle.accepted:

            self._node.get_logger().error(
                f'/navigate_to_point goal-{ponit_i} rejected'
            )

            return False

        self._node.get_logger().info(
            f'/navigate_to_point goal-{ponit_i} accepted'
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
                f'/navigate_to_point-{ponit_i} result'
            )

            return False

        result = result_response.result

        if result.success:
            self._node.get_logger().info(f'/navigate_to_point-{ponit_i} completed')
            return True
        else:
            self._node.get_logger().error(f'/navigate_to_point-{ponit_i} Failed')
            self._node.get_logger().error(result.message)
            return False

    def navigate_feedback_callback(self, feedback_msg):

        feedback = feedback_msg.feedback

        self._node.get_logger().info(
            f'/navigate_to_point feedback: {feedback}'
        )

    # =============================================================
    # Future helper
    # =============================================================

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

    def make_workflow_result(self, success, msg):
        result = ExecuteWorkflow.Result()
        result.success = success
        result.message = msg
        return result

    # =============================================================
    # ExecuteWorkflow
    # =============================================================

    def execute_callback(self, goal_handle):

        with open(self.config_path, 'r') as f:
            config = json.load(f)

        navi_point_i = 0
        task_step_i = 0
        self._node.get_logger().info(
            'Starting workflow'
        )

        # -------------------- 0. To Start Point --------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Navigate to point-{navi_point_i}'
            )

            if not self.navigate_to_point(
                goal_handle,
                navi_point_i, 
                config['navigation_points'][navi_point_i]
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f'Navigate to point-{navi_point_i} failed'
                )

        navi_point_i += 1
        task_step_i += 1
        # -----------------------------------------------------------


        # -------------------- 1. To Grasp Point --------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Navigate to point-{navi_point_i}'
            )
            if not self.navigate_to_point(
                goal_handle,
                navi_point_i, 
                config['navigation_points'][navi_point_i]
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f'Navigate to point-{navi_point_i} failed'
                )

        navi_point_i += 1
        task_step_i += 1
        # -----------------------------------------------------------


        # ------------------ 2. Execute grasp -----------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Execute grasp'
            )
            if not self.execute_grasp(goal_handle):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, 'Execute grasp failed'
                )
        
        task_step_i += 1
        # -----------------------------------------------------------


        # ------------------ 3. To Peel Point -----------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Navigate to point-{navi_point_i}'
            )
            if not self.navigate_to_point(
                goal_handle,
                navi_point_i, 
                config['navigation_points'][navi_point_i]
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f'Navigate to point-{navi_point_i} failed'
                )

        navi_point_i += 1
        task_step_i += 1
        # -----------------------------------------------------------


        # ----------------------- 4. Peel ---------------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Execute peel'
            )
            if not self.execute_peel(goal_handle):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, 'Execute peel failed'
                )
        task_step_i += 1
        # -----------------------------------------------------------

        # ----------------- 5. To Install Point ---------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Navigate to point-{navi_point_i}'
            )
            if not self.navigate_to_point(
                goal_handle,
                navi_point_i, 
                config['navigation_points'][navi_point_i]
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f'Navigate to point-{navi_point_i} failed'
                )

        navi_point_i += 1
        task_step_i += 1
        # -----------------------------------------------------------

        
        # --------------- 6. Install Car Badge ----------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Execute Assembly'
            )
            if not self.execute_assembly(goal_handle):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, 'Execute Assembly failed'
                )
            
        task_step_i += 1
        # -----------------------------------------------------------


        # -------------------- 7. To End Point ----------------------
        if config['flow_mask'][task_step_i][0]:
            self._node.get_logger().info(
                f'Step {task_step_i}: Navigate to point-{0}'
            )

            if not self.navigate_to_point(
                goal_handle,
                navi_point_i, 
                config['navigation_points'][0]
            ):
                goal_handle.abort()
                return self.make_workflow_result(
                    False, f'Navigate to point-{0} failed'
                )
        # -----------------------------------------------------------


        # ---------------- Workflow completed------------------------
        self._node.get_logger().info('Workflow completed successfully')
        goal_handle.succeed()
        result = ExecuteWorkflow.Result()
        result.success = True
        result.message = 'Workflow completed successfully'
        return result
