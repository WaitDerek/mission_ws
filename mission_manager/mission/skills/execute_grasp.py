import os
import time
import json
import threading
import numpy as np
from .utils import *
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecuteGrasp

from rclpy.action import (
    ActionClient,
    ActionServer,
    CancelResponse,
    GoalResponse
)

GRASP_BADGE_CONFIG = 'grasp_badge_config.json'
GRASP_CONNECTOR_CONFIG = 'grasp_connector_config.json'


class ExecutorGrasp:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):

        self.badge_config_path = os.path.join(config_dir, GRASP_BADGE_CONFIG)
        with open(self.badge_config_path, 'r') as f:
            self.badge_config = json.load(f)

        self.connector_config_path = os.path.join(config_dir, GRASP_CONNECTOR_CONFIG)
        with open(self.connector_config_path, 'r') as f:
                self.connector_config = json.load(f)

        self.current_config = None
        
        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.wait_server_timeout = 5.0
        self.wait_accept_timeout = 5.0
        self.execution_timeout = 100.0
        
        self.callback_group = ReentrantCallbackGroup()

        self._grasp_client = ActionClient(
            self._node,
            ExecuteGrasp,
            '/execute_grasp',
            callback_group=self.callback_group,
        )

        self._action_server = ActionServer(
            node,
            ExecuteGrasp,
            "execute_grasp",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        node.get_logger().info(
            "/execute_grasp action server started"
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
        Accept an ExecuteGrasp goal.

        The action has no goal fields, so every goal request
        represents a request to start the grasp workflow.
        """

        self._node.get_logger().info(
            "Received ExecuteGrasp goal"
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept cancellation requests.
        """

        self._node.get_logger().info(
            "Received ExecuteGrasp cancellation request"
        )

        return CancelResponse.ACCEPT

    def make_result(self, success, message):
        result = ExecuteGrasp.Result()
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
        Publish ExecuteGrasp feedback.
        """
        feedback = ExecuteGrasp.Feedback()

        feedback.stage = stage
        feedback.detail = detail
    
        goal_handle.publish_feedback(feedback)

    def execute_callback(self, goal_handle):
        """
        Execute the badge tracking / grasp workflow.
        """
        model_label = goal_handle.request.model_label
        if_update_config = goal_handle.request.if_update_config        
                
        if if_update_config and model_label == 'badge':
            with open(self.badge_config_path, 'r') as f:
                self.badge_config = json.load(f)

        if if_update_config and model_label == 'badge_connector':
            with open(self.connector_config_path, 'r') as f:
                self.connector_config = json.load(f)

        if model_label == 'badge_connector':
            self.current_config = self.connector_config
            self._node.get_logger().info('current config: badge_connector')
        else:
            self.current_config = self.badge_config
            self._node.get_logger().info('current config: badge')
        
        self._node.get_logger().info("Start Grasping Car Badge")
        self.publish_feedback(
            goal_handle,
            stage="start",
            detail="Start Grasping Car Badge",
        )

        movement_i = 0
        
        if self.current_config['movement_switch'][movement_i][0]:
            if not self.prepare():
                goal_handle.abort()
                return self.make_result(False, "Grasp Prepare Failed")
        movement_i += 1

        if self.current_config['movement_switch'][movement_i][0]:
            if not self.move_above():
                goal_handle.abort()
                return self.make_result(False, "Move Above Failed")
        movement_i += 1

        if self.current_config['movement_switch'][movement_i][0]:
            if not self.suction_move_t():
                goal_handle.abort()
                return self.make_result(False, "Suction Failed")
        movement_i += 1

        if self.current_config['movement_switch'][movement_i][0]:
            if not self.withdraw_move_t():
                goal_handle.abort()
                return self.make_result(False, "Withdraw Failed")
        movement_i += 1

        if self.current_config['movement_switch'][movement_i][0]:
            if not self.ending():
                goal_handle.abort()
                return self.make_result(False, "Ending Failed") 

        goal_handle.succeed()
        self._node.get_logger().info("Grasp Succeeds")
        self.publish_feedback(
            goal_handle,
            stage="end",
            detail="Grasp Succeeds",
        )
        return self.make_result(True, "Grasp Succeeds") 
    
    def cancel_grasp(self, goal_handle):
        """
        Cancel the current grasp workflow.
        """
        goal_handle.canceled()


        result = ExecuteGrasp.Result()
        result.success = False
        result.message = "Grasp workflow canceled"

        self._node.get_logger().warn(
            "ExecuteGrasp workflow canceled"
        )
        return result

    def destroy(self):
        self._action_server.destroy()
     
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
        rot = np.array(self.current_config['offset_matrix']) @ t1 @ t2

        t3 = np.eye(4)
        t3[1, 3] = -self.current_config['bottom_to_left_ee']-self.current_config['above_dist']
        obj_T_above_target = rot @ t3
        return obj_T_above_target

    def prepare(self):
        if not self._robot.set_torso_height(self.current_config['torso_height']):
            return False

        left_traj = self.current_config['prepare_traj']['left']
        right_traj = self.current_config['prepare_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False

        time.sleep(2.0)
        return True

    def move_above(self):

        left_ee_T_cam = dict_2_tf_mat(self.current_config['left_ee_T_cam'])
        obj_T_above_target = self.get_obj_T_target()

        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        cam_T_obj = self._robot.get_cam_T_obj(self.current_config['model_label'])
        if cam_T_obj is None :
            return False

        base_T_obj = base_T_left_ee @ left_ee_T_cam @ cam_T_obj
        base_T_down_tar = base_T_obj @ obj_T_above_target
        down_target_pose = (mat_2_pose_array(base_T_down_tar))
        self._node.get_logger().info(
            f"Down target pose:\n{base_T_down_tar}"
        )

        return self._robot.move_p(down_target_pose, [])

    def suction_move_l(self):
        # Turn on vacuum
        if not self._gripper.turn_on('left'):
            return False

        # Get forward pose
        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        left_ee_T_approach = np.eye(4)
        approach_distance = self.current_config['max_approach_distance']
        left_ee_T_approach[1, 3] = approach_distance
        forward_target_pose = base_T_left_ee @ left_ee_T_approach

        self._node.get_logger().info(
            f"Starting approaching: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )
        return self._robot.move_l_force_feedback(
            force_threshold = self.current_config['suction_force_threshold'],
            left_pose = mat_2_pose_array(forward_target_pose),
        )

    def suction_move_t(self):
        
        # Turn on vacuum
        if not self._gripper.turn_on('left'):
            return False

        left_pose = [0.0] * 6
        approach_distance = self.current_config['max_approach_distance']
        left_pose[1] = approach_distance
        self._node.get_logger().info(
            f"Left gripper starts approaching: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )

        return self._robot.move_t_force(
            force_threshold=self.current_config['suction_force_threshold'],
            left_pose=left_pose,
            right_pose=None,
            move_timeout=30.0,
            speed=0.4,
            velocity=0.1,
            disable_environment_collision=True,
        )
    
    def withdraw(self):
        # Get backward pose
        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        left_ee_T_backward = np.eye(4)
        backward_dist = self.current_config['backward_dist']
        left_ee_T_backward[1, 3] = -backward_dist
        backward_target_pose = base_T_left_ee @ left_ee_T_backward

        self._node.get_logger().info(
            f"Starting moving backward: "
            f"{backward_dist * 1000:.1f} mm along local +Y"
        )
        return self._robot.move_l(
            left_pose = mat_2_pose_array(backward_target_pose),
        )

    def withdraw_move_t(self):        
        left_pose = [0.0] * 6
        backward_dist = self.current_config['backward_dist']
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
        left_traj = self.current_config['ending_joint_states']['left']
        right_traj = self.current_config['ending_joint_states']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False
        
        return True

    def client_request(self, workflow_goal_handle, **kwargs):

        model_label = kwargs.get('model_label', None)
        if (not model_label or \
            model_label not in ('badge', 'badge_connector')
        ):
            self._node.get_logger().error(
                'Require model_label for grasping'
            )
            return False
    
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
        goal_msg.model_label = model_label
        goal_msg.if_update_config = kwargs.get('if_update_config', True)

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

