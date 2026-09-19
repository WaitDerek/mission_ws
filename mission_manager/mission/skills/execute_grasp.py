import os
import time
import json
import numpy as np
from .utils import *
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecuteGrasp
from rclpy.action import ActionServer, CancelResponse, GoalResponse

GRIP_CONFIG_FILE = 'grasp_config.json'


class ExecuteGraspServer:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):

        self.config_path = os.path.join(config_dir, GRIP_CONFIG_FILE)

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.fake_camera_pose = None
        self.left_ee_T_cam = None
        self.obj_T_above_target = None
        self._cal_parameter()
        
        self.callback_group = ReentrantCallbackGroup()

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

    def _cal_parameter(self):
        self.fake_camera_pose = self.config['fake_camera_pose']
        self.left_ee_T_cam = dict_2_tf_mat(self.config['left_ee_T_cam'])
        self.obj_T_above_target = self.get_obj_T_target()

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
        self._node.get_logger().info("Start Grasping Car Badge")
        self.publish_feedback(
            goal_handle,
            stage="start",
            detail="Start Grasping Car Badge",
        )

        if self.config['refresh_config']:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                self._cal_parameter()

        movement_i = 0
        
        if self.config['movement_switch'][movement_i][0]:
            if not self.prepare():
                goal_handle.abort()
                return self.make_result(False, "Grasp Prepare Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.move_above():
                goal_handle.abort()
                return self.make_result(False, "Move Above Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.suction_move_t():
                goal_handle.abort()
                return self.make_result(False, "Suction Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.withdraw_move_t():
                goal_handle.abort()
                return self.make_result(False, "Withdraw Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.ending():
                goal_handle.abort()
                return self.make_result(False, "Ending Failed") 

        self.publish_feedback(
            goal_handle,
            stage="end",
            detail="Badge Grasp Succeeds",
        )

        goal_handle.succeed()
        return self.make_result(True, "Badge Grasp Succeeds") 
    
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
        rot = np.array(self.config['offset_matrix']) @ t1 @ t2

        t3 = np.eye(4)
        t3[1, 3] = -self.config['bottom_to_left_ee']-self.config['above_dist']
        obj_T_above_target = rot @ t3
        return obj_T_above_target

    def prepare(self):
        if not self._robot.set_torso_height(self.config['torso_height']):
            return False

        left_traj = self.config['prepare_traj']['left']
        right_traj = self.config['prepare_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False

        time.sleep(2.0)
        return True

    def move_above(self):
        if self.fake_camera_pose:
            down_target_pose = dict_2_pose_array(
                self.config['fake_pose']['down_target_pose']
            )
            return self._robot.move_p(down_target_pose, [])
        
        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        cam_T_obj = self._robot.get_cam_T_obj(self.config['model_label'])
        if cam_T_obj is None :
            return False

        base_T_obj = base_T_left_ee @ self.left_ee_T_cam @ cam_T_obj
        base_T_down_tar = base_T_obj @ self.obj_T_above_target
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
        approach_distance = self.config['max_approach_distance']
        left_ee_T_approach[1, 3] = approach_distance
        forward_target_pose = base_T_left_ee @ left_ee_T_approach

        self._node.get_logger().info(
            f"Starting approaching: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )
        return self._robot.move_l_force_feedback(
            force_threshold = self.config['suction_force_threshold'],
            left_pose = mat_2_pose_array(forward_target_pose),
        )

    def suction_move_t(self):
        
        # Turn on vacuum
        if not self._gripper.turn_on('left'):
            return False

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
        backward_dist = self.config['backward_dist']
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
        left_traj = self.config['ending_joint_states']['left']
        right_traj = self.config['ending_joint_states']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False
        
        return True

