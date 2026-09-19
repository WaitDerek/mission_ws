import os
import time
import json
import numpy as np
from .utils import *
from rclpy.callback_groups import ReentrantCallbackGroup
from mission_manager_interfaces.action import ExecutePeel
from rclpy.action import ActionServer, CancelResponse, GoalResponse

PEEL_CONFIG_FILE = 'peel_config.json'


class ExecutePeelServer:
    def __init__(self, node, config_dir, gripper, force_sensor, robot):

        self.config_path = os.path.join(config_dir, PEEL_CONFIG_FILE)

        with open(self.config_path, 'r') as f:
            self.config = json.load(f)

        self._node = node
        self._gripper = gripper
        self._force_sensor = force_sensor
        self._robot = robot

        self.obj_T_above_target = None
        self.right_ee_T_cam = None
        self.fake_camera_pose = None
        self._cal_parameter()
                
        self.callback_group = ReentrantCallbackGroup()

        self._action_server = ActionServer(
            node,
            ExecutePeel,
            "execute_peel",
            execute_callback=self.execute_callback,
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            callback_group=self.callback_group,
        )

        node.get_logger().info(
            "/execute_peel action server started"
        )

    def _cal_parameter(self):
        self.fake_camera_pose = self.config['fake_camera_pose']
        self.obj_T_above_target = self.get_obj_T_target()
        self.right_ee_T_cam = dict_2_tf_mat(self.config['right_ee_T_cam'])
        

    def goal_callback(self, goal_request):
        """
        Accept an ExecutePeel goal.

        The action has no goal fields, so every goal request
        represents a request to start the grasp workflow.
        """

        self._node.get_logger().info(
            "Received ExecutePeel goal"
        )

        return GoalResponse.ACCEPT

    def cancel_callback(self, goal_handle):
        """
        Accept cancellation requests.
        """

        self._node.get_logger().info(
            "Received ExecutePeel cancellation request"
        )

        return CancelResponse.ACCEPT

    def make_result(self, success, message):
        result = ExecutePeel.Result()
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
        Publish ExecutePeel feedback.
        """
        feedback = ExecutePeel.Feedback()

        feedback.stage = stage
        feedback.detail = detail
    
        goal_handle.publish_feedback(feedback)

    def execute_callback(self, goal_handle):
        """
        Execute the back-film-peeling workflow.
        """
        self._node.get_logger().info("Start Peeling Back Film")
        self.publish_feedback(
            goal_handle,
            stage="start",
            detail="Start Peeling Back Film",
        )

        if self.config['refresh_config']:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
                self._cal_parameter()

        movement_i = 0
        
        if self.config['movement_switch'][movement_i][0]:
            if not self.joint_prepare():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Joint Preparation Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.align_right_camera():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Alignment Failed")
        movement_i += 1
        
        if self.config['movement_switch'][movement_i][0]:
            if not self.move_above():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Move above Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.approach_forward_move_t():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Approach Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.peel_off():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Peel Off Failed")
        movement_i += 1

        if self.config['movement_switch'][movement_i][0]:
            if not self.ending():
                goal_handle.abort()
                return self.make_result(False, "[Peel Film] Ending Failed")
        movement_i += 1

        self.publish_feedback(
            goal_handle,
            stage="end",
            detail="Peel Back Film Succeeds",
        )

        goal_handle.succeed()
        return self.make_result(True, "Peel Back Film Succeeds")
    
    def cancel_grasp(self, goal_handle):
        """
        Cancel the current grasp workflow.
        """
        goal_handle.canceled()
        return self.make_result(False, "Peel workflow canceled")

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
        t3[1, 3] = -self.config['bottom_to_right_ee']-self.config['above_dist']

        obj_T_above_target = rot @ t3
        return obj_T_above_target
    
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
        return self._robot.move_p(test_left_pose, test_right_pose)
    
    def joint_prepare(self):
        left_traj = self.config['prepare_traj']['left']
        right_traj = self.config['prepare_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False

        return True

    def align_right_camera(self):
        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        trans = np.array(self.config['camera_alignment_trans'])
        rot = np.array([
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ])
        left_ee_T_right_target = trans @ rot
        base_T_right_ee_target = base_T_left_ee @ left_ee_T_right_target

        return self._robot.move_p([], mat_2_pose_array(base_T_right_ee_target))

    def fixed_move_above(self):

        base_T_left_ee, _ = self._robot.get_base_T_ee()
        if base_T_left_ee is None:
            return False

        trans = np.array(self.config['arm_alignment_trans'])
        rot = np.array([
            [-1.0,  0.0,  0.0,  0.0],
            [ 0.0, -1.0,  0.0,  0.0],
            [ 0.0,  0.0,  1.0,  0.0],
            [ 0.0,  0.0,  0.0,  1.0],
        ])
        left_ee_T_right_target = trans @ rot
        base_T_right_ee_target = base_T_left_ee @ left_ee_T_right_target

        return self._robot.move_p([], mat_2_pose_array(base_T_right_ee_target))
    
    def move_above(self):
        time.sleep(3.0)

        if self.fake_camera_pose:
            return self.fixed_move_above()

        _, base_T_right_ee = self._robot.get_base_T_ee()
        if base_T_right_ee is None:
            return False

        base_T_above_target = None
        cam_T_obj = self._robot.get_cam_T_obj(self.config['model_label'])

        if cam_T_obj is None :
            return False

        base_T_above_target = base_T_right_ee @ self.right_ee_T_cam @ cam_T_obj @ self.obj_T_above_target
        return self._robot.move_p([], mat_2_pose_array(base_T_above_target))
            

    def approach_forward_move_l(self):
        # Turn on vacuum
        if not self._gripper.turn_on('right'):
            return False

        _, base_T_right_ee = self._robot.get_base_T_ee()
        if base_T_right_ee is None:
            return False
        
        # Get current EE pose and create a large target along local +Y
        right_ee_T_approach = np.eye(4)
        right_ee_T_approach[1, 3] = self.config['max_approach_dist']
        target_right_pose = base_T_right_ee @ right_ee_T_approach
        self._node.get_logger().info(
            f"Starting suction approach: "
            f"{self.config['max_approach_dist'] * 1000:.1f} mm along local +Y"
        )

        return self._robot.move_l(
            right_pose = mat_2_pose_array(target_right_pose),
        )

    def approach_forward_move_t(self): 
        # Turn on vacuum
        if not self._gripper.turn_on('right'):
            return False

        right_pose = [0.0] * 6
        approach_distance = self.config['max_approach_dist']
        right_pose[1] = approach_distance
        self._node.get_logger().info(
            f"Right gripper starts approaching: "
            f"{approach_distance * 1000:.1f} mm along local +Y"
        )

        return self._robot.move_t_force(
            force_threshold=self.config['suction_force_threshold'],
            left_pose=None,
            right_pose=right_pose,
            move_timeout=25.0,
            speed=0.1,
            velocity=0.1,
            disable_environment_collision=True
        )
    
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
        
        base_T_left_ee, base_T_right_ee = self._robot.get_base_T_ee()
        if base_T_left_ee is None or base_T_right_ee is None:
            return False

        base_T_left_peel_target = base_T_left_ee @ l_ee_T_l_peel_target
        base_T_right_peel_target = base_T_right_ee @ r_ee_T_r_peel_target
        if not self._robot.move_p(
            mat_2_pose_array(base_T_left_peel_target),
            mat_2_pose_array(base_T_right_peel_target),
        ):
            return False

        self._node.get_logger().info("Peel off Success")
        return True
    
    def ending(self):
        left_traj = self.config['end_traj']['left']
        right_traj = self.config['end_traj']['right']
        
        for left_joint_states, right_joint_states in zip(left_traj, right_traj):
            if not self._robot.move_j(left_joint_states, right_joint_states):
                return False

        return self._gripper.turn_off('right')
    