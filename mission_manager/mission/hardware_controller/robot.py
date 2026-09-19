import threading
import time

import numpy as np
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation as R

from task_interfaces.msg import G1dTorsoCommand, G1dTorsoState
from task_interfaces.action import MoveArmJoints, MoveArmPose
from object_pose_interfaces.action import EstimateObjectPose


def pose_2_tf_mat(pose):
    """
    Convert a geometry_msgs/Pose to a 4x4 homogeneous transformation matrix.
    """
    T = np.eye(4)

    q = [
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    ]

    T[:3, :3] = R.from_quat(q).as_matrix()
    T[:3, 3] = [
        pose.position.x,
        pose.position.y,
        pose.position.z,
    ]

    return T


class RobotController:

    def __init__(self, node, force_sensor):

        self._node = node
        self._sensor = force_sensor

        # ------------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------------

        self.server_timeout = 5.0
        self.wait_cancel_timeout = 20.0
        self.loop_interval = 0.01
        
        # ------------------------------------------------------------------
        # Thread synchronization
        # ------------------------------------------------------------------

        # Protects all shared state below.
        self.state_lock = threading.Lock()

        # ------------------------------------------------------------------
        # Shared state
        # ------------------------------------------------------------------

        self.torso_state = None
        self.torso_update_time = None

        self.left_ee_update_time = None
        self.right_ee_update_time = None

        self.base_T_left_ee = None
        self.base_T_right_ee = None

        # ------------------------------------------------------------------
        # ROS callback group
        # ------------------------------------------------------------------

        # RobotController methods can block while other callbacks belonging
        # to this controller continue to execute on other executor threads.
        self.callback_group = ReentrantCallbackGroup()

        # ------------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------------

        self.left_ee_pose_sub = self._node.create_subscription(
            PoseStamped,
            "/left_ee_pose",
            self.left_ee_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        self.right_ee_pose_sub = self._node.create_subscription(
            PoseStamped,
            "/right_ee_pose",
            self.right_ee_pose_callback,
            10,
            callback_group=self.callback_group,
        )

        self.torso_sub = self._node.create_subscription(
            G1dTorsoState,
            "/g1_d/torso/state",
            self.torso_state_callback,
            10,
            callback_group=self.callback_group,
        )

        # ------------------------------------------------------------------
        # Publisher
        # ------------------------------------------------------------------

        self.torso_pub = self._node.create_publisher(
            G1dTorsoCommand,
            "/g1_d/torso/command",
            10,
        )

        # ------------------------------------------------------------------
        # Action clients
        # ------------------------------------------------------------------

        self.object_client = ActionClient(
            self._node,
            EstimateObjectPose,
            "/object_pose/estimate",
            callback_group=self.callback_group,
        )

        self.move_j_client = ActionClient(
            self._node,
            MoveArmJoints,
            "/move_arm_j",
            callback_group=self.callback_group,
        )

        self.move_p_client = ActionClient(
            self._node,
            MoveArmPose,
            "/move_arm_p",
            callback_group=self.callback_group,
        )

        self.move_l_client = ActionClient(
            self._node,
            MoveArmPose,
            "/move_arm_l",
            callback_group=self.callback_group,
        )

        self.move_t_client = ActionClient(
            node,
            MoveArmPose,
            "/moveT",
            callback_group=self.callback_group,
        )

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _wait_for_future(self, future, timeout=2.0):
        """
        Wait for a ROS future without spinning the node.

        The ROS executor must already be running elsewhere, normally through
        a MultiThreadedExecutor in MissionManager.
        """

        done_event = threading.Event()

        def on_done(_):
            done_event.set()

        future.add_done_callback(on_done)
        return done_event.wait(timeout)

    # ======================================================================
    # ROS callbacks
    # ======================================================================

    def left_ee_pose_callback(self, msg):
        """
        Update the latest left end-effector pose.

        The pose and its timestamp are updated atomically with respect to
        RobotController readers.
        """

        pose = pose_2_tf_mat(msg.pose)
        update_time = self._node.get_clock().now()

        with self.state_lock:
            self.base_T_left_ee = pose
            self.left_ee_update_time = update_time

    def right_ee_pose_callback(self, msg):
        """
        Update the latest right end-effector pose.
        """

        pose = pose_2_tf_mat(msg.pose)
        update_time = self._node.get_clock().now()

        with self.state_lock:
            self.base_T_right_ee = pose
            self.right_ee_update_time = update_time

    def torso_state_callback(self, msg):
        """
        Update the latest torso state.

        The message and its update timestamp are stored together.
        """

        update_time = self._node.get_clock().now()

        with self.state_lock:
            self.torso_state = msg
            self.torso_update_time = update_time

    # ======================================================================
    # Torso
    # ======================================================================

    def set_torso_height(
        self,
        target_height,
        speed=0.3,
        tolerance=0.015,
        timeout=30.0,
    ):
        """
        Command the torso to move to target_height and wait until the torso
        reaches the target.

        Args:
            target_height: Target torso height in meters.
            speed: Torso movement speed.
            tolerance: Acceptable position error in meters.
            timeout: Maximum waiting time in seconds.

        Returns:
            True:
                Torso successfully reached target.

            False:
                Timeout or ROS shutdown.
        """

        # Clamp target height.
        target_height = min(float(target_height), 0.21)

        # Record the time immediately before publishing the command.
        # We use this to ensure that the state used for verification was
        # received after this command was issued.
        command_time = self._node.get_clock().now()

        # Publish torso command.
        msg = G1dTorsoCommand()
        msg.control_mode = 3
        msg.target_position = target_height
        msg.speed = speed

        self.torso_pub.publish(msg)

        self._node.get_logger().info(
            "Published torso command: "
            f"control_mode={msg.control_mode}, "
            f"target_position={msg.target_position}, "
            f"speed={msg.speed}"
        )

        # ------------------------------------------------------------------
        # Wait for a new torso state and then check whether the target has
        # been reached.
        # ------------------------------------------------------------------

        start_time = time.monotonic()

        while True:

            # Check local timeout.
            elapsed = time.monotonic() - start_time

            if elapsed >= timeout:
                with self.state_lock:
                    state = self.torso_state

                if state is not None:
                    current_height = state.column_height_m
                else:
                    current_height = None

                self._node.get_logger().error(
                    "Torso movement timed out. "
                    f"Current height={current_height}, "
                    f"target={target_height:.4f} m"
                )

                return True

            # Take a consistent snapshot of torso state.
            with self.state_lock:
                state = self.torso_state
                state_time = self.torso_update_time

            # No state received yet.
            if state is None or state_time is None:
                time.sleep(self.loop_interval)
                continue

            # Ignore state received before the command was published.
            if state_time <= command_time:
                time.sleep(self.loop_interval)
                continue

            # Read all fields from the same message snapshot.
            current_height = state.column_height_m
            velocity = state.column_velocity_mps

            height_error = abs(current_height - target_height)

            self._node.get_logger().info(
                f"Torso: height={current_height:.4f} m, "
                f"target={target_height:.4f} m, "
                f"error={height_error:.4f} m, "
                f"velocity={velocity:.4f} m/s"
            )

            if (
                height_error <= tolerance
                and abs(velocity) < 0.001
            ):
                self._node.get_logger().info(
                    "Torso reached target height: "
                    f"{current_height:.4f} m"
                )

                return True

            if (not state.column_target_active and abs(velocity) < 0.001):
                self._node.get_logger().error(
                    "Torso stopped before target: "
                    f"current={current_height:.4f} m, target={target_height:.4f} m"
                )
                return True


            # Avoid busy waiting.
            time.sleep(self.loop_interval)

    # ======================================================================
    # End-effector poses
    # ======================================================================

    def get_base_T_ee(self, timeout=None):
        """
        Wait for a fresh left and right end-effector pose.

        The method does not clear the currently stored poses. Instead, it
        records the request time and waits until both pose callbacks have
        produced newer data.

        Args:
            timeout:
                Maximum waiting time in seconds.
                None means wait indefinitely.

        Returns:
            (left_T, right_T)

            Both are copies of the internal NumPy matrices, so modifying the
            returned matrices cannot modify RobotController's internal state.

        Raises:
            TimeoutError:
                If timeout is exceeded.
        """

        self._node.get_logger().info(
            "Waiting for new end-effector poses..."
        )

        request_time = self._node.get_clock().now()
        start_time = time.monotonic()

        while True:
            if (
                timeout is not None
                and time.monotonic() - start_time >= timeout
            ): 
                self._node.get_logger().error(
                    "Timed out while waiting for new end-effector poses."
                )
                return False

            # Take one consistent snapshot of all related state.
            with self.state_lock:
                left_pose = self.base_T_left_ee
                right_pose = self.base_T_right_ee

                left_time = self.left_ee_update_time
                right_time = self.right_ee_update_time

            # We need both poses to exist and both to have been updated
            # after this request.
            if (
                left_pose is not None
                and right_pose is not None
                and left_time is not None
                and right_time is not None
                and left_time > request_time
                and right_time > request_time
            ):
                # Return independent snapshots.
                left_pose = left_pose.copy()
                right_pose = right_pose.copy()

                self._node.get_logger().info(
                    f"Got Left EE pose:\n{left_pose}"
                )

                self._node.get_logger().info(
                    f"Got Right EE pose:\n{right_pose}"
                )

                return left_pose, right_pose

            # Avoid busy waiting.
            time.sleep(self.loop_interval)

    # ======================================================================
    # Joint-space arm motion
    # ======================================================================

    def move_j(self, left_joint_states=None, right_joint_states=None):

        if not self.move_j_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "/move_arm_j server not available"
            )
            return False

        goal = MoveArmJoints.Goal()

        goal.left_joints = (
            left_joint_states
            if left_joint_states is not None
            else []
        )

        goal.right_joints = (
            right_joint_states
            if right_joint_states is not None
            else []
        )

        goal.dry_run = False
        goal.duration = 0.0

        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.move_j_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 5.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_j goal."
            )
            return False

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "/move_arm_j returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "/move_arm_j goal rejected"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_j goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for result
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, 30.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_j result."
            )
            return False

        result = result_future.result().result

        if not result.success:
            self._node.get_logger().error(
                f"move_arm_j failed: "
                f"error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_j result: "
            f"success={result.success}, "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )

        return True

    # ======================================================================
    # Cartesian pose arm motion
    # ======================================================================

    def move_p(self, left_pose=None, right_pose=None):

        if not self.move_p_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "/move_arm_p server not available"
            )
            return False

        goal = MoveArmPose.Goal()

        goal.left_pose = (
            left_pose
            if left_pose is not None
            else []
        )

        goal.right_pose = (
            right_pose
            if right_pose is not None
            else []
        )

        goal.dry_run = False
        goal.disable_environment_collision = True

        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.move_p_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 2.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_p goal."
            )
            return False

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "/move_arm_p returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "/move_arm_p goal rejected"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_p goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for result
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, 30.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_p result."
            )
            return False

        result = result_future.result().result

        if not result.success:
            self._node.get_logger().error(
                f"/move_arm_p failed: "
                f"error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_p result: "
            f"success={result.success}, "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )

        return True

    # ======================================================================
    # Linear arm motion
    # ======================================================================

    def move_l(self, left_pose=None, right_pose=None):

        if not self.move_l_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "/move_arm_l server not available while withdrawing."
            )
            return False

        goal = MoveArmPose.Goal()

        goal.left_pose = (
            left_pose
            if left_pose is not None
            else []
        )

        goal.right_pose = (
            right_pose
            if right_pose is not None
            else []
        )

        goal.dry_run = False

        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.move_l_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 2.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_l goal."
            )
            return False

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "/move_arm_l returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "/move_arm_l goal rejected"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_l goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for result
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, 20.0):
            self._node.get_logger().error(
                "Failed while waiting for /move_arm_l result."
            )
            return False

        result = result_future.result().result

        if not result.success:
            self._node.get_logger().error(
                f"/move_arm_l failed: "
                f"error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False

        self._node.get_logger().info(
            "/move_arm_l result: "
            f"success={result.success}, "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )

        return True
    
    def move_t(
        self,
        left_pose=None,
        right_pose=None,
        move_timeout=20.0,
        speed=1,
        velocity=0.1,
        disable_environment_collision=True
    ):

        if not self.move_t_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "/moveT server not available while withdrawing."
            )
            return False

        goal = MoveArmPose.Goal()
        goal.left_pose = (
            left_pose
            if left_pose is not None
            else []
        )
        goal.right_pose = (
            right_pose
            if right_pose is not None
            else []
        )
        goal.dry_run = False
        goal.speed = speed
        goal.velocity = velocity
        goal.disable_environment_collision = disable_environment_collision

        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.move_t_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 2.0):
            self._node.get_logger().error(
                "Failed while waiting for /moveT goal."
            )
            return False

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "/moveT returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "/moveT goal rejected"
            )
            return False

        self._node.get_logger().info(
            "/moveT goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for result
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, move_timeout):
            self._node.get_logger().error(
                "Failed while waiting for /moveT result."
            )
            return False

        result = result_future.result().result

        if not result.success:
            self._node.get_logger().error(
                f"/moveT failed: "
                f"error_code={result.error_code}, "
                f"message={result.message}"
            )
            return False

        self._node.get_logger().info(
            "/moveT result: "
            f"success={result.success}, "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )

        return True

    def move_t_force(
        self,
        force_threshold,
        left_pose=None,
        right_pose=None,
        move_timeout=20.0,
        speed=0.3,
        velocity=0.3,
        disable_environment_collision=True,
    ):
        """
        Move the end-effector in its self coordinate frame while monitoring
        force feedback.

        The /moveT action interprets the pose as a relative motion in the
        end-effector's local coordinate frame.

        Returns:
            True:
                Contact was detected.

            False:
                No contact, action failed, or force sensor data was
                unavailable.
        """
        # ------------------------------------------------------------------
        # Check action server
        # ------------------------------------------------------------------

        if not self.move_t_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "/moveT server not available."
            )
            return False

        # ------------------------------------------------------------------
        # Get initial force
        # ------------------------------------------------------------------

        force = self._sensor.get_next_force()
        if force is None:
            self._node.get_logger().error(
                "No force sensor data received"
            )
            return False

        _, _, initial_fz = force
        self._node.get_logger().info(f"Initial force: Fz={initial_fz:.4f} kg")

        # ------------------------------------------------------------------
        # Construct goal
        # ------------------------------------------------------------------

        goal = MoveArmPose.Goal()
        goal.left_pose = (
            left_pose
            if left_pose is not None
            else []
        )
        goal.right_pose = (
            right_pose
            if right_pose is not None
            else []
        )
        goal.dry_run = False
        goal.speed = speed
        goal.velocity = velocity
        goal.disable_environment_collision = disable_environment_collision
                
        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.move_t_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 5.0):
            self._node.get_logger().error(
                "Failed while waiting for /moveT goal."
            )
            return False

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "/moveT returned no goal handle."
            )
            return False

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "/moveT goal rejected"
            )
            return False

        self._node.get_logger().info(
            "/moveT goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for movement result or force contact.
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()
        contact_detected = False
        start_time = time.monotonic()
        while not result_future.done():

            elapsed_time = time.monotonic() - start_time

            if elapsed_time >= move_timeout:
                self._node.get_logger().warning(
                    f"/moveT force-feedback timeout after "
                    f"{elapsed_time:.2f} seconds."
                )

                cancel_future = goal_handle.cancel_goal_async()

                if not self._wait_for_future(
                    cancel_future,
                    self.wait_cancel_timeout
                ):
                    self._node.get_logger().warning(
                        "Timed out while waiting for /moveT cancellation."
                    )
                break

            force = self._sensor.get_next_force(timeout_sec=0.1)
            if force is None:
                continue

            fx, fy, fz = force
            delta_fz = abs(fz - initial_fz)
            self._node.get_logger().info(
                f"Suction force: "
                f"Fx={fx:.4f}, "
                f"Fy={fy:.4f}, "
                f"Fz={fz:.4f}, "
                f"ΔFz={delta_fz:.4f}"
            )

            if delta_fz >= force_threshold:
                self._node.get_logger().info(
                    "Suction force exceeds the threshold, "
                    "stopping /moveT motion."
                )
                cancel_future = goal_handle.cancel_goal_async()
                if not self._wait_for_future(
                    cancel_future,
                    self.wait_cancel_timeout
                ):
                    self._node.get_logger().warning(
                        "Timed out while waiting for "
                        "/moveT cancellation."
                    )
                contact_detected = True
                break

            time.sleep(self.loop_interval)

        # ------------------------------------------------------------------
        # Wait for final action result.
        # ------------------------------------------------------------------
        if not result_future.done():
            self._wait_for_future(result_future, 10.0)

        if result_future.result() is None:
            self._node.get_logger().error(
                "No result received from /moveT."
            )
            return False

        result = result_future.result().result

        # ------------------------------------------------------------------
        # Contact detected
        # ------------------------------------------------------------------

        if contact_detected:
            self._node.get_logger().info(
                "Suction successful: contact detected."
            )
            return True

        # ------------------------------------------------------------------
        # Motion completed without contact.
        # ------------------------------------------------------------------

        if result.success:
            self._node.get_logger().warning(
                "Maximum /moveT approach distance reached "
                "without detecting contact."
            )
            return True

        # ------------------------------------------------------------------
        # Action itself failed.
        # ------------------------------------------------------------------

        self._node.get_logger().error(
            f"/moveT failed: "
            f"error_code={result.error_code}, "
            f"message={result.message}"
        )
        return False

    # ======================================================================
    # Object pose estimation
    # ======================================================================

    def get_cam_T_obj(self, model_label):
        self._node.get_logger().info(
            f'Object pose server ready: '
            f'{self.object_client.server_is_ready()}'
        )

        # ------------------------------------------------------------------
        # Check action server
        # ------------------------------------------------------------------

        if not self.object_client.wait_for_server(
            timeout_sec=self.server_timeout
        ):
            self._node.get_logger().error(
                "Wait object pose action server timeout"
            )
            return None

        # ------------------------------------------------------------------
        # Construct goal
        # ------------------------------------------------------------------

        goal = EstimateObjectPose.Goal()

        goal.model_label = model_label
        goal.instance_index = 0
        goal.confidence_threshold = 0.0

        # ------------------------------------------------------------------
        # Send goal
        # ------------------------------------------------------------------

        send_future = self.object_client.send_goal_async(goal)

        if not self._wait_for_future(send_future, 2.0):
            self._node.get_logger().error(
                "Failed while waiting for object pose goal."
            )
            return None

        goal_handle = send_future.result()

        if goal_handle is None:
            self._node.get_logger().error(
                "Object pose action returned no goal handle."
            )
            return None

        if not goal_handle.accepted:
            self._node.get_logger().error(
                "Object pose goal rejected"
            )
            return None

        self._node.get_logger().info(
            "Object pose goal accepted"
        )

        # ------------------------------------------------------------------
        # Wait for result
        # ------------------------------------------------------------------

        result_future = goal_handle.get_result_async()

        if not self._wait_for_future(result_future, 25.0):
            self._node.get_logger().error(
                "Failed while waiting for object pose result."
            )
            return None

        action_result = result_future.result()

        if action_result is None:
            self._node.get_logger().error(
                "Object pose action returned no result."
            )
            return None

        result = action_result.result

        # ------------------------------------------------------------------
        # Convert camera-frame object pose to transformation matrix.
        # ------------------------------------------------------------------

        cam_T_obj = pose_2_tf_mat(result.pose.pose)

        self._node.get_logger().info(
            f"cam_T_obj:\n{cam_T_obj}"
        )

        return cam_T_obj
