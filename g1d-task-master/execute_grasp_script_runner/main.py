#!/usr/bin/env python3

import time

import rclpy
from rclpy.action import ActionClient
from task_interfaces.action import MoveArmJoints, MoveArmPose


DRY_RUN = False
MOVEJ_DURATION = 0.0
SLEEP_BETWEEN_MOVEJ_AND_MOVEP = 0.0

LEFT_JOINT_WAYPOINTS = [
    [
        0.26833879947662354,
        0.3913688361644745,
        -0.22823956608772278,
        1.237898588180542,
        -0.1101350262761116,
        0.03600061312317848,
        -0.3565187156200409,
    ],
    [
        0.5869633555412292,
        0.5509747266769409,
        0.14594389498233795,
        1.239037036895752,
        -0.327037513256073,
        -0.8019363880157471,
        -0.7658519148826599,
    ],
    [
        0.5553130507469177,
        0.6595518589019775,
        0.09400426596403122,
        0.12288624793291092,
        -0.042699795216321945,
        0.8717325329780579,
        -0.5522091388702393,
    ],

]

LEFT_POSE_WAYPOINTS = [
    [
        0.13550195139488017,
        0.316090741827874,
        -0.05474292413424675,
        -0.503923688907523,
        0.5042768028136182,
        -0.5032209535665213,
        0.4883999322210336,
    ],
    [
        0.13550195139488017,
        0.316090741827874,
        -0.08674292413424675,
        -0.503923688907523,
        0.5042768028136182,
        -0.5032209535665213,
        0.4883999322210336,
    ],
    [
        0.13550195139488017,
        0.316090741827874,
        -0.04474292413424675,
        -0.503923688907523,
        0.5042768028136182,
        -0.5032209535665213,
        0.4883999322210336,
    ],
]

FINAL_LEFT_JOINTS = [
    0.5284803509712219,
    0.8249341249465942,
    0.22120483219623566,
    0.708902895450592,
    0.06507433950901031,
    0.24904417991638184,
    -0.8627563714981079,
]


def send_move_arm_j(node, action_client, left_joints, right_joints=None,
                    dry_run=False, duration=5.0):
    goal = MoveArmJoints.Goal()
    goal.left_joints = left_joints
    goal.right_joints = right_joints or []
    goal.dry_run = dry_run
    goal.duration = duration

    def feedback_callback(feedback_msg):
        feedback = feedback_msg.feedback
        node.get_logger().info(
            "move_arm_j feedback: "
            f"stage={feedback.stage}, progress={feedback.progress:.2f}, "
            f"detail={feedback.detail}"
        )

    node.get_logger().info("Waiting for action server: /move_arm_j")
    action_client.wait_for_server()

    send_future = action_client.send_goal_async(
        goal,
        feedback_callback=feedback_callback,
    )
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        raise RuntimeError("move_arm_j goal rejected")

    node.get_logger().info("move_arm_j goal accepted")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result

    node.get_logger().info(
        "move_arm_j result: "
        f"success={result.success}, error_code={result.error_code}, "
        f"message={result.message}"
    )
    if not result.success:
        raise RuntimeError(
            f"move_arm_j failed: error_code={result.error_code}, "
            f"message={result.message}"
        )

    return result


def send_move_arm_p(node, action_client, left_pose=None, right_pose=None,
                    dry_run=False):
    goal = MoveArmPose.Goal()
    goal.left_pose = left_pose or []
    goal.right_pose = right_pose or []
    goal.dry_run = dry_run

    def feedback_callback(feedback_msg):
        feedback = feedback_msg.feedback
        node.get_logger().info(
            "move_arm_p feedback: "
            f"stage={feedback.stage}, progress={feedback.progress:.2f}, "
            f"detail={feedback.detail}"
        )

    node.get_logger().info("Waiting for action server: /move_arm_p")
    action_client.wait_for_server()

    send_future = action_client.send_goal_async(
        goal,
        feedback_callback=feedback_callback,
    )
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        raise RuntimeError("move_arm_p goal rejected")

    node.get_logger().info("move_arm_p goal accepted")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result

    node.get_logger().info(
        "move_arm_p result: "
        f"success={result.success}, error_code={result.error_code}, "
        f"message={result.message}"
    )
    if not result.success:
        raise RuntimeError(
            f"move_arm_p failed: error_code={result.error_code}, "
            f"message={result.message}"
        )

    return result


def send_move_arm_l(node, action_client, left_pose=None, right_pose=None,
                    dry_run=False):
    goal = MoveArmPose.Goal()
    goal.left_pose = left_pose or []
    goal.right_pose = right_pose or []
    goal.dry_run = dry_run

    def feedback_callback(feedback_msg):
        feedback = feedback_msg.feedback
        node.get_logger().info(
            "move_arm_l feedback: "
            f"stage={feedback.stage}, progress={feedback.progress:.2f}, "
            f"detail={feedback.detail}"
        )

    node.get_logger().info("Waiting for action server: /move_arm_l")
    action_client.wait_for_server()

    send_future = action_client.send_goal_async(
        goal,
        feedback_callback=feedback_callback,
    )
    rclpy.spin_until_future_complete(node, send_future)
    goal_handle = send_future.result()

    if not goal_handle.accepted:
        raise RuntimeError("move_arm_l goal rejected")

    node.get_logger().info("move_arm_l goal accepted")
    result_future = goal_handle.get_result_async()
    rclpy.spin_until_future_complete(node, result_future)
    result = result_future.result().result

    node.get_logger().info(
        "move_arm_l result: "
        f"success={result.success}, error_code={result.error_code}, "
        f"message={result.message}"
    )
    if not result.success:
        raise RuntimeError(
            f"move_arm_l failed: error_code={result.error_code}, "
            f"message={result.message}"
        )

    return result


def main():
    rclpy.init()
    node = rclpy.create_node("task_eg4_direct_action_client")
    movej_client = ActionClient(node, MoveArmJoints, "/move_arm_j")
    movep_client = ActionClient(node, MoveArmPose, "/move_arm_p")
    movel_client = ActionClient(node, MoveArmPose, "/move_arm_l")

    try:
        for index, left_joints in enumerate(LEFT_JOINT_WAYPOINTS, start=1):
            print(f"Moving left arm joint waypoint {index}/{len(LEFT_JOINT_WAYPOINTS)}")
            send_move_arm_j(
                node,
                movej_client,
                left_joints=left_joints,
                right_joints=[],
                dry_run=DRY_RUN,
                duration=MOVEJ_DURATION,
            )

        print(f"Sleeping {SLEEP_BETWEEN_MOVEJ_AND_MOVEP:.0f} seconds before move_arm_p")
        time.sleep(SLEEP_BETWEEN_MOVEJ_AND_MOVEP)

        for index, left_pose in enumerate(LEFT_POSE_WAYPOINTS, start=1):
            print(f"Moving left arm pose waypoint {index}/{len(LEFT_POSE_WAYPOINTS)} with move_arm_p")
            send_move_arm_p(
                node,
                movep_client,
                left_pose=left_pose,
                right_pose=[],
                dry_run=DRY_RUN,
            )

        print("Moving left arm to final joint waypoint")
        send_move_arm_j(
            node,
            movej_client,
            left_joints=FINAL_LEFT_JOINTS,
            right_joints=[],
            dry_run=DRY_RUN,
            duration=MOVEJ_DURATION,
        )
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
