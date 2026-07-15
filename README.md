# Changan mission workspace

This ROS 2 workspace owns mission-level robot sequencing. Grasp perception stays
in the separate `grasp_ws`; dual-arm planning and execution stay in the separate
dual-arm workspace.

## Actions

- `/execute_grasp` (`mission_interfaces/action/ExecuteGrasp`)
- `/execute_place` (`mission_interfaces/action/ExecutePlace`)
- `/execute_bin_grasp` (`mission_interfaces/action/ExecuteBinGrasp`)
- `/execute_bin_place` (`mission_interfaces/action/ExecuteBinPlace`)

Only one mission is accepted at a time.

The bin actions are registered separately from the material actions. They are
disabled by default until bin-specific observation joint targets and the bin
perception service are configured. They reject goals while
`bin_mission_enabled` is false and do not publish robot commands.

The intended bin grasp sequence is: initialize the bin observation pose, call
the bin perception service, execute the detected grasp pose, close the selected
gripper, and lift only the torso while keeping the arm at the grasp pose. The
intended bin place sequence is: move the chassis to its fixed location, bend the
torso, open the selected gripper, then home the arms and reset the torso.

### Grasp sequence

1. Start opening both grippers, moving the torso to its preparation target, and
   calling `/move_arm_j` with the configured dual-arm preparation joints in
   parallel. Wait for all three preparations to settle before continuing.
2. Call `/detect_grasp_pose` and receive a grasp-center pose in the D405 color
   optical frame.
3. Transform the grasp center to `torso_link4`, apply the configured 0.15 m
   grasp-center-to-gripper retreat, then use the URDF
   `gripper_link -> arm_link7` transform to generate the `/move_arm_p` target.
4. Close the selected gripper.
5. Publish the torso reset target. The arms remain in the grasp pose so the
   object can be carried.

### Place sequence

1. Publish a constant chassis velocity for the configured distance and duration,
   then always publish zero velocity.
2. Publish the configured torso target and call `/move_arm_j` with only the
   configured right-arm place joint target. The left arm target is empty and
   remains at its current position.
3. Open the selected gripper.
4. Publish the torso reset target, then call `/home` to reset both arms.

The chassis move is open-loop. Calibrate the distance, duration, direction, and
joint targets in `mission_controller/config/mission.yaml` before hardware use.

## R1PRO command transport

The chassis and gripper publishers follow the examples under
`dual_arm_manipulation/tools/r1pro_test`:

- Commands use reliable, keep-last depth 10, transient-local QoS.
- Grippers publish `sensor_msgs/msg/JointState` with one percentage value on
  `/motion_target/target_position_gripper_left` or `_right`; `0` is closed and
  `100` is open.
- The chassis publishes `geometry_msgs/msg/TwistStamped` on
  `/motion_target/target_speed_chassis` at 10 Hz by default and publishes one
  all-zero command when motion finishes, fails, or is canceled.
- By default the node waits three seconds for a command subscriber, warns, and
  publishes anyway like the reference scripts. Set
  `require_command_subscribers: true` for strict mission failure instead.

## Build

Activate `changan`, then source the installed dual-arm and perception
workspaces before building:

```bash
export DUAL_ARM_WS="<dual-arm-workspace>"
export GRASP_WS="<grasp-workspace>"
export MISSION_WS="<mission-workspace>"

conda activate changan
source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
cd "$MISSION_WS"
export PYTHONNOUSERSITE=1
colcon build --merge-install --symlink-install \
  --cmake-args "-DCMAKE_BUILD_TYPE=Release" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Launch

Start the dual-arm implementation separately. Activate `changan` so the included
grasp detector and mission controller use that Conda environment, then overlay
all workspaces:

```bash
conda activate changan
source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
source "$MISSION_WS/install/setup.zsh"
ros2 launch mission_controller mission.launch.py
```

The launch includes the perception launch by default but does not launch the
dual-arm implementation. Useful launch overrides are:

```bash
ros2 launch mission_controller mission.launch.py start_perception:=false
ros2 launch mission_controller mission.launch.py start_detector_daemon:=false
ros2 launch mission_controller mission.launch.py config_file:="<mission-config>"
```

The perception repository contains no machine path. If its inferred merged
workspace paths are not suitable, set `GRASPNESS_C_DIR`, `GRASPNESS_CHECKPOINT`,
and `GRASP_RUNTIME_DIR` before launching.

### RViz Graspness transform preview

The offline preview uses the fixed Graspness sample captured in
`hdas/camera_wrist_right_color_optical_frame`. It publishes the same torso and
dual-arm preparation positions used immediately before detection by
`/execute_grasp`, starts a preview-only robot state publisher, and executes the
same mission transform functions without sending any robot command:

```bash
source /opt/ros/humble/setup.zsh
source /home/dekc/libraries/ws_moveit2/install/setup.zsh
source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
source "$MISSION_WS/install/setup.zsh"
ros2 launch mission_controller grasp_preview.launch.py
```

RViz displays three labeled poses on `/mission/grasp_visualization`:

- orange `grasp_center`: corrected Graspness result in the execution frame,
  shown as axes plus a translucent full gripper assembly before applying
  `grasp_to_gripper_rpy`;
- green `right_gripper_link_target`: 0.15 m behind the grasp center, shown as
  axes plus a translucent full gripper assembly after `grasp_to_gripper_rpy`
  and the local-Z `gripper_target_post_rpy` correction;
- cyan `arm_link7_target`: final target after applying the URDF fixed joint.

The corresponding `PoseStamped` topics are `/mission/grasp_pose`,
`/mission/gripper_link_target`, and `/mission/arm_link7_target`. Set
`start_rviz:=false` for a headless transform check.
The orange and green gripper assemblies deliberately use the same URDF mesh
set: their relative opening directions expose any 90-degree grasp-frame
convention error without changing the pose sent to planning.
Both finger-link poses come from the URDF TF chain. The offline preview opens
each finger by 0.04 m so the opening axis remains visible.
The target is also broadcast as TF child
`mission_target/right_gripper_link` for direct inspection in the TF display.

#### Manually verify `move_arm_p` planning

Use three terminals. First start the R1 Pro planning stack. Keeping its global
`dry_run` enabled prevents any trajectory execution:

```bash
source /opt/ros/humble/setup.zsh
source /home/dekc/libraries/ws_moveit2/install/setup.zsh
source "$DUAL_ARM_WS/install/setup.zsh"
ros2 launch robot_bringup planning_only.launch.py \
  robot_profile:=r1_pro dry_run:=true enable_rviz:=false
```

Then publish the `/execute_grasp` preparation joints directly to the planning
stack and generate the fixed Graspness target. The planning stack already owns
`robot_state_publisher`, so the preview copy is disabled:

```bash
source /opt/ros/humble/setup.zsh
source /home/dekc/libraries/ws_moveit2/install/setup.zsh
source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
source "$MISSION_WS/install/setup.zsh"
ros2 launch mission_controller grasp_preview.launch.py \
  start_robot_state_publisher:=false joint_states_topic:=/joint_states
```

After checking the orange, green, and cyan markers in RViz, explicitly forward
the cyan target to `/move_arm_p`. The executor reads the current arm-link TF,
splits position linearly and orientation with quaternion SLERP, and sends ten
segments in sequence. The default remains plan-only:

```bash
source /opt/ros/humble/setup.zsh
source /home/dekc/libraries/ws_moveit2/install/setup.zsh
source "$DUAL_ARM_WS/install/setup.zsh"
source "$MISSION_WS/install/setup.zsh"
ros2 run mission_controller grasp_target_executor
```

Override the number of segments with `-p interpolation_steps:=N`. In dry-run
mode every segment is planned from the unchanged measured robot state; during
confirmed real execution, each successful segment updates the measured state
before the next goal is sent.
The preview RViz subscribes to `/display_planned_path`, queues successful
segments without interrupting the active animation, and shows the trajectory
robot trail. `segment_pause_sec` defaults to 0.5 seconds and can be overridden
on `grasp_target_executor` when a longer visual pause is useful.

Success is reported as `move_arm_p prepared in dry-run mode`. A failure such as
`MoveIt IK failed` is a hard stop: do not switch to real execution. After the
local-Z 180-degree gripper correction, the fixed sample produces the right-arm
target `[0.640249, -0.005418, 0.029806, -0.123568, -0.160279, -0.050835,
-0.977986]`. Exact IK failed for this target, for a full 360-degree local-Z
sweep in 45-degree steps, and when retaining the initial end-effector
orientation at the exact grasp point. A nearby diagnostic pre-grasp pose
`[0.462374, -0.087679, 0.030488, 0.206985, -0.445968, 0.295979, 0.818942]`
did plan successfully in dry-run mode, but it is not the final grasp pose.

For a physical robot, do not publish preview joint states on `/joint_states`:
place the robot at the `/execute_grasp` preparation pose first and use its
measured joint-state TF. Only after the dry-run succeeds and the workspace is
clear, launch the hardware stack with its global dry-run disabled and make the
explicit execution call:

```bash
ros2 run mission_controller grasp_target_executor --ros-args \
  -p dry_run:=false -p execute_confirmed:=true
```

Both parameters are required for real execution. This command moves the arm to
the grasp pose; it does not close the gripper.

## Run

Execute a real grasp only after the topic directions and joint targets have been
validated:

```bash
ros2 action send_goal --feedback \
  /execute_grasp mission_interfaces/action/ExecuteGrasp \
  "{request_id: grasp_1, target_frame: torso_link, target_label: 0, arm: right, publish_pose: true, detection_timeout_sec: 20.0, dry_run: false}"
```

Execute the configured place flow:

```bash
ros2 action send_goal --feedback \
  /execute_place mission_interfaces/action/ExecutePlace \
  "{request_id: place_1, arm: right, dry_run: false}"
```

For a safe integration check, use `dry_run: true`. Direct chassis, torso,
arm-joint, and gripper commands are skipped. Grasp detection and `/move_arm_p`
planning still run; place calls `/home` with its `dry_run` flag set.

The repository also contains an isolated mock integration fixture under
`mission_controller/test`. Run it on a non-production `ROS_DOMAIN_ID`; it
checks the full grasp and place ordering without connecting to robot nodes.
