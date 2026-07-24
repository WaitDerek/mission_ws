# Changan mission workspace

This ROS 2 workspace owns mission-level robot sequencing. Grasp perception stays
in the separate `grasp_ws`; dual-arm planning and execution stay in the separate
dual-arm workspace.

## Actions

- `/execute_grasp` (`mission_interfaces/action/ExecuteGrasp`)
- `/execute_place` (`mission_interfaces/action/ExecutePlace`)
- `/execute_box_grasp` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/execute_box_place` (`mission_interfaces/action/ExecuteBoxPlace`)
- `/execute_box_stack` (`mission_interfaces/action/ExecuteBoxStack`)
- `/move_chassis` (`mission_interfaces/action/MoveChassis`)

Only one mission is accepted at a time.

The executable is assembled from small task-domain modules:

- `mission_controller.py`: ROS node construction, parameters, and goal admission.
- `action_runtime.py`: shared command publication, fresh-feedback checks,
  readiness checks, dependency calls, and cancellation-aware waits.
- `material_actions.py` / `box_actions.py`: mission-stage ordering.
- `grasp_support.py` / `box_support.py`: perception transforms and delegated
  arm execution.
- `stack_action.py` / `chassis_action.py`: independent stack and chassis flows.

The box actions are registered separately from the material actions. The node's
safe fallback keeps them disabled, while the shipped R1 Pro configuration
enables them with calibrated observation targets and FoundationPose settings.
They reject goals while `box_mission_enabled` is false and do not publish robot
commands.

The box grasp sequence moves both arms through the shared collision-clearance
posture before entering the saved dual-arm observation pose, calls FoundationPose,
transforms the OBB centre from the D405 optical frame into `torso_link4`, preserves
the configured model axes and geometric-centre semantics, and delegates the two-hand
geometry and motion to `/pickup_task`.
The configured box dimensions are passed explicitly as width and height. Once
pickup execution succeeds, mission closes both grippers and requires fresh
feedback proving that both retained the box. Only then may it lift the torso to
the shallower box-carry waist posture. A failed `/pickup_task` restores the
observation posture and retries once from a fresh FoundationPose result; two
failures abort before any close or lift command.
The box place sequence remains separate: bend the torso, open both grippers,
clear the released box, straighten the torso, then call `/go_ready`. Chassis
movement is deliberately not embedded in either place sequence; call
`/move_chassis` explicitly before or after a place action as required.

### Box grasp sequence

1. Open both grippers and wait for them to settle, move both arms to the shared
   intermediate posture, then move the torso and both arms together to the
   saved box observation posture.
2. Run FoundationPose, transform the geometric-centre pose into `torso_link4`,
   and call `/pickup_task` with the configured dimensions. Restore the
   observation posture, capture a fresh pose, and retry once if it fails.
3. After `/pickup_task` reports success, close both grippers and require fresh
   feedback from both sides. An empty or stale-feedback result aborts without
   lifting.
4. Only after both grippers pass the retained-object check, publish
   `box_grasp_torso_lift_positions` while preserving the final arm targets.

### Box place sequence

1. Publish `box_place_torso_positions` to return to the same waist posture used
   for box detection.
2. Open both grippers and wait for them to settle.
3. Move only Torso2 from `-0.81` to `-1.20` and verify the configured
   intermediate waist posture.
4. Straighten and verify all torso joints at `[0, 0, 0, 0]`, then call
   `/go_ready` for the dual-arm work posture.

### Grasp sequence

1. Open both grippers and wait for them to settle, call `/move_arm_j` with the
   shared intermediate dual-arm posture, then move the torso and both arms
   together to the configured material-observation posture.
2. Call `/detect_grasp_pose` and receive a grasp-center pose in the D405 color
   optical frame. Mission enforces `grasp_detection_min_timeout_sec` (currently
   60 seconds), even if the action requests a shorter timeout. Detection
   timeout/failure or failure of the highest-score candidate starts a fresh detection;
   `grasp_detection_attempts: 0` means this repeats until a candidate executes
   successfully or the action is canceled. Only one candidate is executed from
   each RGB-D result (`grasp_candidates_per_detection: 1`).
3. Transform the grasp center to `torso_link4`, apply the configured 0.03 m
   grasp-center-to-gripper retreat, then use the URDF
   `gripper_link -> arm_link7` transform to generate the `/move_arm_p` target.
   For the parallel-jaw 180-degree symmetry, compare both equivalent branches
   against the current gripper orientation and keep the branch requiring less
   wrist rotation so the wrist camera remains on the same side.
4. Read the current `arm_link7` TF and execute two `/move_arm_p` stages: the
   halfway interpolated pose followed by the final grasp pose. Position uses
   linear interpolation and orientation uses shortest-path quaternion SLERP.
5. Close the selected gripper and read a fresh raw HDAS position measurement.
   Convert it between the configured open and closed positions. A close ratio
   above `grasp_empty_close_ratio_threshold` (currently 95%) means the fingers
   nearly met without retaining material, so the grasp failed. A ratio at or
   below 95% means material is holding the fingers apart.
6. After an empty close, reopen the selected gripper, return directly
   to the configured material-observation posture, request a fresh detection,
   and retry. `grasp_max_empty_close_attempts` is currently `0`, which means
   unlimited empty-grasp retries: closure feedback alone never aborts the
   action. If the observation-return `/move_arm_j` is transiently canceled
   (`error_code=17`), resend the same observation target after
   `grasp_recovery_retry_delay_sec` instead of aborting. Otherwise, keep the
   selected gripper closed and return to the observation posture normally.
7. At the final observation posture, reassert the closed-gripper command and
   read fresh HDAS feedback. A measured closure above 95% reopens the gripper
   and starts a fresh detection/grasp attempt; a closure at or below 95%
   confirms that material remains held and completes the action.

### Place sequence

1. Publish the configured torso target and call `/move_arm_j` with only the
   configured right-arm place joint target. The left arm target is empty and
   remains at its current position.
2. Open the selected gripper.
3. Return both arms and the torso to the configured material-observation
   posture.
4. Straighten and verify Torso3, wait two seconds, then straighten Torso1 and
   Torso2 together and verify the complete `[0, 0, 0, 0]` posture. The action
   does not call `/home`.

### Chassis sequence

`/move_chassis` accepts only one of six directions: `forward`, `backward`,
`left`, `right`, `clockwise`, or `counterclockwise`. Configure
`chassis_linear_speed`, `chassis_angular_speed`, and
`chassis_move_duration_sec` in `mission_controller/config/mission.yaml`.
The action continuously publishes the configured velocity for the configured
duration, then always publishes zero velocity on success, failure, or
cancellation. This is open-loop movement; the nominal displacement is
`speed * duration`.

### Fixed-level box stack sequence

`/execute_box_stack` accepts only `level` in `[1, 4]`. At goal start, Mission
checks the fixed arm posture and upright torso. If either is not ready, it moves
both arms to the configured loading posture over eight seconds while returning
the torso to zero, then verifies measured feedback. Mission closes both
grippers, keeps both arms fixed, moves the torso to the waist state for the
selected level, verifies the measured torso feedback, opens both grippers, and
raises the torso back to the same highest posture. The four level targets are
the four rows in `stack_level_torso_positions`. This action does not call
`/place_task` and does not solve a new arm IK for each level.

After measured torso feedback confirms the selected stack level, Mission holds
the pose for `stack_release_delay_sec` (currently `2.0 s`) before opening both
grippers.

Every stack torso move uses Torso1 as the timing reference. Torso1 runs at
`stack_torso1_speed` (currently `0.1 rad/s`); Mission calculates the remaining
Torso1 time from measured feedback, then assigns Torso2, Torso3, and Torso4
their own `remaining_angle / Torso1_time` velocity so all moving waist joints
are expected to finish together, except Torso3 uses the fixed
`stack_torso3_speed` setting (currently `0.13 rad/s`) for every stack level,
including lowering and returning upright.

The current hardware-validated fixed arm posture is:

```text
left  = [-1.133818,  0.120475, -1.197170, -0.672971,  2.354960, 1.046860,  1.240171]
right = [-1.129491, -0.125152,  1.203598, -0.676157, -2.343158, 1.040511, -1.249240]
```

The waist targets are:

```text
highest manual-load/retreat posture = [0.0, 0.0, 0.0, 0.0]
level 1 / lowest = [1.630000, -2.500000, -0.920000, 0.0]
level 2 (18.0 cm) = [1.356518, -2.506617, -1.049991, 0.0]
level 3 (30.0 cm) = [1.154311, -2.139907, -0.885489, 0.0]
level 4 (45.0 cm) = [0.906723, -1.646855, -0.640023, 0.0]
```

Level 1 is moved inward from the hardware safety-boundary freeze observed
during commissioning. Levels 2 and 3 are linearly
interpolated from the tested 12.5 cm calibration points; Level 4 is a linear
extrapolation to 45 cm and must be verified cautiously on hardware.

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

Source the installed dual-arm and perception workspaces before building:

```bash
export DUAL_ARM_WS="<dual-arm-workspace>"
export GRASP_WS="<grasp-workspace>"
export MISSION_WS="<mission-workspace>"

source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
cd "$MISSION_WS"
colcon build --merge-install --symlink-install \
  --cmake-args "-DCMAKE_BUILD_TYPE=Release" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
```

## Launch

Start the dual-arm implementation separately, then overlay all workspaces:

```bash
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
source "$MOVEIT2_WS/install/setup.zsh"
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
`/mission/gripper_link_target`, `/mission/arm_link7_target`, and the actual
two-stage midpoint `/mission/arm_link7_intermediate`. Set
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
source "$MOVEIT2_WS/install/setup.zsh"
source "$DUAL_ARM_WS/install/setup.zsh"
ros2 launch robot_bringup planning_only.launch.py \
  robot_profile:=r1_pro dry_run:=false enable_rviz:=true \
  enable_fake_ros2_control:=true
```

This planning-only launch uses mock ros2_control: no robot adapter is started,
but `/move_arm_j` and successful MoveIt trajectories animate the virtual robot
through `joint_state_broadcaster`. Use `dry_run:=true` when only plan validation
and markers are wanted.

Then publish the `/execute_grasp` preparation joints directly to the planning
stack and generate the fixed Graspness target. The planning stack already owns
`robot_state_publisher`, so the preview copy is disabled:

```bash
source /opt/ros/humble/setup.zsh
source "$MOVEIT2_WS/install/setup.zsh"
source "$DUAL_ARM_WS/install/setup.zsh"
source "$GRASP_WS/install/setup.zsh"
source "$MISSION_WS/install/setup.zsh"
ros2 launch mission_controller grasp_preview.launch.py \
  start_robot_state_publisher:=false joint_states_topic:=/joint_states
```

After checking the orange, green, and cyan markers in RViz, explicitly forward
the cyan target to `/move_arm_p`. The executor reads the current arm-link TF,
splits position linearly and orientation with quaternion SLERP, and sends two
segments in sequence. The default remains plan-only:

```bash
source /opt/ros/humble/setup.zsh
source "$MOVEIT2_WS/install/setup.zsh"
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
  "{request_id: grasp_1, target_label: 0, arm: right, publish_pose: true, detection_timeout_sec: 20.0, dry_run: false}"
```

Execute the configured place flow:

```bash
ros2 action send_goal --feedback \
  /execute_place mission_interfaces/action/ExecutePlace \
  "{request_id: place_1, arm: right, dry_run: false}"
```

Move the chassis right at 0.3 m/s for 3 seconds:

```bash
ros2 action send_goal --feedback \
  /move_chassis mission_interfaces/action/MoveChassis \
  "{direction: right}"
```

Execute the complete box place flow after a successful real box grasp:

```bash
ros2 action send_goal --feedback \
  /execute_box_place mission_interfaces/action/ExecuteBoxPlace \
  "{request_id: box_place_1, dry_run: false}"
```

Execute the fixed level-4 box stack cycle:

```bash
ros2 action send_goal --feedback \
  /execute_box_stack mission_interfaces/action/ExecuteBoxStack \
  "{level: 4}"
```

For a safe integration check, use `dry_run: true`. Direct chassis, torso,
arm-joint, and gripper commands are skipped. Grasp detection and `/move_arm_p`
planning still run. Box grasp still calls `/pickup_task` for planning, while box
place calls `/go_ready` with its `dry_run` flag set.

The repository also contains an isolated mock integration fixture under
`mission_controller/test`. Run it on a non-production `ROS_DOMAIN_ID`; it
checks material grasp/place and the complete box grasp/place ordering without
connecting to robot nodes.
