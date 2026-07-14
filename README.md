# Changan mission workspace

This ROS 2 workspace owns mission-level robot sequencing. Grasp perception stays
in the separate `grasp_ws`; dual-arm planning and execution stay in the separate
dual-arm workspace.

## Actions

- `/execute_grasp` (`mission_interfaces/action/ExecuteGrasp`)
- `/execute_place` (`mission_interfaces/action/ExecutePlace`)

Only one mission is accepted at a time.

### Grasp sequence

1. Open the selected gripper, publish the configured torso target, and call
   `/move_arm_j` with the configured dual-arm preparation joints.
2. Call `/detect_grasp_pose` and receive a pose in `torso_link` by default.
3. Send that pose to the `/move_arm_p` action for the selected arm.
4. Close the selected gripper.
5. Publish the torso reset target. The arms remain in the grasp pose so the
   object can be carried.

### Place sequence

1. Publish a constant chassis velocity for the configured distance and duration,
   then always publish zero velocity.
2. Publish the configured torso target and call `/move_arm_j` with the place
   joint target.
3. Open the selected gripper.
4. Call `/home` to reset both arms and publish the torso reset target.

The chassis move is open-loop. Calibrate the distance, duration, direction, and
joint targets in `mission_controller/config/mission.yaml` before hardware use.

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
