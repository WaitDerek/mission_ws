# G1-D mission workspace

This workspace contains the G1-D badge tracking mission.  It exposes one
mission action:

```text
/execute_grasp  mission_interfaces/action/ExecuteGrasp
```

The mission does not start the detector.  Run the FoundationPose action server
from `vision_ws` in its `foundationpose` environment, then start this mission
with the G1-D dual-arm bringup.

## Sequence

1. Send `LEFT_JOINT_WAYPOINTS` through `/move_arm_j` and wait for a fresh
   `/pinocchio_g1d/left_ee_pose` measurement.
2. Request the `badge` pose from `/object_pose/estimate`.
3. Compose the measured
   `torso_link -> left_gripper_base_link` pose with the hand-eye matrix,
   camera-frame badge pose, and the configured `obj_T_tar` matrix.
4. Send the resulting `left_gripper_base_link` target to `/move_arm_p`.
5. Return to `LEFT_JOINT_WAYPOINTS` after the target motion completes.

The target transform is configured in `mission_controller/config/mission.yaml`.
The camera extrinsic is loaded from the YAML file named by `handeye_file`
(`mission_controller/config/handeye_result_12.yaml` by default).  The
controller publishes the raw camera pose, the torso-frame badge pose, the
calculated target, and RViz axis markers for inspection.

To select another calibration file in the same config directory, override only
the filename:

```bash
ros2 launch mission_controller mission_system.launch.py \
  handeye_file:=handeye_result_12.yaml
```

## Build

```bash
# Source the ROS 2 environment using the installation method of the machine.
# Set MOVEIT2_WS only when ws_moveit2 is not in the sibling layout expected by
# src/setup_all.zsh.
cd mission_ws
source src/setup_all.zsh
colcon build --merge-install --symlink-install \
  --cmake-args "-DCMAKE_BUILD_TYPE=Release" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.zsh
```

On a fresh checkout, `src/setup_all.zsh` may report that the mission install
tree is not present yet; this is expected before the first build.  Source it
again after building so the new mission package is overlaid.

## Launch

Simulation (RViz and fake ros2_control enabled by default):

```bash
ros2 launch mission_controller mission_system.launch.py \
  simulation:=true hardware:=false
```

Hardware:

```bash
ros2 launch mission_controller mission_system.launch.py \
  simulation:=false hardware:=true robot_ip:=enP8p1s0
```

The detector remains a separate process.  Once `/object_pose/estimate` is
available, call the mission action; legacy goal fields are accepted for CLI
compatibility, but G1-D always uses the left arm and the configured `badge`
model:

```bash
ros2 action send_goal --feedback \
  /execute_grasp \
  mission_interfaces/action/ExecuteGrasp \
  "{request_id: 'g1d_badge_test',
    target_label: 0,
    arm: 'left',
    publish_pose: true,
    detection_timeout_sec: 120.0,
    dry_run: false}"
```

Use `dry_run: true` to exercise detection, transform calculation, and RViz
publishing without moving the arm.
