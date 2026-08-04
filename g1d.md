# G1-D 车标抓取流程

本文说明 G1-D 的仿真、实物启动和 `execute_grasp` 调用流程。

## 1. 工作区关系

仓库目录保持以下相邻结构：

```text
dual_arm_ws/
vision_ws/
mission_ws/
```

`mission_ws/src/setup_all.zsh` 会根据脚本位置加载 dual-arm、vision、mission 和
MoveIt 工作区，不依赖机器绝对路径。首次使用前，先按当前机器的方式加载 ROS 2
环境；MoveIt 工作区不在默认相邻位置时设置 `MOVEIT2_WS`。

```bash
cd mission_ws
source src/setup_all.zsh
```

## 2. 编译 mission

```bash
cd mission_ws
colcon build --merge-install --symlink-install \
  --cmake-args "-DCMAKE_BUILD_TYPE=Release" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.zsh
```

`mission_interfaces` 只生成一个 action：

```text
/execute_grasp  mission_interfaces/action/ExecuteGrasp
```

## 3. 启动 dual-arm 和 mission

推荐使用组合 launch。它会启动 dual-arm、MoveIt、Pinocchio FK、RViz 和 mission
controller，不需要再单独启动 `robot_bringup` 或 `mission.launch.py`。

### 仿真

```bash
cd mission_ws
source src/setup_all.zsh
source install/setup.zsh

ros2 launch mission_controller mission_system.launch.py \
  simulation:=true \
  hardware:=false
```

默认使用：

```text
robot_profile: g1_d
planning_pipeline: stomp
FK: /pinocchio_g1d/left_ee_pose
FK: /pinocchio_g1d/right_ee_pose
```

若只想规划、不执行轨迹，可将 action goal 中的 `dry_run` 设为 `true`。

### 实物

```bash
cd mission_ws
source src/setup_all.zsh
source install/setup.zsh

ros2 launch mission_controller mission_system.launch.py \
  simulation:=false \
  hardware:=true \
  robot_ip:=enP8p1s0
```

`robot_ip` 对 G1-D 表示使用的网络接口名称，按实际机器修改。

## 4. 启动 vision

vision 使用 `foundationpose` conda 环境，必须单独启动。mission 不会自动启动
FoundationPose。

```bash
cd vision_ws
conda activate foundationpose
source install/setup.zsh

ros2 launch object_pose_ros object_pose_action.launch.py \
  camera_source:=d405 \
  camera_model:=d405 \
  server_output:=screen
```

启动后应存在：

```text
/object_pose/estimate
```

当前 mission 请求的模型标签由配置文件设置为 `badge`。

## 5. 启动前检查

```bash
ros2 action list | rg 'execute_grasp|move_arm_j|move_arm_p|object_pose'
ros2 topic echo /pinocchio_g1d/left_ee_pose --once
ros2 topic echo /pinocchio_g1d/right_ee_pose --once
```

至少应看到：

```text
/execute_grasp
/move_arm_j
/move_arm_p
/object_pose/estimate
```

## 6. 调用 execute_grasp

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

G1-D 固定使用左臂和 `badge` 模型。`target_label`、`arm`、`publish_pose`、
`detection_timeout_sec` 仅保留为命令行兼容字段。

## 7. action 内部流程

```text
1. /move_arm_j
   左臂移动到 LEFT_JOINT_WAYPOINTS，右臂保持当前位置。

2. /pinocchio_g1d/left_ee_pose
   等待新的左末端位姿，末端为 left_gripper_base_link。

3. /object_pose/estimate
   获取相机坐标系下的 badge 位姿。

4. 位姿计算
   T_torso_target =
     T_torso_ee * T_ee_camera * T_camera_badge * T_badge_target

5. /move_arm_p
   将计算出的目标发送给 left_gripper_base_link。

6. /move_arm_j
   执行完成后返回 LEFT_JOINT_WAYPOINTS。
```

手眼外参默认从以下文件读取：

```text
mission_controller/config/handeye_result_12.yaml
```

也可以传入同一 config 目录下的其他文件名：

```bash
ros2 launch mission_controller mission_system.launch.py \
  handeye_file:=handeye_result_12.yaml
```

## 8. 可视化 topic

```text
/mission/badge_pose_camera
/mission/badge_pose
/mission/badge_target_pose
/mission/grasp_visualization
```

- `badge_pose_camera`：相机原始检测结果
- `badge_pose`：转换到 `torso_link` 的车标位姿
- `badge_target_pose`：应用 `obj_T_tar` 后的左夹爪目标
- `grasp_visualization`：末端、相机、车标和目标坐标轴

## 9. 常见问题

### 没有 `/object_pose/estimate`

确认 vision 使用 `foundationpose` 环境启动，并确认相机输入正常。

### `dual_arm_controller` 缺少 `type`

这是 dual-arm bringup 的 controller 配置问题。MoveIt 可能仍能启动并规划，
但轨迹执行会失败，需要先修复 dual-arm controller 配置。

### 出现多个 `/execute_grasp` action server

不要同时启动 `mission_system.launch.py` 和单独的 `mission.launch.py`；组合 launch
已经包含 mission controller。

### hand-eye YAML 读取失败

确认文件位于 `mission_controller/config`，且包含：

```yaml
ee_to_camera:
  matrix:  # 4x4 homogeneous matrix
```
