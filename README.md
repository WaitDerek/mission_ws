# Changan mission workspace

这个工作区只保留 RealBot 双臂的两个 Mission action：

- `/execute_box_grasp` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/execute_box_place` (`mission_interfaces/action/ExecuteBoxPlace`)

关节名称与 `dual_arm_ws` 的 `realbot` profile 保持一致：

```text
left : L_JOINT_1 ... L_JOINT_7
right: R_JOINT_1 ... R_JOINT_7
```

## 环境

在 mission workspace 的源代码目录执行：

```zsh
source ./setup_all.zsh
```

脚本会按顺序加载 ROS 2、`rm_robot_ws`、`dual_arm_ws` 和当前
`mission_ws`。路径从脚本位置自动推导，不依赖机器用户名或固定工作区路径。

## 编译

```zsh
cd ../..
colcon build --merge-install --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release
```

## 启动

先启动 RealBot 双臂 bringup，再启动 Mission：

```zsh
ros2 launch mission_controller mission.launch.py \
  require_command_subscribers:=true
```

若需要由一个 launch 同时启动双臂、FoundationPose box perception 和 Mission：

```zsh
ros2 launch mission_controller mission_system.launch.py \
  mode:=hardware \
  enable_rviz:=false \
  enable_robot_state_publisher:=false
```

`mission_system.launch.py` 只启动 box perception，不再包含旧的 grasp、stack
或 chassis 流程。`object_pose_ros` 需要先被单独构建并加载。

## 发送 action

```zsh
ros2 action send_goal --feedback \
  /execute_box_grasp \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{dry_run: false}"

ros2 action send_goal --feedback \
  /execute_box_place \
  mission_interfaces/action/ExecuteBoxPlace \
  "{dry_run: false}"
```

Box grasp 会使用 `box_grasp_*_joint_positions` 进入观测姿态，调用
`/object_pose/estimate` 和 `/pickup_task`，闭合双夹爪并确认反馈后抬升躯干。
Box place 会释放夹爪并按配置的 RealBot 躯干目标复位。
