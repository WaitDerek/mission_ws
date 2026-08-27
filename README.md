# Changan mission workspace

这个工作区提供六个 RealBot 双臂 Mission Action：

- `/execute_adaptive_box_grasp` (`mission_interfaces/action/ExecuteAdaptiveBoxGrasp`)
- `/execute_box_grasp` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/grasp_box_tf` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/execute_drag_box_grasp` (`mission_interfaces/action/ExecuteDragBoxGrasp`)
- `/execute_drag_box_grasp_tf` (`mission_interfaces/action/ExecuteDragBoxGrasp`)
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
  /execute_adaptive_box_grasp \
  mission_interfaces/action/ExecuteAdaptiveBoxGrasp \
  "{task_id: box-001, target_instance_index: -1, dry_run: true}"

ros2 action send_goal --feedback \
  /execute_box_grasp \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{dry_run: false}"

ros2 action send_goal --feedback \
  /execute_box_place \
  mission_interfaces/action/ExecuteBoxPlace \
  "{dry_run: false}"
```

`/execute_adaptive_box_grasp` 的任务流是：

1. 调用 `/object_pose/estimate`，严格使用检测结果自带时间戳查询 TF，仅一次转换并冻结 `object_pose_base`。
2. 由箱体 Pose、宽度和抓取参数生成并冻结 `left_grasp_pose_base`、`right_grasp_pose_base`。
3. 将左右抓取位从 `base_link` 转换到各自机械臂基座，并用夹具中心偏移换算为 `Link8` SDK 目标。
4. 直接并发调用左右 RealMan Python SDK `rm_movel` 到达抓取位置。
5. 将冻结的左右抓取位置沿 `base_link +Z` 偏移，再执行第二次 `rm_movel` 完成抬升。

这个 Action 不调用 `/move_arm_p`、`/move_torso_p` 或其他 dual_arm 规划接口，也不自动移动腰部、闭合夹爪或验证夹取结果。`dry_run=true` 只验证检测、冻结坐标及 SDK 目标换算，不连接机械臂或发送物理运动。

因为绕过了 MoveIt，这条路径没有 IK 预检查、碰撞检查或轨迹规划；SDK 拒绝不可达目标时 Action 会执行双臂 slow-stop 并失败。底盘、腰部和目标物在整个 Action 中必须保持不动，否则冻结到 `base_link` 的目标不再有效。

Box grasp 会使用 `box_grasp_*_joint_positions` 进入观测姿态，调用
`/object_pose/estimate` 和 `/pickup_task`，闭合双夹爪并确认反馈后抬升躯干。
Box place 会释放夹爪并按配置的 RealBot 躯干目标复位。
