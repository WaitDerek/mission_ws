# R1 Pro 料箱抓取测试

所有终端先加载 ROS 2 和 MoveIt 2：

```bash
source /opt/ros/humble/setup.zsh
source /home/dekc/libraries/ws_moveit2/install/setup.zsh
```

vision 终端使用 `foundationpose` conda 环境；dual_arm 和 mission 终端使用
`changan` conda 环境。

## 1. 启动仿真

在 dual_arm 终端启动带虚拟控制器的 MoveIt 规划环境：

```bash
source /home/dekc/april/changan/dual_arm_ws/install/setup.zsh
ros2 launch robot_bringup planning_only.launch.py \
  robot_profile:=r1_pro \
  planning_pipeline:=ompl \
  dry_run:=false \
  enable_fake_ros2_control:=true \
  enable_rviz:=true
```

在 vision 终端启动 FoundationPose：

```bash
source /home/dekc/april/changan/vision_ws/install/setup.zsh
ros2 launch object_pose_ros object_pose_action.launch.py server_output:=screen
```

在 mission 终端启动任务节点，不重复拉起感知进程：

```bash
source /home/dekc/april/changan/dual_arm_ws/install/setup.zsh
source /home/dekc/april/changan/grasp_ws/install/setup.zsh
source /home/dekc/april/changan/vision_ws/install/setup.zsh
source /home/dekc/april/changan/mission_ws/install/setup.zsh
ros2 launch mission_controller mission.launch.py \
  start_perception:=false \
  start_detector_daemon:=false
```

## 2. 到达初始观察位姿

`dry_run: false` 会让虚拟控制器在 RViz 中执行轨迹：

```bash
ros2 action send_goal --feedback \
  /move_arm_j \
  task_interfaces/action/MoveArmJoints \
  "{left_joints: [-0.88, 0.84, -1.13, -1.80, 1.25, 0.29, 0.13],
    right_joints: [0.16, -0.04, 0.20, -2.095894, 0.174647, -0.718606, -0.094098],
    dry_run: false,
    duration: 5.0}"
```

## 3. 使用相机结果规划料箱抓取

该命令会调用 FoundationPose，将料箱几何中心变换到 `torso_link4`，再把
位姿交给 `/pickup_task`。`dry_run: true` 只规划并显示目标，不执行轨迹。

```bash
ros2 action send_goal --feedback \
  /execute_bin_grasp \
  mission_interfaces/action/ExecuteBinGrasp \
  "{request_id: 'f320_camera_sim_test',
    target_frame: 'torso_link4',
    target_label: -1,
    arm: 'right',
    publish_pose: true,
    detection_timeout_sec: 120.0,
    dry_run: true}"
```

## 4. 使用固定料箱位姿单独测试 pickup

```bash
ros2 action send_goal --feedback \
  /pickup_task \
  task_interfaces/action/PickupTask \
  "{box_pose:
      {header: {frame_id: 'torso_link4'},
       pose:
         {position: {x: 0.64229, y: 0.0, z: -0.03098},
          orientation: {x: 0.589368, y: 0.390699, z: -0.390699, w: 0.589368}}},
    box_width: 0.357,
    box_height: 0.127,
    box_type: 'vertical_down_gripper_test',
    dry_run: true}"
```

## 实物测试

当前提交验证了编译、mock 任务链和仿真规划接口，尚未验证实物执行。完成
RViz 规划检查后，再在实物控制栈中使用 `dry_run: false`。
