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
colcon build --base-paths src/mission_interfaces src/mission_controller \
  --merge-install --symlink-install \
  --cmake-args "-DCMAKE_BUILD_TYPE=Release" \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
source install/setup.zsh
```

Mission 提供七个任务 Action 端点：

```text
/execute_grasp  mission_interfaces/action/ExecuteGrasp
/run_grip      mission_interfaces/action/ExecuteGrip
/run_peel      mission_interfaces/action/ExecutePeel
/execute_grip   mission_interfaces/action/ExecuteGrip
/execute_peel   mission_interfaces/action/ExecutePeel
/execute_assembly mission_interfaces/action/ExecuteAssembly
/execute_workflow mission_interfaces/action/ExecuteWorkflow
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

./start_object_pose_action.sh \
  --actions all \
  --config g1d \
  --camera-prefix /d405
```

启动后应存在：

```text
/object_pose/estimate
/front_bumper_pose/estimate
```

当前 mission 请求的模型标签由配置文件设置为 `badge`。

## 5. 启动前检查

```bash
ros2 action list | rg 'execute_grasp|run_grip|run_peel|execute_grip|execute_peel|execute_assembly|execute_workflow|move_arm_j|move_arm_p|object_pose|front_bumper'
ros2 topic echo /pinocchio_g1d/left_ee_pose --once
ros2 topic echo /pinocchio_g1d/right_ee_pose --once
```

至少应看到：

```text
/execute_grasp
/run_grip
/run_peel
/execute_grip
/execute_peel
/execute_assembly
/execute_workflow
/move_arm_j
/move_arm_p
/object_pose/estimate
/front_bumper_pose/estimate
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

## 7. 调用 grip 和 peel

两个 Action 的动作顺序、轨迹、手眼参数、吸盘通道和力阈值与
`g1d-task-master` 中原有 `GripBadgePipeline.run()`、`PeelPipeline.run()`
保持一致，原目录不参与修改。

`/run_grip`、`/run_peel` 是保留原行为的一次性 Action；
`execute_grip.py`、`execute_peel.py` 是两套独立的 ROS Action 实现，不会调用
`run_*`。迁移期间长期 Mission 节点同时保留两套旧流程 Action，后续再逐步删除；
正式 Workflow 只调用 `Execute*` Action。Action 版本额外要求新鲜末端位姿、
继电器新回执和有效力接触，并处理子 Action 超时/取消及失败后的吸盘清理。

连接件使用相同抓取步骤，但几何配置独立受
`mission_controller/config/connector_grip.json` 的 `calibrated` 开关保护；
完成连接件抓取偏移标定前，`target_type: connector` 会失败关闭。

```bash
ros2 action send_goal --feedback \
  /run_grip \
  mission_interfaces/action/ExecuteGrip \
  "{request_id: 'run_grip_test', target_type: 'badge'}"

ros2 action send_goal --feedback \
  /run_peel \
  mission_interfaces/action/ExecutePeel \
  "{request_id: 'run_peel_test'}"

ros2 action send_goal --feedback \
  /execute_grip \
  mission_interfaces/action/ExecuteGrip \
  "{request_id: 'grip_test', target_type: 'badge'}"

ros2 action send_goal --feedback \
  /execute_peel \
  mission_interfaces/action/ExecutePeel \
  "{request_id: 'peel_test'}"
```

调用前需要启动力传感器和 USB 继电器驱动，并确保以下接口可用：

```text
/pinocchio_g1d/left_ee_pose
/pinocchio_g1d/right_ee_pose
/force_torque/data
/arto/usb_relay_ctrl_goal
/arto/usb_relay_ctrl_result
/move_arm_j
/move_arm_p
/move_arm_l
/object_pose/estimate
```

上述 Pinocchio 末端位姿供新的 `execute_grip`、`execute_peel` 使用。迁移期保留的
`run_grip`、`run_peel` 继续按 `g1d-task-master` 原实现订阅：

```text
/left_ee_pose
/right_ee_pose
```

两组订阅和缓存彼此独立，`run_*` 不读取 `execute_*` 的末端位姿缓存。

Grip 实际顺序：

```text
双臂 /move_arm_j 到准备位
-> 读取左末端 /pinocchio_g1d/left_ee_pose
-> /object_pose/estimate 检测 badge 或 badge_connector
-> 计算 down/up 目标
-> 左臂 /move_arm_p 到吸取位
-> 打开左吸盘（继电器 1、2）
-> 获取新的 Fz 基线
-> 左臂 /move_arm_l 沿末端局部 +Y 最多 0.10 m
-> |ΔFz| >= 5.0 时取消直线运动
-> 左臂 /move_arm_l 撤回 up 目标
```

这里列出的是 `execute_grip` 的位姿话题；旧 `run_grip` 的相同步骤读取
`/left_ee_pose`。

Peel 实际顺序：

```text
双臂 /move_arm_j 到撕膜准备位
-> 获取新的左右末端位姿
-> /object_pose/estimate 检测 badge_back
-> 右臂 /move_arm_p 到膜上方
-> 打开右吸盘（继电器 3、4）
-> 获取新的 Fz 基线
-> 再获取新的左右末端位姿
-> 右臂 /move_arm_l 沿末端局部 +Y 最多 0.038 m
-> |ΔFz| >= 3.0 时取消直线运动
-> 再获取新的左右末端位姿
-> 双臂 /move_arm_p 平移并相向旋转 15° 完成撕膜
-> 关闭右吸盘
```

这里列出的是 `execute_peel` 的位姿话题；旧 `run_peel` 对应读取
`/left_ee_pose` 和 `/right_ee_pose`。

## 8. 调用 assembly

```bash
ros2 action send_goal --feedback \
  /execute_assembly \
  mission_interfaces/action/ExecuteAssembly \
  "{request_id: 'assembly_test', target_type: 'connector'}"
```

Assembly 流程：

```text
双臂 /move_arm_j 抬臂到安装准备位
-> 获取新的左末端位姿
-> /front_bumper_pose/estimate 检测 connector 或 badge_bracket
-> 计算左夹具预安装位和最终安装位
-> /move_arm_p 到预安装位
-> 获取新的 Fz 基线
-> /move_arm_l 向最终位插入并持续监测 |ΔFz|
-> 达到阈值后取消直线运动，避免继续压迫前保
-> 关闭左吸盘释放连接件或车标
```

`mission_controller/config/assembly.json` 默认 `calibrated: false`。必须完成
前保 patch、`task_T_tool` 和力阈值实机标定后才能改为 `true`。

## 9. MQTT 任务流

标准顺序：

```text
点1 连接件位 -> grip(connector)
点3 前保位   -> assembly(connector)
点2 车标位   -> grip(badge) -> peel
点3 前保位   -> assembly(badge)
```

点4预留。所有导航通过 MQTT，任务严格等待上一项成功后才进入下一项。

平台启动消息，Topic `mission/workflow/start`：

```json
{"robot_id":"6","start":true,"request_id":"platform-001"}
```

`robot_id` 必须与 `taskflow.yaml` 中配置的机器人 ID 一致。

Mission 状态 Topic：`mission/workflow/status`。导航请求 Topic：
`mission/navigation/request`：

```json
{"id":1,"frame_id":"map","pos":[2.14,-2.84,-2.89]}
```

平台导航结果 Topic：`mission/navigation/result`：

```json
{"robot_id":"6","success":true,"message":"arrived"}
```

导航请求中的 `id` 是点位编号；导航结果通过 `robot_id` 匹配当前机器人，并使用
`success=false` 明确报告失败。其他机器人回执、遗留消息、失败回执或超时都不会
放行下一步；ROS 子 Action 同样必须以 `SUCCEEDED` 结束且返回 `success=true`。

实际点位坐标填写在 `mission_controller/config/taskflow.yaml` 的
`mqtt_navigation_points_json`。1～4 任一点未配置或 `paho-mqtt` 不可用时，
workflow 节点拒绝启动，不发布零位导航。

本地调试可直接调用同一状态机：

```bash
ros2 action send_goal --feedback \
  /execute_workflow \
  mission_interfaces/action/ExecuteWorkflow \
  "{start: true}"
```

## 10. execute_grasp 内部流程

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

## 11. 可视化 topic

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

## 12. 常见问题

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
