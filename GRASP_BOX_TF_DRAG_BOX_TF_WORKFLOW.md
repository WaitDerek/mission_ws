# GraspBox TF、DragBox TF 与自动 Workflow 使用说明

本文档对应当前 `mission_controller` 版本，整理以下内容：

- `/grasp_box_tf`
- `/execute_drag_box_grasp_tf`
- `/execute_workflow`
- 主要参数、调用方式和故障排查

除非特别说明，ROS 参数节点均为 `/mission_controller`；顶层 Workflow 参数节点为 `/execute_workflow`。

## 1. 启动

```bash
cd /rm_nvme/recordings/code/mission_ws
source ./setup_mission_env.sh
```

完整硬件启动：

```bash
ros2 launch mission_controller mission_system.launch.py \
  mode:=hardware \
  direct_motion_backend:=python_sdk \
  enable_global_tf:=true \
  enable_robot_state_publisher:=true \
  enable_box_perception:=true \
  enable_rviz:=false
```

如果视觉由外部进程单独启动，则使用 `enable_box_perception:=false`。Mission 默认在启动后延迟约 30 秒进入工作状态。

确认 Action：

```bash
ros2 action list | sort
```

## 2. Action 调用

### GraspBox TF

```bash
ros2 action send_goal --feedback \
  /grasp_box_tf \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'grasp-tf-test', target_label: 0, box_layer: 1, box_type: 'bigbox', dry_run: false}"
```

### DragBox TF

```bash
ros2 action send_goal --feedback \
  /execute_drag_box_grasp_tf \
  mission_interfaces/action/ExecuteDragBoxGrasp \
  "{request_id: 'drag-tf-test', target_label: 0, box_layer: 1, box_type: 'bigbox', dry_run: false}"
```

`box_type` 支持 `bigbox` 和 `smallbox`，`box_layer` 支持 1～4。 `dry_run:true` 只计算和反馈目标，不执行实机运动。

## 3. GraspBox TF 流程

1. 右臂移动到对应料箱层的检测姿态。
2. 检测前稳定等待，调用 `/object_pose/estimate\)，检测后再次稳定等待。
3. FoundationPose Pose 通过 TF 转换到 `base_link\)，不是只修改 `frame_id`。
4. 使用实时 `L_base_Link`、`R_base_Link` 和末端 TF 计算左右 Link7 目标。
5. 腰部 Joint1/2/3 移动到当前层目标角度。
6. 左右臂执行初始 MoveJ_P。
7. Step1 执行双臂力控夹紧（若 `force_clamp_mode=closed_loop`）。
8. 按 `box_post_movel_step_count` 执行 Step2～Step5。
9. 若开启 `grasp_box_tf_body_home_carry_enabled`，腰部回零并由双臂跟随，保持料箱刚性姿态。

当前默认关键值：

```yaml
grasp_box_tf_detection_arm: right
grasp_box_tf_force_clamp_mode: closed_loop
grasp_box_tf_body_home_carry_enabled: true
grasp_box_tf_body_home_carry_continuous_enabled: true
box_post_movel_step_count: 2
```

## 4. DragBox TF 流程

1. 左臂移动到检测姿态并调用 FoundationPose。
2. TF 转换检测结果，腰部移动到对应层。
3. 右臂 MoveJ_P 到料箱侧面。
4. 右臂执行 Step1 力控夹紧。
5. 右臂执行 Drag1、Drag2、Drag3。
6. 根据 Drag3 后右臂实际 TF 重新估计料箱位姿。
7. 左臂先执行加入前 MoveJ，再执行加入料箱的 MoveJ_P。
8. 左臂加入后执行双臂 Step1。
9. 双臂执行 Step2 及后续步骤。
10. 若开启 `drag_box_tf_body_home_carry_enabled`，执行腰部回零跟随。

当前默认关键值：

```yaml
drag_box_tf_detection_arm: left
drag_box_left_arm_enabled: true
drag_box_left_join_mode: after_drag3
drag_box_left_join_motion_mode: movej_p
drag_box_tf_reanchor_after_drag3_enabled: true
drag_box_tf_force_clamp_mode: closed_loop
drag_box_tf_body_home_carry_enabled: true
```

## 5. 感知和 TF 参数

| 参数 | 当前默认值 | 说明 |
|---|---:|---|
| `box_object_pose_action_name` | `/object_pose/estimate` | FoundationPose Action |
| `box_object_pose_confidence_threshold` | `0.25` | 感知置信度阈值 |
| `box_object_pose_instance_index` | `0` | 目标实例序号 |
| `box_foundation_pose_pre_settle_sec` | `5.0` | 检测前等待时间 |
| `box_foundation_pose_post_settle_sec` | `5.0` | 检测后等待时间 |
| `grasp_box_tf_freeze_frame` | `base_link` | 冻结检测 Pose 的参考坐标系 |
| `grasp_box_tf_detection_tf_timeout_sec` | `5.0` | 检测 Pose TF 转换超时 |
| `grasp_box_tf_runtime_tf_timeout_sec` | `5.0` | 运行时 TF 查询超时 |
| `left_arm_base_frame` | `L_base_Link` | 左臂基座 TF |
| `right_arm_base_frame` | `R_base_Link` | 右臂基座 TF |
| `left_ee_frame` | `left_arm_8_Link` | 左臂末端 Link7 |
| `right_ee_frame` | `right_arm_8_Link` | 右臂末端 Link7 |
| `box_tf_equalize_dual_target_z_enabled` | `true` | 双臂目标数值 Z 统一 |

目标变换逻辑：

```text
相机 Pose → base_link → 左/右 arm base → 左/右 Link7 目标 Pose
```

## 6. 分层参数

以下记号中：

```text
<type> = bigbox 或 smallbox
<N> = 1、2、3、4
```

### GraspBox TF

```text
grasp_box_tf_box_layer_pre_detection_right_movej_joint_units_<type>_layer<N>
grasp_box_tf_box_layer_joint1_approach_angle_deg_<type>_layer<N>
grasp_box_tf_box_layer_joint2_approach_angle_deg_<type>_layer<N>
grasp_box_tf_box_layer_joint3_approach_angle_deg_<type>_layer<N>
grasp_box_tf_direct_movel_left_offset_xyz_<type>_layer<N>
grasp_box_tf_direct_movel_right_offset_xyz_<type>_layer<N>
grasp_box_tf_joint123_left_target_correction_pose_box_<type>_layer<N>
grasp_box_tf_joint123_right_target_correction_pose_box_<type>_layer<N>
```

后续位移：

```text
grasp_box_tf_post_movel_left_step1_xyz_<type>_layer<N>
grasp_box_tf_post_movel_right_step1_xyz_<type>_layer<N>
...
grasp_box_tf_post_movel_left_step5_xyz_<type>_layer<N>
grasp_box_tf_post_movel_right_step5_xyz_<type>_layer<N>
```

### DragBox TF

```text
drag_box_tf_box_layer_pre_detection_left_movej_joint_units_<type>_layer<N>
drag_box_tf_box_layer_pre_detection_right_movej_joint_units_<type>_layer<N>
```

标准 Step：

```text
drag_box_tf_post_movel_left_step<N>_xyz_<type>_layer<M>
drag_box_tf_post_movel_right_step<N>_xyz_<type>_layer<M>
```

抽拉专用参数：

```text
drag_box_tf_post_movel_step_drag1_left_xyz_<type>_layer<N>
drag_box_tf_post_movel_step_drag1_right_xyz_<type>_layer<N>
drag_box_tf_post_movel_step_drag2_left_xyz_<type>_layer<N>
drag_box_tf_post_movel_step_drag2_right_xyz_<type>_layer<N>
drag_box_tf_post_movel_step_drag3_left_xyz_<type>_layer<N>
drag_box_tf_post_movel_step_drag3_right_xyz_<type>_layer<N>
```

`offset_xyz` 和 `target_correction_pose_box` 均基于料箱坐标系；补偿 Pose 格式为：

```text
[x, y, z, qx, qy, qz, qw]
```

## 7. 腰部回零跟随

GraspBox 使用前缀：

```text
grasp_box_tf_body_home_carry_*
```

DragBox 使用前缀：

```text
drag_box_tf_body_home_carry_*
```

| 参数后缀 | 作用 |
|---|---|
| `enabled` | 总开关 |
| `continuous_enabled` | 连续轨迹或分段轨迹 |
| `segments` | 轨迹段数 |
| `arm_motion_mode` | `movel` 或 `movej_p` |
| `body_velocity` | 腰部 MoveJ 速度 |
| `left_movel_velocity_percent` | 左臂跟随速度 |
| `right_movel_velocity_percent` | 右臂跟随速度 |
| `body_blend_radius` | 腰部交融半径 |
| `arm_blend_radius` | 手臂交融半径 |
| `position_tolerance_m` | 位置容差 |
| `orientation_tolerance_rad` | 姿态容差 |
| `timeout_sec` | 跟随总超时 |
| `tf_timeout_sec` | TF 查询超时 |
| `final_correction_enabled` | 最终末端修正开关 |
| `arm_start_delay_sec` | 腰部发布后手臂延迟时间 |
| `arm_start_lead_sec` | 手臂提前启动时间 |

旧路径 `box_step2_waist_endpoint_sync_enabled` 当前默认关闭，不应与新的 TF Carry 同时开启。

## 8. 力控参数

两套系统分别使用：

```text
grasp_box_tf_force_clamp_*
drag_box_tf_force_clamp_*
```

| 参数后缀 | 作用 |
|---|---|
| `mode` | `disabled`、`monitor_only`、`closed_loop` |
| `contact_threshold_left/right_counts` | 接触阈值 |
| `clamped_threshold_left/right_counts` | 夹紧确认阈值 |
| `hold_threshold_left/right_counts` | 保持阈值 |
| `emergency_threshold_left/right_counts` | 紧急停止阈值 |
| `max_distance_left/right_m` | 最大夹紧搜索距离 |
| `search_step_m` | 粗搜索步长 |
| `fine_step_m` | 精搜索步长 |
| `movel_velocity_percent` | 力控 MoveL 速度 |
| `sensor_max_age_sec` | 力传感器数据最大允许延迟 |
| `force_sign_left/right` | 力方向符号 |
| `timeout_sec` | 力控总超时 |
| `stop_after_clamp_confirmed` | 夹紧确认后是否停止 |

当前默认力控模式为 `closed_loop`。GraspBox Step1 为双臂夹紧；DragBox 先右臂夹紧，左臂加入后再进行双臂夹紧。

## 9. DragBox 左臂加入

| 参数 | 当前默认值 | 说明 |
|---|---:|---|
| `drag_box_left_arm_enabled` | `true` | 是否启用左臂 |
| `drag_box_left_join_mode` | `after_drag3` | Drag3 后加入 |
| `drag_box_left_join_motion_mode` | `movej_p` | 左臂加入方式 |
| `drag_box_left_join_pre_movej_enabled` | `true` | 加入前是否先 MoveJ |
| `drag_box_left_join_velocity_percent` | `10.0` | 加入速度 |
| `drag_box_left_join_timeout_sec` | `60.0` | 加入超时 |
| `drag_box_tf_reanchor_after_drag3_enabled` | `true` | Drag3 后根据右臂实际 TF 重算料箱和左臂目标 |

## 10. 顶层 Workflow

调用：

```bash
ros2 action send_goal --feedback \
  /execute_workflow \
  mission_interfaces/action/ExecuteWorkflow \
  "{start: true}"
```

Workflow 只接受 `start:true`，`workflow_id` 由系统自动生成，并通过 lease 防止并发任务。

执行逻辑：

```text
获取 lease
→ 导航到观察点1
→ 调用 /depalletizing/observe
→ 点1有计划：处理点1，再处理点3
  点1无计划：处理点2，再处理点4
→ 每个料箱：导航到抓取点 → 调用抓取 Action
→ 导航到放置点16 → 调用 /execute_box_place
→ 释放 lease → 返回统计结果
```

抓取点映射：

| 观察点 | 第一个堆位 | 第二个堆位 |
|---|---|---|
| 1 | 点6，GraspBox TF | 点5，DragBox TF |
| 2 | 点8，GraspBox TF | 点7，DragBox TF |
| 3 | 点10，GraspBox TF | 点9，DragBox TF |
| 4 | 点12，GraspBox TF | 点11，DragBox TF |

偶数抓取点使用 `/grasp_box_tf`，奇数抓取点使用 `/execute_drag_box_grasp_tf`。

Workflow 依赖：

```yaml
global_observation_action_name: /depalletizing/observe
navigation_adapter: mqtt
mqtt_start_enabled: true
mqtt_navigation_points_json: "{}"
```

当前最需要补齐的是 `mqtt_navigation_points_json`。为空时没有 1～16 号导航点，实际 Workflow 导航会失败。还必须保证 MQTT Broker、`/depalletizing/observe` 和相关抓取 Action 已启动。

MQTT 启动消息格式：

```json
{"robot_id":"realman-001","start":true,"request_id":"workflow-001"}
```

发送到 `mission/workflow/start`，状态发布到 `mission/workflow/status`。

## 11. 参数查询和临时修改

```bash
ros2 param get /mission_controller PARAMETER_NAME
ros2 param dump /mission_controller
ros2 param dump /execute_workflow
ros2 param set /mission_controller PARAMETER_NAME VALUE
```

例如：

```bash
ros2 param set /mission_controller box_post_movel_step_count 4
ros2 param set /mission_controller grasp_box_tf_body_home_carry_enabled false
ros2 param set /mission_controller grasp_box_tf_force_clamp_mode disabled
ros2 param set /mission_controller grasp_box_tf_body_home_carry_continuous_enabled true
```

`ros2 param set` 只修改当前运行进程。要变成启动默认值，需要修改 `config/mission/*.yaml`，重新构建并重启任务系统。

## 12. 常见问题

- Action 被拒绝：确认 Action server 已启动，且没有同一任务正在运行。
- 感知超时：确认 `/object_pose/estimate` 已启动，并检查相机/TF frame。
- TF 转换失败：检查 `base_link`、`L_base_Link`、`R_base_Link`、`left_arm_8_Link`、`right_arm_8_Link` 及相机链是否在同一 TF 树。
- 腰部回零失败：检查 `*_body_home_carry_enabled`、`*_carrier_frame`、SDK 连接和腰部反馈。
- 力控达到最大距离：检查力方向符号、阈值、夹具初始位置和 `max_distance_*_m`。
- Workflow 导航失败：检查 `navigation_adapter` 和 `mqtt_navigation_points_json` 是否配置完整。
