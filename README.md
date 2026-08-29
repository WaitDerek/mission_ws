# Changan mission workspace

这个工作区提供八个 RealBot Mission Action：

- `/execute_adaptive_box_grasp` (`mission_interfaces/action/ExecuteAdaptiveBoxGrasp`)
- `/execute_box_grasp` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/grasp_box_tf` (`mission_interfaces/action/ExecuteBoxGrasp`)
- `/execute_drag_box_grasp` (`mission_interfaces/action/ExecuteDragBoxGrasp`)
- `/execute_drag_box_grasp_tf` (`mission_interfaces/action/ExecuteDragBoxGrasp`)
- `/execute_box_place` (`mission_interfaces/action/ExecuteBoxPlace`)
- `/place_box_test` (`mission_interfaces/action/PlaceBoxTest`)
- `/execute_workflow` (`mission_interfaces/action/ExecuteWorkflow`)

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

`execute_workflow` 是 Mission launch 的固定节点，会与原 MissionController 一起启动，
无需额外 enable 参数；原有 ActionServer 的名称、Goal 和调用方式保持不变。

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
  "{request_id: manual-place-001, dry_run: false}"

ros2 action send_goal --feedback \
  /execute_workflow \
  mission_interfaces/action/ExecuteWorkflow \
  "{start: true}"
```

## 自动拆垛任务流

`/execute_workflow` 只接受 `start`。Mission 内部生成
`workflow_id`，并严格等待上一项的最终成功 Result 后才启动下一项：

```text
导航到全局观测点
  -> /depalletizing/observe
  -> 导航到操作点
  -> 奇数点调用 /execute_drag_box_grasp_tf
     偶数点调用 /grasp_box_tf
  -> 导航到固定放置点 16
  -> /execute_box_place
  -> 下一箱/下一个对称观测点
```

操作点沿栈板外围顺时针编号为 5～12。这里的左右始终以机器人在对应观测点
面向栈板时的视野为准：视野左摞使用偶数点直接抱，视野右摞使用奇数点抽拉，
因此 Vision 的 `(左摞, 右摞)` 映射仍为 `1->{6,5}`、`2->{8,7}`、
`3->{10,9}`、`4->{12,11}`。长边/短边及计划是否可执行仍由 Vision 的
`success` 和 `plan_valid` 决定；Mission 不使用 `top_box_camera_poses` 分析箱体姿态，
只用其中的位置核对当前结果确实是同一前排的左右两摞：两摞必须均为 front、列号
为 0/1、相机 frame 相同、左右间距足够且深度差不超过配置阈值。Mission 随后只用
ID、层数、左右列和 size 规划任务。1 有有效计划时继续 3；1 无计划时尝试 2，2 有
有效计划时继续 4。点 13～15 保留，点 16 为通用放置位置。

前排核对默认启用，左右最小间距为 `0.20 m`，两摞最大深度差为 `0.35 m`，相机
前向最大深度为 `1.20 m`。现场相机距离不同，应据实调整
`global_observation_front_max_camera_depth_m`；设置为 `0` 只关闭绝对深度上限，其他
前排一致性检查仍然生效。

### MQTT 导航协议

`taskflow.yaml` 默认启用最小 MQTT 导航适配器，Broker 默认为
`127.0.0.1:1883`：

平台通过 MQTT 启动完整任务流：

- `execute_workflow` 节点启动后订阅 `mission/workflow/start`。
- 推荐发布 `{"id":0,"start":true,"request_id":"platform-001"}`；`id`
  缺省为 `0`，只有 `id=0` 才会调用任务流 Action。兼容纯文本 `start`、
  `true` 或 `1`，这些纯文本消息同样按 `id=0` 处理。
- Mission 收到后通过 ROS ActionClient 调用现有
  `/execute_workflow`，因此仍经过原有 goal 校验、Mission lease、取消和
  严格串行状态机。
- `mission/workflow/status` 会依次发布 JSON 状态，`event` 包括 `received`、
  `accepted`、`feedback`、`rejected` 和最终 `result`。最终消息包含
  `success`、`workflow_id`、`final_stage`、完成观测数和完成箱数。
- 同一时刻只接受一个 MQTT 启动请求；任务执行中收到的新启动消息会返回
  `rejected`。

示例启动消息：

```json
{"id":0,"start":true,"request_id":"platform-001"}
```

ROS 节点本身仍需先通过 `mission.launch.py` 或 `mission_system.launch.py` 启动；MQTT
消息负责启动顶层任务流 Action，而不是启动 ROS 进程。

任务流内部需要导航时：

- Mission 向 `mission/navigation/request` 发布点位及地图位姿，例如
  `{"id":5,"frame_id":"map","x":1.2,"y":3.4,"yaw":1.57}`。
- 平台到点后向 `mission/navigation/result` 返回相同纯文本 ID，例如 `5`，即为成功。
- 平台也可返回 JSON：
  `{"id":"5","success":true,"message":"arrived"}`；将 `success` 设为
  `false` 可让任务流在当前导航步骤失败并停止。
- 任务流同一时刻只等待一个导航结果，忽略其他 ID 和 retained 旧消息；默认 QoS 1、
  连接超时 10 秒、单次导航超时 300 秒。

Broker、Topic、QoS 和超时均可在
`mission_controller/config/mission/taskflow.yaml` 修改。若需要禁用平台导航，设置
`navigation_adapter: disabled`，任务流会在导航步骤明确失败，不会模拟到点成功。
点位坐标通过同文件中的 `mqtt_navigation_points_json` 配置，格式如下：

```json
{"1":{"x":1.0,"y":2.0,"yaw":0.0},"5":{"x":3.0,"y":4.0,"yaw":1.57}}
```

仓库不包含现场测量坐标，因此默认值为空对象。任务流请求未配置的点位时会明确失败，
且不会向平台发布 `(0,0,0)` 等可能导致误移动的占位坐标。

任务流采用内部 Mission lease，防止平台在自动任务中间直接插入旧 Action。
异常退出后 lease 保持 fail-closed，需要受控重启 MissionController；第一版不做
TTL、自动恢复或 crash resume。

全局观测位置和 2x2 料箱关系见
[depalletizing-observation-layout.svg](./depalletizing-observation-layout.svg)。

Mission 参数已按职责拆到 `mission_controller/config/mission/`。launch 按固定顺序
加载这些片段，最后再加载 `config_file`，所以原有调用者覆盖优先级保持不变。

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
