# Mission Action 简明命令

以下命令在 `rm1` 上运行。先执行：

```bash
source /rm_nvme/recordings/code/setup_mission_env.sh
```

`/grasp_box_tf`：双臂直接抱箱，默认右臂观测。

```bash
ros2 action send_goal --feedback /grasp_box_tf \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'grasp-001', target_label: 0, box_layer: 1, box_type: 'smallbox', dry_run: false}"
```

`/execute_drag_box_grasp_tf`：默认右臂观测、右臂拖箱，随后左臂加入抱箱。

```bash
ros2 action send_goal --feedback /execute_drag_box_grasp_tf \
  mission_interfaces/action/ExecuteDragBoxGrasp \
  "{request_id: 'drag-001', target_label: 0, box_layer: 1, box_type: 'bigbox', dry_run: false}"
```

`/navigate_to_point`：导航到指定 `map` 坐标。

```bash
ros2 action send_goal --feedback /navigate_to_point \
  mission_interfaces/action/NavigateToPoint \
  "{request_id: 'nav-001', start: true, point_id: 4, use_custom_pos: true, pos: [-0.37, 0.18, 0.07], dry_run: false}"
```

`/execute_micro_navigation`：底盘相对微调；按实测，`pos[0]` 对应 `base_footprint -Y`，`pos[1]` 对应 `base_footprint +X`。

```bash
ros2 action send_goal --feedback /execute_micro_navigation \
  mission_interfaces/action/ExecuteMicroNavigation \
  "{request_id: 'micro-001', pos: [-0.30, 0.0, 0.0], dry_run: false}"
```

`/execute_fixed_box_workflow`：完整抓取、导航、放置流程。`use_global_observation: false` 按固定 16 箱顺序；改为 `true` 则由前后侧全局感知决定箱数和顺序。

```bash
ros2 action send_goal --feedback /execute_fixed_box_workflow \
  mission_interfaces/action/ExecuteFixedBoxWorkflow \
  "{request_id: 'fixed-001', start: true, start_item_index: 1, stop_after_item_index: 0, dry_run: false, use_global_observation: false}"
```

`/execute_force_carry_box_workflow`：同样的全流程编排，但使用 force-carry 抓取、放置分支。**DragBox TF 在右臂拖出、左臂加入后，默认以左臂 −50 N、右臂 +50 N 的目标力直接夹紧箱子**；达到目标后关闭力控，保持夹持间距随腰部回零，并非持续施加 50 N。GraspBox TF 的 force-carry 目标力为左臂 −10 N、右臂 +10 N。

```bash
ros2 action send_goal --feedback /execute_force_carry_box_workflow \
  mission_interfaces/action/ExecuteFixedBoxWorkflow \
  "{request_id: 'force-carry-001', start: true, start_item_index: 1, stop_after_item_index: 0, dry_run: false, use_global_observation: false}"
```

## 当前常规流程

全流程从 `start_item_index: 1` 启动。启用全局观测时，先观测并执行前侧任务，
再观测并执行后侧任务；每箱执行取箱导航、抓取、后退、放置导航和放置。
`start_item_index` 只选择任务范围，不表示底盘已经到点，也不会跳过前侧全局观测。
临时断点恢复参数和跳过导航逻辑已撤回。

Grasp 检测后的准备姿态使用双臂完整 J1–J7 MoveJ；Drag 保留各自的准备与接入顺序。
两种 TF 抓取先将料箱高度轴 X 对齐机器人竖直方向，保留中心和水平朝向；
腰部搜索和抓取目标使用同一修正后的料箱位姿。夹具标定外参继续生效。
前后偏移直接合入目标 pose，不增加接入后的独立前移 MoveL。

Drag 腰部选姿同时考虑右臂目标与预测左臂接入的关节余量，并对左臂搜索解做 FK 回算。
这只验证规划中的目标，不保证后续实际接触、抬升和整段路径都可达。
常规 Tool-Y 夹紧速度为 5 mm/s；抱箱抬升、腰部回正 MoveL 和连续放置下降使用左 10%、右 15%。
完整参数以 `config/mission/*.yaml` 为准。

## 临时切换观测臂与观测关节角

以下 `ros2 param set` 只对当前运行的 Mission 进程生效，重启后恢复 YAML 默认值。`left`、`right` 按需二选一：

```bash
# GraspBox TF：大小箱共用这个观测臂选择参数。
ros2 param set /mission_controller grasp_box_tf_detection_arm right

# DragBox TF：大箱与小箱的观测臂分别设置。
ros2 param set /mission_controller drag_box_tf_detection_arm right
ros2 param set /mission_controller drag_box_tf_detection_arm_smallbox right
```

观测角度是 **7 个关节的控制器整数单位，1000 单位 = 1°**。下面以现有的第 1 层参数为例；要设置其他箱型、层数或手臂，把参数名中的 `bigbox/smallbox`、`layer1`、`left/right` 改掉，并填入对应的 7 个关节值：

```bash
# GraspBox TF / smallbox / Layer 1 / 右臂
ros2 param set /mission_controller \
  grasp_box_tf_box_layer_pre_detection_right_movej_joint_units_smallbox_layer1 \
  '[-43681,64069,-32832,-82829,52815,-5888,-85689]'

# GraspBox TF / smallbox / Layer 1 / 左臂
ros2 param set /mission_controller \
  grasp_box_tf_box_layer_pre_detection_left_movej_joint_units_smallbox_layer1 \
  '[172102,-3751,14348,95105,7730,-5615,31841]'

# DragBox TF / bigbox / Layer 1 / 左臂
ros2 param set /mission_controller \
  drag_box_tf_box_layer_pre_detection_left_movej_joint_units_bigbox_layer1 \
  '[-72452,133071,70555,-100099,-114234,-49202,-25357]'

# DragBox TF / bigbox / Layer 1 / 右臂
ros2 param set /mission_controller \
  drag_box_tf_box_layer_pre_detection_right_movej_joint_units_bigbox_layer1 \
  '[144725,-5335,7032,9843,7540,-5611,85414]'
```

**DragBox TF 的料箱 X 对齐和自动导航微调只在上述全流程 Action 中执行。单独调用 `/execute_drag_box_grasp_tf` 不会自动导航微调。**

## 腰部回零

下面的 MoveJ 会真实驱动腰部四关节回到 `[0,0,0,0]`：

```bash
ros2 service call /robot/command \
  rm_robot_interfaces/srv/StringCmd \
  "{data: '{\"device\":2,\"payload\":{\"command\":\"movej\",\"joint\":[0,0,0,0],\"v\":15,\"r\":0,\"trajectory_connect\":0}}'}"
```

以上 `dry_run: false` 命令会执行真实运动；逐条运行，并等待上一个 Action 返回后再发送下一个。
