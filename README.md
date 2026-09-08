# G1-D Mission 启动与调用

## 编译

```bash
cd /home/dekc/april/changan/mission_ws
source src/setup_all.zsh

colcon build \
  --base-paths src/mission_manager_interfaces src/mission_manager \
  --merge-install \
  --symlink-install

source install/setup.zsh
```


## 启动 Mission

实物：

```bash
cd /home/dekc/april/changan/mission_ws
source src/setup_all.zsh
source install/setup.zsh

ros2 launch mission_manager mission_system.launch.py \
  simulation:=false \
  hardware:=true \
  robot_ip:=enP8p1s0 \
  enable_rviz:=false \
  taskflow_config_file:=/absolute/path/to/taskflow.yaml
```

仿真：

```bash
ros2 launch mission_manager mission_system.launch.py \
  simulation:=true \
  hardware:=false \
  taskflow_config_file:=/absolute/path/to/taskflow.yaml
```

Dual Arm 已经启动时，只启动 Mission：

```bash
ros2 launch mission_manager mission.launch.py \
  taskflow_config_file:=/absolute/path/to/taskflow.yaml
```

检查接口：

```bash
ros2 action list -t | grep -E \
  'execute_grasp|run_grip|run_peel|execute_grip|execute_peel|execute_assembly|execute_workflow'
```

## 调用 `/execute_grasp`

```bash
ros2 action send_goal --feedback \
  /execute_grasp \
  mission_manager_interfaces/action/ExecuteGrasp \
  "{request_id: 'grasp_test',
    target_label: 0,
    arm: 'left',
    publish_pose: true,
    detection_timeout_sec: 120.0,
    dry_run: false}"
```

## 调用 `/run_grip`

```bash
ros2 action send_goal --feedback \
  /run_grip \
  mission_manager_interfaces/action/ExecuteGrip \
  "{request_id: 'run_grip_test', target_type: 'badge'}"
```

## 调用 `/run_peel`

```bash
ros2 action send_goal --feedback \
  /run_peel \
  mission_manager_interfaces/action/ExecutePeel \
  "{request_id: 'run_peel_test'}"
```

## 调用 `/execute_grip`

```bash
ros2 action send_goal --feedback \
  /execute_grip \
  mission_manager_interfaces/action/ExecuteGrip \
  "{request_id: 'execute_grip_test', target_type: 'badge'}"
```

## 调用 `/execute_peel`

```bash
ros2 action send_goal --feedback \
  /execute_peel \
  mission_manager_interfaces/action/ExecutePeel \
  "{request_id: 'execute_peel_test'}"
```

## 调用 `/execute_assembly`

```bash
ros2 action send_goal --feedback \
  /execute_assembly \
  mission_manager_interfaces/action/ExecuteAssembly \
  "{request_id: 'execute_assembly_test', target_type: 'connector'}"
```

## 调用 `/execute_workflow`

导航1 → 连接件抓取 → 导航3 → 安装 → 导航2 → 车标抓取、撕膜 → 导航3 → 安装。

```bash
ros2 action send_goal --feedback \
  /execute_workflow \
  mission_manager_interfaces/action/ExecuteWorkflow \
  "{start: true}"
```

## 调用 `/navigate_to_point`

```bash
ros2 action send_goal --feedback /navigate_to_point \
  mission_manager_interfaces/action/NavigateToPoint \
  "{request_id: nav-001, start: true, point_id: 1, use_custom_pos: false, dry_run: false}"

ros2 action send_goal --feedback /navigate_to_point \
  mission_manager_interfaces/action/NavigateToPoint \
  "{request_id: nav-002, start: true, point_id: 1, use_custom_pos: true, pos: [2.14, -2.84, -2.89], dry_run: false}"
```

## 调用 `/execute_navigation`

纯导航：1 → 3 → 2 → 3，每次等待平台到点成功，不调用感知或机械臂 Action。

```bash
ros2 action send_goal --feedback /execute_navigation \
  mission_manager_interfaces/action/ExecuteNavigation "{start: true}"

mosquitto_pub -h 127.0.0.1 -t mission/workflow/start \
  -m '{"robot_id":"g1d","start":true,"workflow":"navigation","request_id":"test-001"}'
```
