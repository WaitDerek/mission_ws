# MQTT Workflow 与导航交互说明

本文档说明平台、Mission 和导航系统之间的 MQTT 交互协议。

## 1. 参与方

```text
平台/上位机
    ↕ MQTT
Mission execute_workflow
    ↕ ROS 2 Action
视觉、抓取、放置模块
```

Mission 当前使用 MQTT Broker：

```yaml
mqtt_host: 127.0.0.1
mqtt_port: 1883
mqtt_robot_id: "realman-001"
mqtt_navigation_robot_id: "realman-001"
mqtt_qos: 1
```

## 2. 平台启动 Workflow

平台向 Mission 发布：

```text
Topic: mission/workflow/start
```

消息格式：

```json
{
  "robot_id": "realman-001",
  "start": true,
  "request_id": "platform-001"
}
```

字段：

| 字段 | 含义 |
|---|---|
| `robot_id` | 机器人编号，当前为 `realman-001` |
| `start` | 必须为 `true` |
| `request_id` | 平台请求编号，可选；为空时 Mission 自动生成 |
| `workflow` | 可选；`full`（默认）或 `observation_navigation` |
| `observation_point_id` | 测试流的观察点，范围 1～4；测试流必填 |

Mission 收到后会校验机器人编号、启动标志以及是否已有任务运行，然后在内部调用：

```yaml
/execute_workflow
start: true
```

只运行“导航 → 全局观测 → 导航”的测试流：

```json
{
  "robot_id": "realman-001",
  "start": true,
  "request_id": "vision-nav-001",
  "workflow": "observation_navigation",
  "observation_point_id": 1
}
```

Mission 会调用 `/execute_observation_navigation`。第二次导航点由 Vision Result 的
`order_stack_indices[0]` 与观察点共同映射。只有导航回执 `success=true`，以及
GlobalObservation 的 ROS 状态为 `SUCCEEDED`、`success=true`、`plan_valid=true` 且
计划数组通过校验时才会继续。该流程不会调用抓取或放置 Action。

## 3. Workflow 状态返回

Mission 向平台发布：

```text
Topic: mission/workflow/status
```

收到请求：

```json
{
  "event": "received",
  "robot_id": "realman-001",
  "request_id": "platform-001",
  "message": "MQTT start request queued"
}
```

Action 被接受：

```json
{
  "event": "accepted",
  "robot_id": "realman-001",
  "request_id": "platform-001",
  "message": "workflow Action goal accepted"
}
```

执行反馈：

```json
{
  "event": "feedback",
  "robot_id": "realman-001",
  "request_id": "platform-001",
  "workflow_id": "wf-xxxx",
  "stage": "NAVIGATE_PICKUP_6",
  "point_id": "6",
  "stack_id": "stack_1",
  "current_order_index": 1,
  "total_order_items": 2,
  "detail": "requesting pickup navigation"
}
```

最终结果：

```json
{
  "event": "result",
  "robot_id": "realman-001",
  "request_id": "platform-001",
  "workflow_id": "wf-xxxx",
  "success": true,
  "message": "depalletizing workflow completed",
  "completed_observation_count": 2,
  "completed_box_count": 4,
  "final_stage": "COMPLETE",
  "ros_status": 4
}
```

常见 `event`：

```text
received  收到启动请求
accepted  Workflow Action 已接受
feedback  正在执行
result    完成或失败
rejected  请求被拒绝
```

## 4. Mission 发送导航目标

Workflow 到达观察点、抓取点或放置点前，Mission 向平台发布：

```text
Topic: mission/navigation/request
```

消息格式：

```json
{
  "robot_id": "realman-001",
  "point_id": 6,
  "frame_id": "map",
  "pos": [1.2, 3.4, 1.57]
}
```

字段：

| 字段 | 含义 |
|---|---|
| `robot_id` | 机器人编号 |
| `point_id` | 导航点编号，范围 1～16，必须存在 |
| `frame_id` | 地图坐标系，当前默认 `map` |
| `pos[0]` | 目标 X 坐标 |
| `pos[1]` | 目标 Y 坐标 |
| `pos[2]` | 目标朝向 Yaw，单位为弧度 |

`x`、`y`、`yaw` 来自参数：

```yaml
mqtt_navigation_points_json
```

格式示例：

```yaml
mqtt_navigation_points_json: >-
  {
    "1": {"x": 1.0, "y": 2.0, "yaw": 0.0},
    "5": {"x": 3.0, "y": 1.0, "yaw": 1.57},
    "6": {"x": 4.0, "y": 1.0, "yaw": 1.57},
    "16": {"x": 0.0, "y": 0.0, "yaw": 3.14}
  }
```

如果当前仍是：

```yaml
mqtt_navigation_points_json: "{}"
```

则没有有效导航点，Mission 会在发送 MQTT 前直接返回失败。

## 5. 平台返回导航结果

平台向 Mission 发布：

```text
Topic: mission/navigation/result
```

成功：

```json
{
  "robot_id": "realman-001",
  "success": true,
  "message": "arrived"
}
```

失败：

```json
{
  "robot_id": "realman-001",
  "success": false,
  "message": "navigation failed"
}
```

注意：导航结果中不要添加 `id` 或 `point_id`。当前 Mission 通过 `robot_id` 和正在等待的单个导航请求进行匹配。

## 6. Workflow 导航顺序

```text
启动 Workflow
→ 获取任务 Lease
→ 导航到观察点
→ 调用 /depalletizing/observe
→ 导航到抓取点
→ 调用 /grasp_box_tf 或 /execute_drag_box_grasp_tf
→ 导航到 16 号放置点
→ 调用 /execute_box_place
→ 处理下一个料箱
→ 释放 Lease
```

抓取点映射：

| 观察点 | 直接抓取 | 抽拉抓取 |
|---|---:|---:|
| 1 | 6 | 5 |
| 2 | 8 | 7 |
| 3 | 10 | 9 |
| 4 | 12 | 11 |

偶数抓取点使用 `/grasp_box_tf`，奇数抓取点使用 `/execute_drag_box_grasp_tf`。

## 7. 超时和失败规则

```yaml
mqtt_connect_timeout_sec: 10.0
mqtt_navigation_timeout_sec: 300.0
```

以下情况会导致导航失败：

- MQTT Broker 无法连接；
- 导航点没有配置；
- 平台没有返回结果；
- 返回的 `robot_id` 不是 `realman-001`；
- `success` 不是布尔值；
- 平台返回 `success: false`。

当前 Workflow 是串行的，同一时间只允许一个导航请求；导航失败后 Workflow 终止并释放 Lease，不会自动重试整个任务。

## 8. 单点导航 Action

Mission 同时提供只执行一次导航的 ROS 2 Action：

```text
/navigate_to_point
```

调用示例：

```bash
ros2 action send_goal --feedback \
  /navigate_to_point \
  mission_interfaces/action/NavigateToPoint \
  "{request_id: 'nav-001', point_id: 6, dry_run: false}"
```

执行流程：

```text
读取 point_id 对应坐标
→ 发布 mission/navigation/request
→ 等待 mission/navigation/result
→ 返回 Action 结果
```

`dry_run: true` 只读取并返回目标坐标，不连接 MQTT，也不会移动平台。

## 9. 调试命令

查看所有任务相关 MQTT 消息：

```bash
mosquitto_sub \
  -h 127.0.0.1 \
  -p 1883 \
  -t 'mission/#' \
  -v
```

通过 MQTT 启动任务：

```bash
mosquitto_pub \
  -h 127.0.0.1 \
  -p 1883 \
  -t mission/workflow/start \
  -q 1 \
  -m '{"robot_id":"realman-001","start":true,"request_id":"platform-001"}'
```

也可以绕过 MQTT 启动入口，直接调用 ROS Action：

```bash
ros2 action send_goal --feedback \
  /execute_workflow \
  mission_interfaces/action/ExecuteWorkflow \
  "{start: true}"
```

此时只绕过 MQTT 启动通道；如果 `navigation_adapter: mqtt`，导航过程仍然使用 MQTT。
