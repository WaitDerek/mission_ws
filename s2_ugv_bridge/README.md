# S2 UGV ROS2 主机命令

这个包让机器人 ROS2 主机通过一条 ROS2 命令，调用 `unitree` Docker 容器内已经验证过的 ROS1 `timed_translate` Action。

它是第一阶段的手动底盘入口，不修改也不依赖 `mission_controller`。后续 ROS2 Mission 需要底盘能力时，应复用这个网关逻辑，而不是重新直接发布 ROS1 底盘 Topic。

## 调用链

```text
ros2 run s2_ugv_bridge timed_translate ...
  -> docker exec unitree
  -> ROS1 /timed_translate Action
  -> /ugv/mode_cmd + /ugv/motion_cmd
  -> S2 底盘驱动
```

## 部署与构建

在 Windows 本机 PowerShell（不是 SSH 进去后的 zsh）执行同步：

```powershell
$OutputEncoding = [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)
scp -r "E:/gid/s2_ugv_bridge" "s2:/rm_nvme/recordings/code/mission_ws/src/"
```

在机器人 ROS2 主机的 zsh 终端执行构建：

```zsh
source /opt/ros/humble/setup.zsh
cd /rm_nvme/recordings/code/mission_ws
colcon build --packages-select s2_ugv_bridge
source install/setup.zsh
ros2 pkg executables s2_ugv_bridge
```

最后一条应包含：

```text
s2_ugv_bridge timed_translate
s2_ugv_bridge move_base_distance
```

## 前置条件

- Docker 容器名为 `unitree`，且机器人 ROS2 主机上的当前用户能运行 `docker exec`。
- Docker 内的 ROS1 `timed_translate` server 已启动。
- 真实运动前，底盘驱动正常、周围无人且运动方向前方留有安全距离。

命令会先在 Docker 内最多等待 10 秒确认 `/timed_translate/goal` 存在；没有服务端时返回 `70`，不会发送运动请求。

## 使用

先在每个新的 ROS2 主机 zsh 终端执行：

```zsh
source /opt/ros/humble/setup.zsh
source /rm_nvme/recordings/code/mission_ws/install/setup.zsh
```

随后调用：

```zsh
ros2 run s2_ugv_bridge timed_translate <direction> <duration_s> [--speed-mps <m/s>]
```

例如，向前运动 5 秒、速度 0.08 m/s：

```zsh
ros2 run s2_ugv_bridge timed_translate forward 5 --speed-mps 0.08
```

可用方向：`forward`、`backward`、`left`、`right`。

若省略 `--speed-mps`，网关会向现有 ROS1 Action 传 `0`，保持其既有的默认速度逻辑；它不是新的速度默认值。

## 按实际距离移动（ROS2 Action）

`move_base_distance` 是一个 ROS2 Action Server。它把目标距离换算成
ROS1 `timed_translate` 所需的持续时间，再调用同一个 Docker 网关。默认速度为
`0.05 m/s`，并使用当前实测的距离比例：

```text
forward/backward: 实际距离 / 理论距离 = 0.17 / 0.30
left/right:       实际距离 / 理论距离 = 0.15 / 0.30
```

启动 action server（机器人终端）：

```zsh
source /opt/ros/humble/setup.zsh
source /rm_nvme/recordings/code/mission_ws/install/setup.zsh
ros2 run s2_ugv_bridge move_base_distance
```

发送目标距离（单位米）：

```zsh
ros2 action send_goal --feedback \
  /move_base_distance \
  mission_interfaces/action/MoveBaseDistance \
  "{direction: forward, distance_m: 0.30}"
```

其它方向把 `direction` 换成 `backward`、`left` 或 `right`。上例会按
`0.05 m/s` 计算约 `10.588 s` 的前进持续时间，以补偿实测只有理论距离
约 `56.67%` 的情况。横移 `0.30 m` 会计算为 `12.0 s`。

如果确实要临时改变速度，可在 goal 中指定：

```zsh
ros2 action send_goal --feedback \
  /move_base_distance \
  mission_interfaces/action/MoveBaseDistance \
  "{direction: right, distance_m: 0.15, speed_mps: 0.05}"
```

校准比例和默认值是 action server 参数，可在启动后查询或调整：

```zsh
ros2 param get /move_base_distance_action_server default_speed_mps
ros2 param get /move_base_distance_action_server forward_actual_per_commanded_ratio
ros2 param get /move_base_distance_action_server lateral_actual_per_commanded_ratio
ros2 param set /move_base_distance_action_server default_speed_mps 0.05
```

单次只允许一个底盘移动目标。取消 action 或按 Ctrl+C 会先取消 ROS1
`timed_translate`，再结束本地进程，避免底盘继续运动。

## 停止与错误码

- 执行中按 `Ctrl+C`：网关先向 `/timed_translate/cancel` 发取消请求，然后结束本机等待进程。现有 ROS1 server 的 `finally` 分支会发布零速度停止命令。退出码为 `130`。
- 找不到 ROS1 Action Server 或 Docker 前置检查失败：退出码为 `70`，且不会发送运动目标。
- 无法启动 Docker 客户端：退出码为 `71`。
- 动作未在“请求时长 + 15 秒余量”内结束：网关会取消动作，退出码为 `124`。

可用 `--server-timeout-s` 和 `--completion-margin-s` 按需调整等待时间；这两个数必须大于零。

## 分层验证顺序

1. ROS1 server 未运行时，执行带 `--server-timeout-s 2` 的命令，应失败且机器人不动。
2. ROS1 server 以 `_dry_run:=true` 运行时，执行一秒命令，应在主机终端看到反馈与结果，但机器人不动。
3. 仅在场地安全并由操作者确认后，执行真实短时运动；另做一次 `Ctrl+C` 取消验证，并通过 `/ugv/motion_state` 确认最终线速度为零。
