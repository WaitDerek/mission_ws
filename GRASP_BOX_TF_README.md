# `/grasp_box_tf` 使用说明

本文档说明 `mission_controller` 中基于实时 TF 的 GraspBox 流程。该流程使用
FoundationPose 检测料箱，再把检测结果冻结到机器人底盘固定坐标系，最后在腰部
运动后使用实时 TF 将左右目标重新表达为各自机械臂基座下的 `Link7` 目标。

## 1. Action 接口

`/grasp_box_tf` 使用的接口类型是：

```text
mission_interfaces/action/ExecuteBoxGrasp
```

Goal 字段如下：

```yaml
request_id: string       # 本次任务的唯一标识
target_label: int32      # FoundationPose 实例序号；通常为 0
box_layer: int32         # 料箱层号，范围 1～4
box_type: string         # bigbox 或 smallbox
dry_run: bool            # true 只计算/验证，不发送实体运动
```

`box_type` 优先于 ROS 参数选择本次检测模型，可填写：

```text
bigbox、big、big_box
smallbox、small、small_box
```

建议每次都显式填写 `box_type` 和 `box_layer`，避免使用旧客户端的默认值。

## 2. 环境和启动

在 `rm1` 上执行：

```zsh
cd /rm_nvme/recordings/code/mission_ws/src
source ./setup_all.zsh
```

如果工作区源码或接口刚刚更新，先重新构建并重新加载环境：

```zsh
cd /rm_nvme/recordings/code/mission_ws
colcon build --packages-select mission_interfaces mission_controller \
  --symlink-install --merge-install
source ./install/setup.zsh
```

使用 rm1 配置启动 Mission，并启用全局 TF：

```zsh
ros2 launch mission_controller mission.launch.py \
  config_file:=/rm_nvme/recordings/code/mission_ws/install/mission_controller/share/mission_controller/config/mission_rm1.yaml \
  direct_motion_backend:=python_sdk \
  require_command_subscribers:=true \
  enable_global_tf:=true \
  enable_robot_state_publisher:=true
```

如果使用的是已经启动好的 bringup 和感知系统，只需要启动上面的 Mission 节点；
FoundationPose 服务必须提供对应的 `/object_pose/estimate_right` Action Server。

检查 Action 和关键节点：

```zsh
ros2 action list | grep -E 'grasp_box_tf|object_pose/estimate_right'
ros2 node list | grep -E 'mission_controller|realbots_global_tf|robot_state_publisher'
```

## 3. 推荐运行顺序

### 3.1 先做 dry-run

以大料箱第一层为例：

```zsh
ros2 action send_goal --feedback \
  /grasp_box_tf \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'grasp-box-tf-bigbox-layer1-dry-run', target_label: 0, box_layer: 1, box_type: 'bigbox', dry_run: true}"
```

以小料箱第四层为例：

```zsh
ros2 action send_goal --feedback \
  /grasp_box_tf \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'grasp-box-tf-smallbox-layer4-dry-run', target_label: 0, box_layer: 4, box_type: 'smallbox', dry_run: true}"
```

dry-run 仍会调用 FoundationPose、检查检测结果时间戳、查询 TF 并计算目标，
但不会向机械臂发送实际 MoveJ/MoveL/MoveJ_P。应先确认反馈中的
`DIRECT_MOVEL_TARGETS` 坐标和姿态，再进行实体运行。

### 3.2 实体运行

确认机器人处于安全状态、机械臂和料箱没有被手动移动后，将 `dry_run` 改为
`false`：

```zsh
ros2 action send_goal --feedback \
  /grasp_box_tf \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'grasp-box-tf-bigbox-layer1-real', target_label: 0, box_layer: 1, box_type: 'bigbox', dry_run: false}"
```

如果 `box_direct_movel_enabled=true`，目标会交给配置的 Python SDK 后端；当前
rm1 配置的 `direct_movel_motion_mode` 为 `movej_p`。Action 成功只代表 SDK 调用和
任务流程成功返回，仍应在动作结束后读取左右 `/mcap/slave_arm_*` 核对实际 EEPose。

## 4. TF 计算流程

### 4.1 检测阶段

FoundationPose 返回：

```text
PoseStamped.header.frame_id = 相机 optical frame
PoseStamped.header.stamp    = 检测时刻
```

`grasp_box_tf` 使用这个检测时间戳查询：

```text
camera optical frame -> grasp_box_tf_freeze_frame
```

默认冻结坐标系为：

```text
base_link
```

冻结后的料箱 Pose 表示料箱在底盘固定坐标系中的位置。这样后续腰部运动不会
改变料箱的物理目标。

检测结果必须有非零时间戳，因为参数
`grasp_box_tf_require_detection_timestamp=true` 会拒绝零时间戳结果。

### 4.2 目标构造阶段

目标的概念计算顺序为：

```text
冻结的 box Pose
  -> 在料箱坐标系叠加每层/每种料箱的 XYZ offset
  -> 在料箱坐标系叠加完整 Pose correction
  -> 使用 box-to-Link7 标定姿态生成 Link7 orientation
  -> 应用可选的夹具中心补偿
  -> 得到左右 Link7 的底盘固定坐标系目标
```

其中：

- `direct_movel_*_offset_xyz_<model>_layer<N>` 是料箱坐标系下的三维位置偏移；
- `joint123_layer<N>_*_target_correction_pose_box` 是料箱坐标系下的完整
  `[x, y, z, qx, qy, qz, qw]` 修正；
- `direct_movel_left/right_box_to_link8_orientation` 是料箱到当前末端
  `Link7` 的相对姿态。参数名保留 `link8` 是历史兼容命名，当前实际末端是
  `Link7`；
- `left_fixture_center_in_link8_xyz` 和
  `right_fixture_center_in_link8_xyz` 是末端坐标系下的夹具中心偏移，仅在
  `direct_movel_fixture_compensation_enabled=true` 时生效。

### 4.3 腰部运动后的实时 TF 重表达

腰部运动完成后，程序使用实时 TF 将冻结的底盘目标转换到左右机械臂基座：

```text
base_link -> L_base_Link
base_link -> R_base_Link
```

实际查询方向由 TF API 自动处理，不能把 `L_base_Link`、`R_base_Link` 的数值
直接当成末端 Pose。转换完成后才调用 SDK 的目标接口。

全局 TF 由 `realbots_global_tf` 根据 URDF、`/mcap/body` 腰部反馈和左右臂反馈
发布。相机安装链及 optical frame 的静态关系来自：

```text
mission_controller/config/global_tf.yaml
```

可用以下命令检查链路：

```zsh
ros2 run tf2_ros tf2_echo base_link R_base_Link
ros2 run tf2_ros tf2_echo base_link L_base_Link
ros2 run tf2_ros tf2_echo right_arm_8_Link right_arm_depth_cam_color_optical_frame
ros2 run tf2_ros tf2_echo left_arm_8_Link left_arm_depth_cam_color_optical_frame
```

实际相机 frame 名称以 FoundationPose 返回的 `header.frame_id` 为准；若名称
不同，应先在 `/tf`、`/tf_static` 中确认实际 frame，而不是猜测名称。

## 5. 关键参数

以下参数是 TF GraspBox 最常用的控制项，可用 `ros2 param get` 读取：

```zsh
ros2 param get /mission_controller grasp_box_tf_action_name
ros2 param get /mission_controller grasp_box_tf_freeze_frame
ros2 param get /mission_controller grasp_box_tf_detection_tf_timeout_sec
ros2 param get /mission_controller grasp_box_tf_runtime_tf_timeout_sec
ros2 param get /mission_controller grasp_box_tf_require_detection_timestamp

ros2 param get /mission_controller box_object_pose_action_name
ros2 param get /mission_controller box_direct_movel_enabled
ros2 param get /mission_controller direct_motion_backend
ros2 param get /mission_controller direct_movel_motion_mode
ros2 param get /mission_controller direct_movel_target_mode
ros2 param get /mission_controller box_grasp_execution_mode
ros2 param get /mission_controller direct_movel_fixture_compensation_enabled
```

腰部模式支持：

```text
arms_only
joint1_then_arms
joint1_then_arms_keep_position
joint123_then_arms
```

TF 模式要求：

```text
direct_movel_target_mode = camera_offset_box_orientation
```

每层料箱的检测关节、腰部角度、目标修正和料箱偏移均独立配置。相关参数可按
模型和层号读取，例如：

```zsh
ros2 param get /mission_controller \
  box_layer_pre_detection_right_movej_joint_units_smallbox

ros2 param get /mission_controller \
  direct_movel_right_offset_xyz_smallbox_layer4

ros2 param get /mission_controller \
  joint123_layer4_right_target_correction_pose_box
```

## 6. 排查顺序

1. **Action 不存在或 Goal 被拒绝**

   确认 `mission_controller` 已启动、接口包已重新构建并且当前 shell 已 source
   最新的 `install/setup.zsh`：

   ```zsh
   ros2 action list | grep grasp_box_tf
   ros2 action info /grasp_box_tf
   ```

2. **FoundationPose 超时**

   确认感知 Action Server 存在且名称与参数一致：

   ```zsh
   ros2 action info /object_pose/estimate_right
   ros2 param get /mission_controller box_object_pose_action_name
   ```

3. **时间戳错误**

   检查 FoundationPose 返回的 `PoseStamped.header.stamp` 是否为零，并确认检测
   与 TF 节点使用的是同一 ROS 时间域。

4. **TF lookup 失败**

   依次检查 `base_link`、`R_base_Link`、`L_base_Link` 和相机 optical frame：

   ```zsh
   ros2 topic echo /tf --once
   ros2 topic echo /tf_static --once
   ros2 run tf2_tools view_frames
   ```

   同时确认 `realbots_global_tf` 已启动，且全局 TF 的 QoS/RMW 配置与机器人
   网络一致。

5. **目标数值与理想 Pose 不一致**

   先比较 `DIRECT_MOVEL_TARGETS` 与实际 `/mcap/slave_arm_left`、
   `/mcap/slave_arm_right` 的 EEPose：

   - 目标与实际差异大：检查 SDK MoveJ_P、机械臂可达性和夹具补偿；
   - 实际接近目标但不接近理想值：检查料箱坐标系 offset、完整 correction 和
     box-to-Link7 orientation；
   - 腰部运动后才出现误差：检查实时 arm-base TF、腰部反馈和冻结 frame；
   - 只有一侧异常：检查该侧 base frame、末端姿态或夹具参数。

## 7. 安全注意事项

- 首次测试必须使用 `dry_run=true`，再使用低速和空旷区域进行实体运行。
- `dry_run=false` 前确认左右臂、腰部和料箱周围没有人员或障碍物。
- TF 目标依赖底盘固定；底盘或料箱在检测后移动会使冻结目标失效。
- SDK 返回成功不等于末端已经达到理想误差范围，动作结束后应再次读取实际
  EEPose。
- 不要同时启动两套会发布同名 arm-base 或 camera optical frame 的 TF 发布器，
  否则可能出现 TF 抖动或跳变。
