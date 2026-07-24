### 仿真流程
ros2 launch mission_controller mission_system.launch.py \
    mode:=simulation \
    pipeline:=box \
    grasp_config_file:=camera_topics_local_d405.yaml

### 逐步启动
1. dual arm
ros2 launch robot_bringup planning_only.launch.py \
    robot_profile:=r1_pro \
    planning_pipeline:=ompl \
    dry_run:=false \
    enable_rviz:=true \
    enable_fake_ros2_control:=true

2. mission
ros2 launch mission_controller mission.launch.py \
    start_perception:=false

3. foundation pose (3D 检测效果可视化)
ros2 launch object_pose_ros object_pose_action.launch.py \
    server_output:=screen

### 到达观测位置
ros2 action send_goal --feedback \
  /move_arm_j \
  task_interfaces/action/MoveArmJoints \
  "{left_joints: [-0.88, 1.24, -0.70, -2.0, 1.25, 0.1, 0.0],
    right_joints: [0.86, -0.24, 0.20, -2.0944, 0.174647, -0.618606, 0.104098],
    dry_run: false,
    duration: 5.0}"

### 检测
ros2 action send_goal --feedback \
    /object_pose/estimate \
    object_pose_interfaces/action/EstimateObjectPose \
    "{model_label: 'f320', instance_index: 0, confidence_threshold: 0.0}"

ros2 action send_goal \
  /object_pose/estimate \
  object_pose_interfaces/action/EstimateObjectPose \
  "{model_label: 'f455',
    instance_index: 0,
    confidence_threshold: 0.0}" \
  --feedback

### mission全流程
ros2 action send_goal --feedback \
  /execute_box_grasp \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{request_id: 'f320_camera_sim_test',
    target_label: -1,
    dry_run: false}"

ros2 action send_goal /execute_box_place \
  mission_interfaces/action/ExecuteBoxPlace \
  "{request_id: box_place_test, dry_run: false}" \
  --feedback

### 实物流程
ros2 launch mission_controller mission_system.launch.py \
  mode:=hardware \
  pipeline:=box \
  hardware_armed:=true \
  dry_run:=false \
  enable_rviz:=false \
  enable_robot_state_publisher:=false

dry_run默认false，true模式不进行运动
hardware_armed默认false

ros2 action send_goal /move_chassis \
  mission_interfaces/action/MoveChassis \
  "{direction: left}" \
  --feedback

ros2 param set /mission_controller chassis_linear_speed 0.5
ros2 param set /mission_controller chassis_move_duration_sec 3.0

ros2 param get /mission_controller chassis_linear_speed
ros2 param get /mission_controller chassis_move_duration_sec

更改对象
ros2 param set /mission_controller box_object_pose_model_label f455
ros2 param set /mission_controller box_width 0.35
ros2 param set /mission_controller box_height 0.1703
ros2 param set /mission_controller box_type f455

ros2 param set /mission_controller box_object_pose_model_label f320
ros2 param set /mission_controller box_width 0.357
ros2 param set /mission_controller box_height 0.127
ros2 param set /mission_controller box_type f320

ros2 param get /mission_controller box_object_pose_model_label
ros2 param get /mission_controller box_width
ros2 param get /mission_controller box_height
ros2 param get /mission_controller box_type


ros2 action send_goal /execute_box_stack \
  mission_interfaces/action/ExecuteBoxStack \
  "{level: 1}" \
  --feedback
