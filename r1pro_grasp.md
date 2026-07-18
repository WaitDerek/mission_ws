### 仿真流程
ros2 launch mission_controller mission_system.launch.py \
    mode:=simulation \
    pipeline:=grasp \
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

3. grasp detection (3D 检测效果可视化)
ros2 launch grasp_orchestrator grasp_detection.launch.py \
    config_file:="$(ros2 pkg prefix grasp_orchestrator)/share/grasp_orchestrator/config/camera_topics_local_d405.yaml" \
    visualize:=true \
    visualization_grasps:=10

### 检测
ros2 service call \
    /detect_grasp_pose \
    grasp_orchestrator_interfaces/srv/DetectGraspPose \
    "{target_frame: '',
      target_label: 0,
      timeout_sec: 30.0}"

### 到达观测位置
ros2 action send_goal --feedback  /move_arm_j   task_interfaces/action/MoveArmJoints \ "{left_joints: [-0.88, 0.84, -1.13, -1.80, 1.25, 0.29, 0.13],    right_joints: [-0.98, -0.84, 1.13, -2.00, -1.25, 0.60, -0.13], dry_run: false,  duration: 5.0}"

### mission全流程
ros2 action send_goal --feedback \
    /execute_grasp \
    mission_interfaces/action/ExecuteGrasp \
    "{request_id: 'grasp_sim_test',
      target_frame: 'torso_link4',
      target_label: 0,
      arm: 'right',
      publish_pose: true,
      detection_timeout_sec: 30.0,
      dry_run: false}"


### 实物流程
ros2 launch mission_controller mission_system.launch.py \
    mode:=hardware \
    pipeline:=grasp \
    grasp_config_file:=camera_topics_local_d405.yaml \
    hardware_armed:=false \
    dry_run:=true

确认反馈、订阅者和急停后，执行实物模式：

ros2 launch mission_controller mission_system.launch.py \
    mode:=hardware \
    pipeline:=grasp \
    grasp_config_file:=camera_topics_local_d405.yaml \
    hardware_armed:=true \

dry_run默认false
hardware_armed默认false