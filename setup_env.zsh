# ------------------------------ Vision -------------------------------------
export ROS_DOMAIN_ID=23
export VISION_WS=/home/unitree/code/vision_ws/src/hangcha-perception
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/vision_ws/install/setup.zsh
ros2 launch realsense_d405_bringup dual_d405.launch.py

cd /home/unitree/code/vision_ws/src/hangcha-perception 
conda deactivate
conda activate changan
export VISION_WS=/home/unitree/code/vision_ws/src/hangcha-perception
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/vision_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
./start_object_pose_action.sh --config g1d

source /opt/ros/foxy/setup.zsh
source /home/unitree/code/vision_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
ros2 action send_goal --feedback \
  /object_pose/estimate \
  object_pose_interfaces/action/EstimateObjectPose \
  "{model_label: badge, instance_index: 0, confidence_threshold: 0.0}"

ros2 action send_goal --feedback \
  /object_pose/estimate \
  object_pose_interfaces/action/EstimateObjectPose \
  "{model_label: badge_back, instance_index: 0, confidence_threshold: 0.0}"

ros2 action send_goal --feedback \
  /object_pose/estimate \
  object_pose_interfaces/action/EstimateObjectPose \
  "{model_label: badge_connector, instance_index: 0, confidence_threshold: 0.0}"
# ---------------------------------------------------------------------------


# ---------------------------- Torso and Arm -----------------------------------
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/dual_arm_ws/install/setup.zsh

ros2 launch robot_bringup test.launch.py \
  robot_profile:=g1_d \
  robot_adapter:=g1d \
  robot_ip:=enP8p1s0 \
  dry_run:=false \
  prefer_hardware:=true \
  allow_mock_fallback:=false \
  enable_robot_state_publisher:=true \
  enable_move_group:=true \
  enable_environment_collision:=false \
  enable_rviz:=false \
  ros_domain_id:=23

ros2 topic pub --once /g1_d/torso/command \
  task_interfaces/msg/G1dTorsoCommand \
  "{initialize: true}"

ros2 topic pub --once /g1_d/torso/command \
  task_interfaces/msg/G1dTorsoCommand \
  "{control_mode: 3, target_position: 0.05, speed: 0.4, initialize: false}"
# ---------------------------------------------------------------------------


# ---------------------------- Sensor & Gripper -----------------------------
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/driver_ws/install_foxy/setup.zsh
ros2 run brsd_extra app_run_component \
  --component_config=/home/unitree/code/driver_ws/deploy/kwr57b_driver_config.yaml \
  --component_name=kwr57b \
  --log_path=/home/unitree/code/driver_ws/run_logs/kwr57b

export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/driver_ws/install_foxy/setup.zsh
ros2 run brsd_quad app_run_component \
  --component_config=/home/unitree/code/driver_ws/deploy/usb_relay.yaml \
  --component_name=usb_relay \
  --log_path=/home/unitree/code/driver_ws/run_logs/usb_relay

export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/driver_ws/install_foxy/setup.zsh

ros2 topic echo /force_torque/data brsd_msgs/msg/ForceData

bash /home/unitree/code/driver_ws/deploy/usb_relay_test.sh 1 on 2 on
bash /home/unitree/code/driver_ws/deploy/usb_relay_test.sh 1 off 2 off

bash /home/unitree/code/driver_ws/deploy/usb_relay_test.sh 3 on 4 on
bash /home/unitree/code/driver_ws/deploy/usb_relay_test.sh 3 off 4 off
# ---------------------------------------------------------------------------


# ----------------------------- Navigation ----------------------------------
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/mission_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$(python3 -c 'import paho, pathlib; print(pathlib.Path(paho.__file__).parent.parent)'):${PYTHONPATH}"
ros2 launch mission_manager navigation.launch.py

source /opt/ros/foxy/setup.zsh
source /home/unitree/code/mission_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

ros2 action send_goal --feedback \
/navigate_to_point \
mission_manager_interfaces/action/NavigateToPoint \
"{
  request_id: '0-start', 
  start: true, 
  point_id: 1, 
  use_custom_pos: true, 
  pos: [-1.18, -0.61, -0.61], 
  dry_run: false
}"
# ---------------------------------------------------------------------------


# ------------------------------ Full Pipline -------------------------------
conda deactivate
conda activate changan
export ROS_DOMAIN_ID=23
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/foxy/setup.zsh
source /home/unitree/code/vision_ws/src/hangcha-perception/install/setup.zsh
source /home/unitree/code/dual_arm_ws/install/setup.zsh
source /home/unitree/code/driver_ws/install_foxy/setup.zsh
source /home/unitree/code/mission_ws/install/setup.zsh
PYTHONPATH=/home/unitree/miniconda3/envs/changan/lib/python3.8/site-packages:$PYTHONPATH

ros2 launch mission_manager mission.launch.py \
    config_dir:=/home/unitree/code/mission_ws/src/mission_manager/config

ros2 action send_goal --feedback /execute_grasp \
    mission_manager_interfaces/action/ExecuteGrasp "{}" 

ros2 action send_goal --feedback /execute_peel \
    mission_manager_interfaces/action/ExecutePeel "{}" 

ros2 action send_goal --feedback /execute_assembly \
    mission_manager_interfaces/action/ExecuteAssembly "{}" 
  
ros2 action send_goal --feedback /execute_workflow \
    mission_manager_interfaces/action/ExecuteWorkflow "{}" 

# ------------------------------ Task Dispatch ------------------------------
source /opt/ros/foxy/setup.zsh 
source /home/unitree/code/dual_arm_ws/install/setup.zsh
source /home/unitree/code/driver_ws/install/setup.zsh
source /home/unitree/code/task_ws/install/setup.zsh
ros2 launch execute_grasp_script_runner \
    execute_grasp_script.launch.py

source /home/unitree/code/task_ws/install/setup.zsh
ros2 action send_goal --feedback /execute_grasp \
mission_interfaces/action/ExecuteGrasp "{request_id: 'grasp_sim_test',
      target_frame: 'torso_link4',
      target_label: 0,
      arm: 'right',
      publish_pose: true,
      detection_timeout_sec: 30.0,
      dry_run: false}"
# ---------------------------------------------------------------------------




source /home/unitree/code/dual_arm_ws/install/setup.zsh
ros2 action send_goal /move_arm_j task_interfaces/action/MoveArmJoints \
"{
  left_joints: [
1.022637963294983,
                0.40355679392814636,
                1.0049611330032349,
                0.26080071926116943,
                -0.371343195438385,
                -0.26601386070251465,
                -1.610739827156067
  ], 
  right_joints: [
  -0.1438107043504715,
                -0.26147183775901794,
                0.1994294971227646,
                1.3429043292999268,
                0.13344435393810272,
                -0.20543359220027924,
                0.3900386095046997

  ],
  dry_run: false, 
  duration: 0.0
}"


ros2 action send_goal /move_arm_p task_interfaces/action/MoveArmPose --feedback \
"{
  left_pose: [
    0.16664711132150462,
    0.16314639015713775,
    0.04660957528779719,
    0.6958462168715666,
    0.03185002109034476,
    0.717322587343012,
    0.015229061349936731
  ],
  right_pose: [
  ],
  dry_run: false,
  disable_environment_collision: true,
  use_end_effector_frame: false,
  speed: 0.03
}"


ros2 action send_goal /moveT \
  task_interfaces/action/MoveArmPose "{
  left_pose: [

], 
  right_pose: [
0.0,
0.0,
0.05, 
0.0,
0.0, 
0.0

],
  dry_run: false,
  disable_environment_collision: true,
  speed: 1.0,
  velocity: 0.1}" \
  --feedback


ros2 service call /stop_motion std_srvs/srv/Trigger "{}"

ros2 service call /resume_motion std_srvs/srv/Trigger "{}"