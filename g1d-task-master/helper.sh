ssh csc101@192.168.10.67
ssh csc101@192.168.10.117
ssh csc101@192.168.123.164


# ------------------------------ Vision -------------------------------------
cd /home/csc101/code/changan/vision_ws/hangcha-perception 
export VISION_WS=/home/csc101/code/changan/vision_ws/hangcha-perception
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=23
ros2 launch realsense_d405_bringup dual_d405.launch.py

cd /home/csc101/code/changan/vision_ws/hangcha-perception 
export VISION_WS=/home/csc101/code/changan/vision_ws/hangcha-perception
source /opt/ros/humble/setup.zsh
source install/setup.zsh
export ROS_DOMAIN_ID=23
./start_object_pose_action.sh --config g1d

source /opt/ros/humble/setup.zsh
source /home/csc101/code/changan/vision_ws/hangcha-perception/install/setup.zsh
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
source /opt/ros/humble/setup.zsh
source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
ros2 launch robot_bringup test.launch.py \
  robot_profile:=g1_d \
  robot_adapter:=g1d \
  robot_ip:=enP8p1s0 \
  dry_run:=false \
  prefer_hardware:=true \
  allow_mock_fallback:=false \
  enable_robot_state_publisher:=true \
  enable_move_group:=true \
  enable_rviz:=false

ros2 topic pub --once /g1_d/torso/command \
  task_interfaces/msg/G1dTorsoCommand \
  "{control_mode: 3, target_position: 0.25, speed: 0.2}"
# ---------------------------------------------------------------------------


# ------------------------------ Task Dispatch ------------------------------
source /opt/ros/humble/setup.zsh 
source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
source /home/csc101/code/changan/driver_ws/install/setup.zsh
source /home/csc101/code/changan/task_ws/install/setup.zsh
ros2 launch execute_grasp_script_runner \
    execute_grasp_script.launch.py

source /home/csc101/code/changan/task_ws/install/setup.zsh
ros2 action send_goal --feedback /execute_grasp \
mission_interfaces/action/ExecuteGrasp "{request_id: 'grasp_sim_test',
      target_frame: 'torso_link4',
      target_label: 0,
      arm: 'right',
      publish_pose: true,
      detection_timeout_sec: 30.0,
      dry_run: false}"
# ---------------------------------------------------------------------------


# ------------------------------ Full Pipline -------------------------------
source /opt/ros/humble/setup.zsh
source /home/csc101/code/changan/vision_ws/hangcha-perception/install/setup.zsh
source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
source /home/csc101/code/changan/driver_ws/install/setup.zsh
export ROS_DOMAIN_ID=23

cd /home/csc101/code/changan/task_ws/src/execute_grasp_script_runner
conda activate changan
python full_task.py
# ---------------------------------------------------------------------------


# ---------------------------- Sensor & Gripper -----------------------------
source /home/csc101/code/changan/driver_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
bash /home/csc101/code/changan/driver_ws/src/brainsys-drivers/script/run_usb_relay.sh

source /home/csc101/code/changan/driver_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
bash /home/csc101/code/changan/driver_ws/src/brainsys-drivers/script/run_kwr57b.sh

source /home/csc101/code/changan/driver_ws/install/setup.zsh
export ROS_DOMAIN_ID=23
bash /home/csc101/code/changan/driver_ws/src/brainsys-drivers/src/brsd_quad/script/usb_relay_test.sh 3 on
bash /home/csc101/code/changan/driver_ws/src/brainsys-drivers/src/brsd_quad/script/usb_relay_test.sh 3 off
ros2 topic echo  --once /force_torque/data
# ---------------------------------------------------------------------------


source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
ros2 action send_goal /move_arm_j task_interfaces/action/MoveArmJoints \
"{
  left_joints: [-0.11945875734090805,
                0.47609731554985046,
                -0.10206964612007141,
                0.3019784986972809,
                -1.0567809343338013,
                1.5041520595550537,
                -1.0111570358276367], 
  right_joints: [],
  dry_run: false, 
  duration: 0.0
}"


ros2 action send_goal /move_arm_p task_interfaces/action/MoveArmPose --feedback \
"{
  left_pose: [-0.021, 0.157, -0.164, -0.559, 0.500, -0.475, 0.460],
  right_pose: [],
  dry_run: false
}"

ros2 action send_goal /move_arm_p task_interfaces/action/MoveArmPose --feedback \
"{
  left_pose: [],
  right_pose: [0.172, -0.205, 0.068, 0.0, 0.7071067811865475, 0.0, 0.7071067811865475],
  dry_run: false
}"


source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
ros2 action send_goal /move_arm_j task_interfaces/action/MoveArmJoints \
"{
  left_joints: [1.147561,
                0.397265,
                -0.167563,
                0.038362,
                -0.090948,
                -0.497813,
                -0.267452], 
  right_joints: [],
  dry_run: false, 
  duration: 0.0
}"

ros2 action send_goal /move_arm_p task_interfaces/action/MoveArmPose --feedback \
"{
  left_pose: [0.243514835, 0.203978889, 0.091616470,
              0.008916218, -0.654694020, 0.752701793, -0.068820430],
  right_pose: [],
  dry_run: false,
  disable_environment_collision: false
}"

newposition:
 0.3853168189525604
 0.09362076967954636
 -0.014524880796670914
 -0.4527280628681183
 -0.05041763558983803
 0.6298189759254456
 -0.1947077065706253

source /home/csc101/code/changan/dual_arm_ws/install/setup.zsh
ros2 action send_goal /move_arm_j task_interfaces/action/MoveArmJoints \
"{
  left_joints: [-0.1765635907649994,
                0.3367447555065155,
                -0.06720753759145737,
                0.1793319433927536,
                -1.102656602859497,
                1.6068569421768188,
                -0.8085277676582336], 
  right_joints: [],
  dry_run: false, 
  duration: 0.0
}"

pose:
  position:
    x: 0.15497289462809247
    y: 0.17195157356366308
    z: 0.020458308874041098
  orientation:
    x: -0.21232910150997428
    y: 0.1947474401716411
    z: -0.7188384243263292
    w: 0.6326619214956756





ros2 action send_goal /move_arm_l task_interfaces/action/MoveArmPose --feedback \
"{
  left_pose: [0.15493438457416647, 0.27232637090151, 0.020466678901818303,
  -0.21230355101766363, 0.19489312614334633, -0.7182728697869052,0.6332676812655508],
  right_pose: [],
  dry_run: false
}"

ros2 action send_goal /moveT task_interfaces/action/MoveArmPose --feedback \
"{left_pose: [0.0, 0.01, 0.0, 0.0, 0.0, 0.0], right_pose: [], dry_run: false}"