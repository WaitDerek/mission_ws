ros2 launch mission_controller mission_system.launch.py \
  mode:=hardware \
  pipeline:=box \
  hardware_armed:=true \
  dry_run:=false \
  enable_rviz:=false \
  enable_robot_state_publisher:=false

ros2 action send_goal --feedback \
  /execute_box_stack \
  mission_interfaces/action/ExecuteBoxStack \
  "{level: 1}"
