# RealBot box mission

```zsh
ros2 launch mission_controller mission_system.launch.py \
  mode:=hardware \
  enable_rviz:=false \
  enable_robot_state_publisher:=false
```

发送 box grasp：

```zsh
ros2 action send_goal --feedback \
  /execute_box_grasp \
  mission_interfaces/action/ExecuteBoxGrasp \
  "{dry_run: false}"
```

发送 box place：

```zsh
ros2 action send_goal --feedback \
  /execute_box_place \
  mission_interfaces/action/ExecuteBoxPlace \
  "{dry_run: false}"
```
