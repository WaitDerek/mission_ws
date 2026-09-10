# stereo_depth_bridge

ROS 2 Humble adapter for the robot's two compressed wide-angle RGB streams and
NVIDIA Isaac ROS 3.2 ESS. It does not change Mission actions, MQTT, TF, or robot
motion logic.

## Data flow

```text
left/right CompressedImage + CameraInfo
  -> synchronized JPEG decode
  -> stereo undistortion and rectification
  -> Isaac ROS GPU resize/color conversion
  -> ESS disparity
  -> Isaac ROS disparity-to-depth
  -> left-aligned bgr8 + 32FC1 depth (metres) + CameraInfo
```

Inputs:

```text
/left_head_wide_cam/color/image_raw/compressed
/left_head_wide_cam/color/camera_info
/right_head_wide_cam/color/image_raw/compressed
/right_head_wide_cam/color/camera_info
```

Outputs:

```text
/stereo/left/color/image_raw      sensor_msgs/Image, bgr8
/stereo/left/depth/image_raw      sensor_msgs/Image, 32FC1 metres
/stereo/left/color/camera_info    sensor_msgs/CameraInfo
```

## Calibration placeholder

Edit `config/stereo_depth.yaml` when the fixed left-to-right extrinsic is known.
The convention is the OpenCV stereo convention:

```text
X_right = R * X_left + T
```

For a parallel pair whose right camera is on the left camera's positive X side,
`T` is normally `[-baseline_m, 0, 0]`. Do not copy this example without checking
the actual camera coordinate frames.

With `calibration_source: auto`, the node first uses a valid baseline already
encoded in the two `CameraInfo.P` matrices. Otherwise it uses the configured
fixed R/T. The checked-in zero translation is intentionally invalid: frames are
rejected and no false metric depth is published until calibration is available.

## Target dependencies

Use the Isaac ROS 3.2 packages built for Jetson Orin, JetPack 6.1/6.2, Ubuntu
22.04 and ROS 2 Humble:

```text
isaac_ros_ess
isaac_ros_image_proc
isaac_ros_stereo_image_proc
```

An ESS TensorRT engine is also required. Build this workspace after installing
those target dependencies.

## Launch

```bash
source /opt/ros/humble/setup.bash
source /home/dekc/april/changan/mission_ws/install/setup.bash

ros2 launch stereo_depth_bridge ess_stereo_depth.launch.py \
  engine_file_path:=/absolute/path/to/ess.engine
```

Light ESS can be selected by supplying its matching engine and dimensions:

```bash
ros2 launch stereo_depth_bridge ess_stereo_depth.launch.py \
  engine_file_path:=/absolute/path/to/light_ess.engine \
  model_input_width:=480 \
  model_input_height:=288
```

Before running FoundationPose, verify encoding, dimensions, timestamps and
metric range:

```bash
ros2 topic echo /stereo/left/color/image_raw --field encoding --once
ros2 topic echo /stereo/left/depth/image_raw --field encoding --once
ros2 topic echo /stereo/left/color/camera_info --once
```
