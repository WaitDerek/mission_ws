"""Stable ROS topic defaults for the stereo depth pipeline."""

LEFT_COMPRESSED_TOPIC = "/left_head_wide_cam/color/image_raw/compressed"
LEFT_CAMERA_INFO_TOPIC = "/left_head_wide_cam/color/camera_info"
RIGHT_COMPRESSED_TOPIC = "/right_head_wide_cam/color/image_raw/compressed"
RIGHT_CAMERA_INFO_TOPIC = "/right_head_wide_cam/color/camera_info"

LEFT_COLOR_OUTPUT_TOPIC = "/stereo/left/color/image_raw"
LEFT_DEPTH_OUTPUT_TOPIC = "/stereo/left/depth/image_raw"
LEFT_CAMERA_INFO_OUTPUT_TOPIC = "/stereo/left/color/camera_info"

LEFT_RECTIFIED_TOPIC = "/stereo/internal/left/image_rect"
LEFT_RECTIFIED_INFO_TOPIC = "/stereo/internal/left/camera_info_rect"
RIGHT_RECTIFIED_TOPIC = "/stereo/internal/right/image_rect"
RIGHT_RECTIFIED_INFO_TOPIC = "/stereo/internal/right/camera_info_rect"

LEFT_ESS_TOPIC = "/stereo/internal/left/image_ess"
RIGHT_ESS_TOPIC = "/stereo/internal/right/image_ess"
RIGHT_ESS_INFO_TOPIC = "/stereo/internal/right/camera_info_ess"
DISPARITY_TOPIC = "/stereo/internal/disparity"
