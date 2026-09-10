from stereo_depth_bridge import contracts


def test_input_topic_contract():
    assert contracts.LEFT_COMPRESSED_TOPIC == (
        "/left_head_wide_cam/color/image_raw/compressed"
    )
    assert contracts.LEFT_CAMERA_INFO_TOPIC == (
        "/left_head_wide_cam/color/camera_info"
    )
    assert contracts.RIGHT_COMPRESSED_TOPIC == (
        "/right_head_wide_cam/color/image_raw/compressed"
    )
    assert contracts.RIGHT_CAMERA_INFO_TOPIC == (
        "/right_head_wide_cam/color/camera_info"
    )


def test_output_topic_contract():
    assert contracts.LEFT_COLOR_OUTPUT_TOPIC == "/stereo/left/color/image_raw"
    assert contracts.LEFT_DEPTH_OUTPUT_TOPIC == "/stereo/left/depth/image_raw"
    assert contracts.LEFT_CAMERA_INFO_OUTPUT_TOPIC == (
        "/stereo/left/color/camera_info"
    )
