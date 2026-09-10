import ast
from pathlib import Path

def test_launch_source_is_valid_and_wires_required_components():
    launch_file = (
        Path(__file__).resolve().parents[1]
        / "launch"
        / "ess_stereo_depth.launch.py"
    )
    source = launch_file.read_text(encoding="utf-8")
    ast.parse(source, filename=str(launch_file))
    assert "nvidia::isaac_ros::dnn_stereo_depth::ESSDisparityNode" in source
    assert "nvidia::isaac_ros::stereo_image_proc::DisparityToDepthNode" in source
    assert "nvidia::isaac_ros::image_proc::ImageFormatConverterNode" in source
