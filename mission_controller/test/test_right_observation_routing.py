"""Offline camera-routing checks; never send robot commands."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import math
import pytest
import yaml
from mission_runtime.box_geometry import BoxGeometryMixin
from mission_runtime.box_perception import BoxPerceptionMixin
from mission_runtime.box_preparation import BoxPreparationMixin
from mission_runtime.taskflow.fixed_operations import FixedRosWorkflowOperations
from test_box_layer_and_post_arm_movej import _DetectionOrderHarness

ROOT = Path(__file__).resolve().parents[1]
def parameters():
    merged = {}
    for name in ("core", "camera", "box_common", "grasp_tf", "drag"):
        merged.update(yaml.safe_load((ROOT/"config/mission"/(name+".yaml")).read_text())["mission_controller"]["ros__parameters"])
    return merged

@pytest.mark.parametrize("drag_mode", [False, True])
@pytest.mark.parametrize("model", ["bigbox", "smallbox"])
def test_detection_action_requests_right_camera(drag_mode, model):
    values = parameters()
    class Captured(Exception): pass
    def capture(goal, **kwargs):
        assert goal.camera_side == "right"
        assert goal.model_label == model
        raise Captured()
    class Harness(BoxPerceptionMixin, BoxGeometryMixin):
        box_object_pose_client = SimpleNamespace(wait_for_server=lambda **kw: True, send_goal_async=capture)
        def _string(self, name): return str(values[name])
        def _float(self, name): return 0.0 if name == "box_foundation_pose_pre_settle_sec" else 2.0
        def _integer(self, name): return 0
        def _check_canceled(self, *args): pass
    with pytest.raises(Captured):
        Harness()._call_box_object_pose(None, SimpleNamespace(box_type=model, target_label=0),
                                       tf_mode=True, drag_mode=drag_mode)

def test_right_drag_observation_preserves_staged_joint_entry():
    harness = _DetectionOrderHarness()
    BoxPreparationMixin._execute_pre_detection_arm_movej_fixed(
        harness, None, True, 2, "bigbox", arm="right", tf_mode=True, drag_mode=True)
    assert [kw["target_joint_indices"] for _, kw in harness.intermediate_calls] == [(1,), (0, 1)]
    assert all(args[3] == "right" for args, _ in harness.intermediate_calls)

def test_all_right_observation_angles_respect_configured_limits():
    p = parameters()
    lower, upper = p["waist_workspace_right_arm_joint_min_deg"], p["waist_workspace_right_arm_joint_max_deg"]
    for model in ("bigbox", "smallbox"):
        for layer in range(1, 5):
            q = p[f"drag_box_tf_box_layer_pre_detection_right_movej_joint_units_{model}_layer{layer}"]
            assert len(q) == 7
            assert all(lo <= x/1000.0 <= hi for x, lo, hi in zip(q, lower, upper))

@pytest.mark.parametrize("backend", ["sdk", "action"])
def test_global_right_observation_routes_motion_to_right_arm(backend):
    obj = FixedRosWorkflowOperations.__new__(FixedRosWorkflowOperations)
    obj._dry_run = False
    obj._sdk_adapter = Mock() if backend == "sdk" else None
    obj._arm_joints_speed_percent = 15.0
    obj._arm_joints_timeout_sec = 120.0
    obj.is_cancel_requested = lambda: False
    obj._arm_joints_client = object()
    obj._arm_joints_duration = 0.0
    obj._arm_joints_action_name = "/move_arm_j"
    obj._call_action = Mock(return_value=SimpleNamespace(success=True, result=SimpleNamespace(success=True, message="ok")))
    joints = [0.1] * 7
    assert obj.move_right_joints(joints, "observation").success
    if backend == "sdk":
        call = obj._sdk_adapter.execute_single_movej.call_args.kwargs
        assert call["arm"] == "right"
        assert call["joint_degrees"] == pytest.approx([math.degrees(.1)]*7)
    else:
        goal = obj._call_action.call_args.args[1]
        assert list(goal.left_joints) == []
        assert list(goal.right_joints) == joints

def test_global_observation_config_selects_right_and_mirrors_calibration():
    payload = yaml.safe_load((ROOT/"config/mission/taskflow.yaml").read_text())
    p = next(iter(payload.values()))["ros__parameters"]
    assert p["global_observation_camera_side"] == "right"
    left = p["fixed_workflow_global_observation_left_joints"]
    right = p["fixed_workflow_global_observation_right_joints"]
    assert right == pytest.approx([s*q for s,q in zip((-1,1,-1,1,-1,-1,-1),left)])
