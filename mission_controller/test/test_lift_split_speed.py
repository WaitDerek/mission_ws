import pytest
from geometry_msgs.msg import Pose
from mission_runtime.mission_controller import MissionController
from test_drag_box_tf_action import _DragTfSequenceHarness, _SequenceAdapter, _MotionGoal

@pytest.mark.parametrize("drag_mode", [True,False])
def test_lift_uses_independent_speeds(drag_mode):
    h=_DragTfSequenceHarness(carry_enabled=False)
    h.values.update({"box_post_movel_enabled": True,
                     "grasp_box_tf_body_home_carry_enabled":False,
                     "drag_box_tf_post_movel_sdk_motion_mode":"movel",
                     "grasp_box_tf_post_movel_sdk_motion_mode":"movel",
                     "box_post_lift_left_velocity_percent":10.,
                     "box_post_lift_right_velocity_percent":15.})
    h._post_movel_targets_with_labels=lambda *a,**k:[("step2",Pose(),Pose())]
    a=_SequenceAdapter(h.events)
    speeds=[]
    def endpoint(left,right,ls,rs,**kwargs):
        speeds.append((ls,rs))
        return "split lift"
    a.execute_dual_movel_endpoint=endpoint
    MissionController._execute_post_movel_sequence(
        h,_MotionGoal(),a,Pose(),Pose(),False,drag_mode=drag_mode,
        right_arm_only=False,delayed_left_join=False,tf_mode=True,
        model_label="bigbox",box_layer=1)
    assert speeds==[(10.,15.)]
