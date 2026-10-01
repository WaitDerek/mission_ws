"""Continuous Place mode: mocked sensing/motion, no robot communication."""
import math
import time
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from scipy.spatial.transform import Rotation
from mission_runtime.box_carry import BoxCarryMixin
from mission_runtime.box_support import BoxSupportMixin
from mission_runtime.common import MissionError
from mission_runtime.realman_sdk_adapter import RealManSdkError

I = ((0.,0.,0.),(0.,0.,0.,1.))
Q = (0.,-math.sqrt(.5),0.,math.sqrt(.5))
ARMS = ('left','right')

class Place(BoxCarryMixin):
    def __init__(self):
        self.mode = 'continuous'
        self.values = {
            'place_box_test_continuous_left_min_z_footprint_m':.9,
            'place_box_test_continuous_right_min_z_footprint_m':.9,
            'place_box_test_continuous_left_velocity_percent':10.,
            'place_box_test_continuous_right_velocity_percent':15.,
            'place_box_test_continuous_max_drop_difference_m':.03,
            'place_box_test_table_support_delta_fz_n':3.,
            'place_box_test_post_support_max_abs_work_fz_n':200.,
            'place_box_test_table_unloaded_abs_fz_n':100.,
            'place_box_test_table_support_sign_left':1.,
            'place_box_test_table_support_sign_right':1.,
            'place_box_test_table_max_overtravel_m':.005,
            'place_box_test_table_early_contact_tolerance_m':.5,
            'place_box_test_timeout_sec':3.,
            'place_box_test_descent_z_feedback_max_age_sec':.25,
        }
        self.actual = {'left':((.3,-.4,1.1),Q),'right':((-.3,-.4,1.102),Q)}
        self.bases = dict.fromkeys(ARMS,I)
        self.feedback = []
        self.read_failure = False
        self._place_box_test_wait_arms_still = Mock()
        self._place_box_test_stable_work_fz_baseline = Mock(return_value=dict.fromkeys(ARMS,0.))

    def _float(self,n): return self.values[n]
    def _string(self,n): return self.mode if n=='place_box_test_descent_mode' else n
    def _check_canceled(self,*_): pass
    def _lookup_tf_carry_transform(self,_,frame):
        if frame=='base_footprint': return I
        return self.bases[frame.split('_')[0]] if frame.endswith('arm_base_frame') else self.actual[frame.split('_')[0]]
    def _place_box_test_read_z_check_tf(self,*_):
        return (None,0,'stale') if self.read_failure else (self.actual,1,'')
    def _publish_place_box_test_feedback(self,_,stage,detail): self.feedback.append((stage,detail))

def run(place, adapter, table=.7):
    return place._place_box_test_descend_to_table(SimpleNamespace(is_cancel_requested=False),adapter,
        'base_link',dict.fromkeys(ARMS,I),I,table,.1)

def adapter_for(place, *, support=True, controlled_stop=False, failure=False):
    adapter = Mock()
    adapter.read_bilateral_work_fz.return_value = dict.fromkeys(ARMS,4. if support else 0.)
    def move(left,right,ls,rs,**kw):
        if controlled_stop:
            kw['after_start']()
            deadline=time.monotonic()+1
            while not adapter.stop_all.called and time.monotonic()<deadline: time.sleep(.01)
            raise RealManSdkError('stopped')
        if failure: raise RealManSdkError('controller failure')
        for a,p in zip(ARMS,(left,right)):
            pose=(tuple(p[:3]),tuple(Rotation.from_euler('xyz',p[3:]).as_quat()))
            place.actual[a]=BoxSupportMixin._compose_transform(place.bases[a],pose)
        kw['after_start']()
        time.sleep(.02)
    adapter.execute_dual_movel_endpoint.side_effect=move
    return adapter

def test_continuous_one_endpoint_fixed_world_xy_orientation_and_speeds():
    place=Place()
    angle=.4
    place.bases={a:((.01,0.,.2),(0.,math.sin(angle/2),0.,math.cos(angle/2))) for a in ARMS}
    start=dict(place.actual)
    adapter=adapter_for(place)
    run(place,adapter)
    assert adapter.execute_dual_movel_endpoint.call_count==1
    args=adapter.execute_dual_movel_endpoint.call_args.args
    assert args[2:]==(10.,15.)
    for a in ARMS:
        assert place.actual[a][0]==pytest.approx((start[a][0][0],start[a][0][1],start[a][0][2]-.2))
        assert abs(sum(x*y for x,y in zip(place.actual[a][1],start[a][1])))==pytest.approx(1.)

def test_max_distance_capped_by_table_clearance():
    place=Place(); adapter=adapter_for(place)
    run(place,adapter,table=.95)
    assert adapter.execute_dual_movel_endpoint.call_args.args[0][2]==pytest.approx(1.1-.056)

def test_no_support_never_silently_falls_back_or_releases():
    place=Place();adapter=adapter_for(place,support=False)
    place._place_box_test_descend_to_table_segmented=Mock()
    with pytest.raises(MissionError,match='without stable bilateral'):
        run(place,adapter)
    place._place_box_test_descend_to_table_segmented.assert_not_called()

def test_expected_contact_stop_still_requires_bilateral_support():
    place=Place();adapter=adapter_for(place,controlled_stop=True)
    run(place,adapter)
    adapter.stop_all.assert_called()
    assert place.feedback[-1][0]=='TABLE_SUPPORT_CONFIRMED'

def test_controller_error_is_not_contact_success():
    place=Place();adapter=adapter_for(place,failure=True)
    with pytest.raises(RealManSdkError,match='controller failure'): run(place,adapter)

def test_stale_feedback_stops_and_refuses_release():
    place=Place();place.read_failure=True;adapter=adapter_for(place,controlled_stop=True)
    with pytest.raises(MissionError,match='TF unavailable'): run(place,adapter)
    adapter.stop_all.assert_called()

def test_invalid_speed_does_not_move():
    place=Place();place.values['place_box_test_continuous_left_velocity_percent']=float('nan')
    adapter=Mock()
    with pytest.raises(MissionError,match='invalid continuous'): run(place,adapter)
    adapter.execute_dual_movel_endpoint.assert_not_called()

def test_at_minimum_z_does_not_move_lower():
    place=Place();place.values['place_box_test_continuous_left_min_z_footprint_m']=1.1
    adapter=Mock()
    with pytest.raises(MissionError,match='minimum TCP Z'): run(place,adapter)
    adapter.execute_dual_movel_endpoint.assert_not_called()

def test_segmented_mode_retained():
    place=Place();place.mode='segmented'
    place._place_box_test_descend_to_table_segmented=Mock(return_value='old')
    place._place_box_test_descend_continuous=Mock()
    assert run(place,Mock())=='old'
    place._place_box_test_descend_to_table_segmented.assert_called_once()
    place._place_box_test_descend_continuous.assert_not_called()

def test_new_mode_does_not_run_extra_three_cm_after_support():
    from test_place_box_waist_only import WaistOnlyPlace, run as run_waist
    place=WaistOnlyPlace()
    place.post_distance=.03
    old_string,old_float,old_lookup=place._string,place._float,place._lookup_tf_carry_transform
    place._string=lambda n: 'continuous' if n=='place_box_test_descent_mode' else old_string(n)
    values={'place_box_test_continuous_left_min_z_footprint_m':.975131,
            'place_box_test_continuous_right_min_z_footprint_m':.966566,
            'place_box_test_continuous_left_velocity_percent':10.,
            'place_box_test_continuous_right_velocity_percent':15.,
            'place_box_test_table_max_overtravel_m':.005}
    place._float=lambda n: values[n] if n in values else old_float(n)
    place._lookup_tf_carry_transform=lambda frame,n: I if n=='base_footprint' else old_lookup(frame,n)
    result=run_waist(place,Mock())
    assert 'force_and_descent' in place.events
    assert 'post_support' not in place.events
    assert 'post_support_arm_base_z_descent=0.0000m' in result[0]

def test_floor_is_footprint_height_not_base_link_height():
    place=Place()
    old_lookup=place._lookup_tf_carry_transform
    place._lookup_tf_carry_transform=lambda frame,n: ((0.,0.,-.035),I[1]) if n=='base_footprint' else old_lookup(frame,n)
    adapter=adapter_for(place)
    run(place,adapter)
    assert adapter.execute_dual_movel_endpoint.call_args.args[0][2]==pytest.approx(.865)
