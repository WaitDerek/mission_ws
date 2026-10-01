from mission_runtime.arm_target_visualization import make_markers
from visualization_msgs.msg import Marker

def test_markers_show_before_after_spheres_without_link8_shapes():
    contact = {"frame": "base_footprint", "xyz": [.2, -.6, .9], "q": [0., 0., 0., 1.]}
    link = dict(contact, xyz=[.2, -.55, 1.0])
    raw = dict(contact, xyz=[.2, -.62, .9])
    record = {"phase": "DragBox initial", "box": contact,
              "arms": {"left": {"raw": raw, "contact": contact, "link8": link}}}
    markers = make_markers(record).markers
    assert markers[0].action == Marker.DELETEALL
    assert all(m.type in (Marker.SPHERE, Marker.TEXT_VIEW_FACING) for m in markers[1:])
    sphere = next(m for m in markers if m.type == Marker.SPHERE and m.scale.x == .045)
    assert sphere.pose.position.y == -.6
    before = next(m for m in markers if m.type == Marker.SPHERE and m.scale.x == .03)
    assert before.pose.position.y == -.62
    assert any('BEFORE' in m.text for m in markers)
    assert any('AFTER' in m.text for m in markers)
    assert all(m.header.frame_id == "base_footprint" for m in markers[1:])
    assert len(set(m.id for m in markers[1:])) == len(markers)-1

def test_actual_markers_expire_when_tf_stops():
    p = {"frame": "base_footprint", "xyz": [0.,0.,1.], "q": [0.,0.,0.,1.]}
    markers = make_markers({"arms": {"right": {"contact": p, "link8": p}}}, actual=True).markers
    assert all(m.lifetime.sec == 2 for m in markers[1:])
    assert any("ACTUAL" in m.text for m in markers)
