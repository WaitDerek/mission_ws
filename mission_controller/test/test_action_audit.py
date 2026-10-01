import json
from types import SimpleNamespace

from rclpy.action import GoalResponse
from mission_runtime.action_audit import ActionAuditMixin, ActionAuditStore


class _Logger:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


class _Node(ActionAuditMixin):
    def __init__(self, directory):
        self._action_audit = ActionAuditStore("mission_controller", _Logger(), str(directory))


def test_action_goal_feedback_result_are_saved_per_goal(tmp_path):
    node = _Node(tmp_path)
    request = SimpleNamespace(request_id="audit-001", start=True)
    goal_handle = SimpleNamespace(
        goal_id=SimpleNamespace(uuid=bytes(range(16))),
        request=request,
        status=4,
    )
    assert node._audit_goal_callback("grasp_box_tf", lambda _: GoalResponse.ACCEPT)(request) == GoalResponse.ACCEPT

    def execute(handle):
        node._audit_feedback(handle, SimpleNamespace(stage="MOVING", detail="test"))
        return SimpleNamespace(success=True, message="done")

    node._audit_execute_callback("grasp_box_tf", execute)(goal_handle)
    goal_file = next(tmp_path.glob("*/mission_controller/grasp_box_tf_*.jsonl"))
    events = [json.loads(line) for line in goal_file.read_text().splitlines()]
    assert [event["event"] for event in events] == ["start", "feedback", "result"]
    assert events[-1]["success"] is True
    assert events[-1]["goal_status"] == 4
    assert events[-1]["message"] == "done"
    decisions = next(tmp_path.glob("*/mission_controller/goal_decisions.jsonl"))
    assert json.loads(decisions.read_text().splitlines()[0])["accepted"] is True


def test_rejected_goal_is_saved(tmp_path):
    node = _Node(tmp_path)
    request = SimpleNamespace(request_id="rejected")
    assert node._audit_goal_callback("navigate_to_point", lambda _: GoalResponse.REJECT)(request) == GoalResponse.REJECT
    decisions = next(tmp_path.glob("*/mission_controller/goal_decisions.jsonl"))
    event = json.loads(decisions.read_text().splitlines()[0])
    assert event["accepted"] is False
    assert event["request_id"] == "rejected"
