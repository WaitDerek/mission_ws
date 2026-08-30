from mission_controller.taskflow.model import StepResult
from mission_controller.taskflow.state_machine import AssemblyWorkflowEngine


class _Operations:
    def __init__(self, *, fail_at=""):
        self.fail_at = fail_at
        self.calls = []
        self.canceled = False

    def is_cancel_requested(self):
        return self.canceled

    def _result(self, name):
        self.calls.append(name)
        return StepResult(name != self.fail_at, f"{name} result")

    def navigate(self, request):
        return self._result(f"navigate:{request.point_id}")

    def grip(self, _request_id, target_type):
        return self._result(f"grip:{target_type}")

    def peel(self, _request_id):
        return self._result("peel:badge")

    def assemble(self, _request_id, target_type):
        return self._result(f"assemble:{target_type}")


def test_workflow_executes_connector_then_badge_sequence():
    operations = _Operations()
    progress = []
    outcome = AssemblyWorkflowEngine(
        operations,
        progress_callback=progress.append,
    ).run("workflow-test")

    assert outcome.success
    assert outcome.completed_task_count == 5
    assert operations.calls == [
        "navigate:1",
        "grip:connector",
        "navigate:3",
        "assemble:connector",
        "navigate:2",
        "grip:badge",
        "peel:badge",
        "navigate:3",
        "assemble:badge",
    ]
    assert [item.current_point_id for item in progress if item.current_point_id] == [
        "1",
        "3",
        "2",
        "3",
        "3",
    ]
    assert progress[-1].stage == "COMPLETE"
    assert progress[-1].current_step == 9


def test_workflow_stops_immediately_after_failed_step():
    operations = _Operations(fail_at="assemble:connector")
    outcome = AssemblyWorkflowEngine(operations).run("workflow-test")

    assert not outcome.success
    assert outcome.final_stage == "ASSEMBLY_CONNECTOR"
    assert outcome.completed_task_count == 1
    assert operations.calls == [
        "navigate:1",
        "grip:connector",
        "navigate:3",
        "assemble:connector",
    ]


def test_workflow_stops_until_platform_navigation_succeeds():
    operations = _Operations(fail_at="navigate:3")
    outcome = AssemblyWorkflowEngine(operations).run("workflow-test")

    assert not outcome.success
    assert outcome.final_stage == "NAVIGATE_3_CONNECTOR"
    assert outcome.completed_task_count == 1
    assert operations.calls == [
        "navigate:1",
        "grip:connector",
        "navigate:3",
    ]


def test_workflow_reports_cancellation_before_first_step():
    operations = _Operations()
    operations.canceled = True
    outcome = AssemblyWorkflowEngine(operations).run("workflow-test")

    assert not outcome.success
    assert outcome.final_stage == "CANCELED"
    assert operations.calls == []


def test_workflow_point_mapping_is_configurable_within_four_points():
    operations = _Operations()
    outcome = AssemblyWorkflowEngine(
        operations,
        connector_point_id="2",
        badge_point_id="1",
        assembly_point_id="4",
    ).run("workflow-test")

    assert outcome.success
    assert [call for call in operations.calls if call.startswith("navigate:")] == [
        "navigate:2",
        "navigate:4",
        "navigate:1",
        "navigate:4",
    ]
