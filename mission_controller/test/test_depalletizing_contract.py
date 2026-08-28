from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestDepalletizingContract(unittest.TestCase):
    def test_top_level_action_goal_contains_only_start(self):
        action = (
            REPO_ROOT
            / "mission_interfaces"
            / "action"
            / "ExecuteWorkflow.action"
        ).read_text(encoding="utf-8")
        goal_section = action.split("---", 1)[0]
        goal_fields = [
            line.strip()
            for line in goal_section.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]

        self.assertEqual(goal_fields, ["bool start"])

    def test_workflow_node_has_explicit_multithreaded_executor(self):
        source = (
            REPO_ROOT
            / "mission_controller"
            / "mission_runtime"
            / "taskflow"
            / "node.py"
        ).read_text(encoding="utf-8")

        self.assertIn("MultiThreadedExecutor(num_threads=4)", source)
        self.assertGreaterEqual(source.count("ReentrantCallbackGroup()"), 2)

    def test_workflow_launch_is_unconditional(self):
        launch_source = (
            REPO_ROOT
            / "mission_controller"
            / "launch"
            / "mission.launch.py"
        ).read_text(encoding="utf-8")
        system_source = (
            REPO_ROOT
            / "mission_controller"
            / "launch"
            / "mission_system.launch.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("enable_execute_workflow", launch_source)
        self.assertNotIn("enable_execute_workflow", system_source)
        self.assertIn('executable="execute_workflow"', launch_source)

    def test_mqtt_dependency_is_declared(self):
        setup_source = (
            REPO_ROOT / "mission_controller" / "setup.py"
        ).read_text(encoding="utf-8")
        package_xml = (
            REPO_ROOT / "mission_controller" / "package.xml"
        ).read_text(encoding="utf-8")

        self.assertIn("paho-mqtt>=1.5,<3", setup_source)
        self.assertIn("<exec_depend>python3-paho-mqtt</exec_depend>", package_xml)


if __name__ == "__main__":
    unittest.main()
