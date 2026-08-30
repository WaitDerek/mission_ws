"""Register grip, peel, and assembly Actions on the mission controller."""

from mission_interfaces.action import ExecuteAssembly, ExecuteGrip, ExecutePeel
from rclpy.action import ActionServer

from .assembly_action import AssemblyActionMixin
from .execute_grip import ExecuteGripMixin
from .execute_peel import ExecutePeelMixin
from .manipulation_runtime import ManipulationRuntimeMixin
from .run_grip import RunGripMixin
from .run_peel import RunPeelMixin
from .run_pose_runtime import RunPoseRuntimeMixin
from .suction_runtime import SuctionRuntimeMixin


class ManipulationMixin(
    AssemblyActionMixin,
    ExecuteGripMixin,
    ExecutePeelMixin,
    RunGripMixin,
    RunPeelMixin,
    RunPoseRuntimeMixin,
    SuctionRuntimeMixin,
    ManipulationRuntimeMixin,
):
    """Add exclusive manipulation Actions to the main controller."""

    def _initialize_manipulation(self) -> None:
        self._initialize_pipeline_runtime()
        self._initialize_run_pose_runtime()
        self._initialize_suction_runtime()
        self.get_logger().warning(
            "legacy /run_grip and /run_peel Actions are loaded for migration; "
            "workflow execution uses ExecuteGrip/ExecutePeel Actions"
        )

        self.run_grip_action_server = ActionServer(
            self,
            ExecuteGrip,
            self._string("run_grip_action_name"),
            execute_callback=self._execute_run_grip,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.run_peel_action_server = ActionServer(
            self,
            ExecutePeel,
            self._string("run_peel_action_name"),
            execute_callback=self._execute_run_peel,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.execute_grip_action_server = ActionServer(
            self,
            ExecuteGrip,
            self._string("execute_grip_action_name"),
            execute_callback=self._execute_grip,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.execute_peel_action_server = ActionServer(
            self,
            ExecutePeel,
            self._string("execute_peel_action_name"),
            execute_callback=self._execute_peel,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.execute_assembly_action_server = ActionServer(
            self,
            ExecuteAssembly,
            self._string("execute_assembly_action_name"),
            execute_callback=self._execute_assembly,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=self._callback_group,
        )
        self.get_logger().info(
            "manipulation actions ready: run_grip=%s run_peel=%s "
            "grip=%s peel=%s assembly=%s"
            % (
                self._string("run_grip_action_name"),
                self._string("run_peel_action_name"),
                self._string("execute_grip_action_name"),
                self._string("execute_peel_action_name"),
                self._string("execute_assembly_action_name"),
            )
        )

    @staticmethod
    def _pipeline_feedback(goal_handle, action_type, stage: str, detail: str) -> None:
        feedback = action_type.Feedback()
        feedback.stage = stage
        feedback.detail = detail
        goal_handle.publish_feedback(feedback)
