import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from .hardware_controller.relay import USBRelay
from .hardware_controller.sensor import ForceTorque
from .hardware_controller.robot import RobotController

from .skills.execute_peel import ExecutorPeel
from .skills.execute_grasp import ExecutorGrasp
from .skills.execute_workflow import ExecutorWorkflow
from .skills.execute_assembly import ExecutorAssembly
from .skills.execute_navigation import ExecutorNavigation


class MissionManager(Node):

    def __init__(self):
        super().__init__("mission_manager")

        self.declare_parameter("config_dir", "")
        self.config_dir = self.get_parameter("config_dir").value

        self.gripper = USBRelay(self)
        self.force_sensor = ForceTorque(self)
        self.robot = RobotController(self, self.force_sensor)

        # Mission Executor        
        self.executor_navigation = ExecutorNavigation(
            self,
            self.config_dir, 
            self.gripper, self.force_sensor, self.robot
        )

        self.executor_grasp = ExecutorGrasp(
            self,
            self.config_dir, 
            self.gripper, self.force_sensor, self.robot
        )

        self.executor_peel = ExecutorPeel(
            self,
            self.config_dir,
            self.gripper, self.force_sensor, self.robot
        )

        self.executor_assembly = ExecutorAssembly(
            self,
            self.config_dir, 
            self.gripper, self.force_sensor, self.robot
        )

        self.workflow_server = ExecutorWorkflow(
            node=self,
            config_dir=self.config_dir,
            executor_navigation=self.executor_navigation,
            executor_grasp=self.executor_grasp,
            executor_peel=self.executor_peel,
            executor_assembly=self.executor_assembly,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManager()
    executor = MultiThreadedExecutor(num_threads=6)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # Stop callbacks before destroying entities.  This avoids a second
        # SIGINT interrupting rclpy's entity teardown during launch shutdown.
        try:
            executor.shutdown()
        except KeyboardInterrupt:
            pass
        try:
            node.destroy_node()
        except KeyboardInterrupt:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
