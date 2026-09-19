import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor

from .hardware_controller.relay import USBRelay
from .hardware_controller.sensor import ForceTorque
from .hardware_controller.robot import RobotController

from .skills.execute_grasp import ExecuteGraspServer
from .skills.execute_peel import ExecutePeelServer
from .skills.execute_assembly import ExecuteAssemblyServer
from .skills.execute_workflow import ExecuteWorkflowServer


class MissionManager(Node):

    def __init__(self):
        super().__init__("mission_manager")

        self.declare_parameter("config_dir", "")
        self.config_dir = self.get_parameter("config_dir").value

        self.gripper = USBRelay(self)
        self.force_sensor = ForceTorque(self)
        self.robot = RobotController(self, self.force_sensor)

        # Mission Server
        self.grasp_server = ExecuteGraspServer(
            self,
            self.config_dir, 
            self.gripper, self.force_sensor, self.robot
        )

        self.peel_server = ExecutePeelServer(
            self,
            self.config_dir,
            self.gripper, self.force_sensor, self.robot
        )

        self.assembly_server = ExecuteAssemblyServer(
            self,
            self.config_dir, 
            self.gripper, self.force_sensor, self.robot
        )

        self.workflow_server = ExecuteWorkflowServer(
            self,
            self.config_dir,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionManager()
    executor = MultiThreadedExecutor(num_threads=5)
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
