import os
import time
import rclpy
from skills.grasp_pipeline import GripBadgePipeline
from skills.suction_gripper import SuctionGripper
from skills.peel_pipeline import PeelPipeline
from skills.sensor import ForceTorque

CONFIG_DIR = 'config'
GRIP_CONFIG_FILE = 'grip_config.json'
PEEL_CONFIG_FILE = 'peel_config.json'

config_dir = os.path.join(os.path.dirname(__file__), CONFIG_DIR)
girp_config_path = os.path.join(config_dir, GRIP_CONFIG_FILE)
peel_config_path = os.path.join(config_dir, PEEL_CONFIG_FILE)

rclpy.init()

sensor = ForceTorque()
suction_gripper = SuctionGripper()

grip_pipeline = GripBadgePipeline(girp_config_path, sensor, suction_gripper)
peel_pipeline = PeelPipeline(peel_config_path, sensor, suction_gripper)


suction_gripper.turn_on('left')
suction_gripper.turn_off('right')
time.sleep(1)

# grip_pipeline.run()
peel_pipeline.run()

grip_pipeline.destroy_node()
peel_pipeline.destroy_node()
suction_gripper.destroy_node()
sensor.shutdown()
rclpy.shutdown()
