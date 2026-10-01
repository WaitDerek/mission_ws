"""Keep ROS Action integration tests away from the hardware Mission graph.

Loaded before test modules import rclpy. Never reuse the sourced hardware
domain or its CycloneDDS network-interface configuration for these tests.
"""
import os


os.environ["ROS_DOMAIN_ID"] = "197"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
os.environ.pop("CYCLONEDDS_URI", None)
