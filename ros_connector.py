import os
import signal
from collections import deque
from time import time

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO


class F1TenthROSNode(Node):
    def __init__(
        self, lidar_scan_topic: str, odometry_topic: str, drive_topic: str, version: str
    ):
        super().__init__("f2tenth_ros_node")

        self.data = {}
        self.threshold_counter = 0
        self.boost = 100
        self.steering = 0
        self.speed = 0
        self.alpha = 0.3

        # Publisher
        self.drive_publisher = self.create_publisher(
            AckermannDriveStamped, drive_topic, 10
        )

        self.timer = self.create_timer(0.1, self.send_actions)

        # Subscribers
        self.lidar_subscription = self.create_subscription(
            LaserScan, lidar_scan_topic, self.lidar_callback, 10
        )

        # Store latest sensor data
        self.lidar_data = deque(maxlen=9)
        self.time_data = deque(maxlen=9)

        signal.signal(signal.SIGINT, self.shutdown_gracefully)

        # policy_kwargs = {
        #     "features_extractor_class": CustomCombinedExtractor,
        #     "features_extractor_kwargs": {"features_dim": 256},
        #     "net_arch": dict(pi=[128, 64], vf=[128, 64]),
        # }

        directory = f"weights/{version}"
        weights = [
            f
            for f in os.listdir(directory)
            if os.path.isfile(os.path.join(directory, f))
        ]

        weight = f"{directory}/{weights[0]}"
        self.get_logger().info(f"Loading weight: {weight}")
        self.model = PPO.load(weight)

    def lidar_callback(self, msg: LaserScan):
        self.lidar_data.append(np.array(msg.ranges[:-1]))  # Store latest LIDAR data
        self.time_data.append(time())
        # self.get_logger().info(f"Received Lidar Data: {len(msg.ranges)} ranges")

    # stable version -- works fine at moderate speed (no weighted averages)
    def send_drive_command(self, speed: float, steering_angle: float):
        scaled_speed = self._map_value(speed, (0.0, 1.0), (0.7, 3.0))
        scaled_steering_angles = self._map_value(
            steering_angle, (0.0, 1.0), (-0.4189, 0.4189)
        )

        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angles
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angles}"
        )

    def form_data(self):
        if not len(self.lidar_data):
            return

        current_lidar_data = self.lidar_data[-1] / 30.0
        previous_lidar_data = self.lidar_data[0] / 30.0

        self.data["scan"] = np.array(current_lidar_data)
        self.data["previous_scan"] = np.array(previous_lidar_data)

        self.get_logger().info(f"Time Data={self.time_data[-1] - self.time_data[0]}")

    def _map_value(self, value, current_range, desired_range):
        cur_min, cur_max = current_range
        des_min, des_max = desired_range
        return des_min + (value - cur_min) * (des_max - des_min) / (cur_max - cur_min)

    def send_actions(self):
        self.form_data()

        if not len(self.data.keys()):
            return
        self.get_logger().info(f"{self.data}")

        if self.data["scan"] is None:
            return
        if self.data["previous_scan"] is None:
            return

        # self.get_logger().info(f"{self.data}")
        action, _ = self.model.predict(self.data, deterministic=True)

        self.get_logger().info(f"{action}")
        self.send_drive_command(float(action[1]), float(action[0]))

    def shutdown_gracefully(self, signum, frame):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_publisher.publish(msg)
        self.get_logger().info("KeyboardInterrupt received! Stopping node execution...")
        raise KeyboardInterrupt


def main():
    rclpy.init()

    # Define topics
    lidar_scan_topic = "/scan"
    # odometry_topic = "/ego_racecar/odom"
    odometry_topic = "/odom"
    # drive_topic = "/drive"
    drive_topic = "/ackermann_cmd"

    # Create an instance of the node
    node = F1TenthROSNode(lidar_scan_topic, odometry_topic, drive_topic, "lo_1")

    try:
        rclpy.spin(node)  # Keep node running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
