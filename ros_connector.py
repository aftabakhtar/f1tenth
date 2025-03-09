import os
import signal

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

        self.timer = self.create_timer(0.01, self.send_actions)

        # Subscribers
        self.lidar_subscription = self.create_subscription(
            LaserScan, lidar_scan_topic, self.lidar_callback, 10
        )
        self.odom_subscription = self.create_subscription(
            Odometry, odometry_topic, self.odom_callback, 10
        )

        # Store latest sensor data
        self.latest_lidar_data = None
        self.latest_odom_data = None

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
        self.latest_lidar_data = msg  # Store latest LIDAR data
        # self.get_logger().info(f"Received Lidar Data: {len(msg.ranges)} ranges")

    def odom_callback(self, msg: Odometry):
        self.latest_odom_data = msg  # Store latest Odometry data
        position = msg.pose.pose.position
        # self.get_logger().info(f"Twist Data: ({self.latest_odom_data.twist})")

    def get_lidar_data(self):
        """Returns the latest LIDAR data received."""
        return self.latest_lidar_data

    def get_odometry_data(self):
        """Returns the latest odometry data received."""
        return self.latest_odom_data

    # stable version -- works fine at moderate speed (no weighted averages)
    def send_drive_command(self, speed: float, steering_angle: float):
        speed = 3.0 * speed
        steering_angle = 0.5 * steering_angle

        if speed < 1.0:
            self.threshold_counter += 1
        else:
            self.threshold_counter = 0
            self.boost = 100

        if self.threshold_counter >= 10:
            if self.boost - 10 >= 0:
                speed = 2.0
                self.boost -= 10

        if self.threshold_counter >= 10 and self.boost == 0:
            self.threshold_counter = 0
            self.boost = 100

        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={speed}, Steering Angle={steering_angle}"
        )

    # drifty controller with weighted averages
    def send_drive_command_2(self, speed: float, steering_angle: float):
        speed = 3.0 * speed
        steering_angle = 0.5 * steering_angle

        if speed < 1.0:
            self.threshold_counter += 1
        else:
            self.threshold_counter = 0
            self.boost = 100

        if self.threshold_counter >= 10:
            if self.boost - 10 >= 0:
                speed = 2.0
                self.boost -= 10

        if self.threshold_counter >= 10 and self.boost == 0:
            self.threshold_counter = 0
            self.boost = 100

        # alpha = 0.8
        self.speed = (1 - self.alpha) * self.speed + self.alpha * speed
        self.steering = (1 - self.alpha) * self.steering + self.alpha * steering_angle

        msg = AckermannDriveStamped()
        msg.drive.speed = self.speed
        # msg.drive.steering_angle = steering_angle
        msg.drive.steering_angle = self.steering
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={speed}, Steering Angle={steering_angle}"
        )

    # insane controller - stable controller at very high speed
    def send_drive_command_3(self, speed: float, steering_angle: float):
        speed = 4.0 * speed
        steering_angle = 0.7 * steering_angle

        if speed < 1.0:
            self.threshold_counter += 1
        else:
            self.threshold_counter = 0
            self.boost = 100

        if self.threshold_counter >= 10:
            if self.boost - 10 >= 0:
                speed = 2.0
                self.boost -= 10

        if self.threshold_counter >= 10 and self.boost == 0:
            self.threshold_counter = 0
            self.boost = 100

        msg = AckermannDriveStamped()
        msg.drive.speed = speed
        msg.drive.steering_angle = steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={speed}, Steering Angle={steering_angle}"
        )

    def form_data(self):
        lidar_data = self.get_lidar_data()
        odom_data = self.get_odometry_data()
        if lidar_data and odom_data:
            self.data["scan"] = np.array(lidar_data.ranges)
            # self.data["scan"] = np.array(lidar_data.ranges[:-1])
            self.data["linear_vel_x"] = odom_data.twist.twist.linear.x
            self.data["linear_vel_y"] = odom_data.twist.twist.linear.y
            self.data["angular_vel_z"] = odom_data.twist.twist.angular.z

    def send_actions(self):
        self.form_data()

        if not len(self.data.keys()):
            return
        self.get_logger().info(f"{self.data}")

        if self.data["scan"] is None:
            return
        if self.data["linear_vel_x"] is None:
            return
        if self.data["linear_vel_y"] is None:
            return
        if self.data["angular_vel_z"] is None:
            return

        # self.get_logger().info(f"{self.data}")
        action, _ = self.model.predict(self.data, deterministic=True)

        self.get_logger().info(f"{action}")
        self.send_drive_command(float(action[1]), float(action[0]))

    def shutdown_gracefully(self, signum, frame):
        self.send_drive_command(0.0, 0.0)
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
    node = F1TenthROSNode(lidar_scan_topic, odometry_topic, drive_topic, "v0.2")

    try:
        rclpy.spin(node)  # Keep node running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

