import csv
import os
import signal
import time
from typing import List

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

        self.csv_data = []

        self.odom_turning = 0.0

        # for calculaating scan f
        self.times = []
        self.last_lidar_scan = np.array([])

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
    # TODO: try with v0.1
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
        speed = 4.0 * speed
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

    # hard minimum value controller
    # TODO: works great with v0.6
    def send_drive_command_4(self, speed: float, steering_angle: float):
        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (0.7, 2.0))

        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}"
        )

    # hard minimum value controller
    def send_drive_command_5(self, speed: float, steering_angle: float):
        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (0.7, 1.5))

        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}"
        )

    # hard minimum with custom distance from nearest obstacle avoidance
    # TODO: works well with v0.10
    def send_drive_command_6(
        self, speed: float, steering_angle: float, scan: List[float]
    ):
        max_speed = 2.0
        min_speed = 0.7
        closest_obstacle = min(scan)

        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (min_speed, max_speed))

        self.get_logger().info(f"Minimum distance={closest_obstacle}")

        threshold_min = 0.3
        threshold_start = 0.7

        if closest_obstacle <= threshold_min:
            scaled_speed = min_speed

        elif closest_obstacle >= threshold_start:
            scaled_speed = max_speed

        else:
            scale = (closest_obstacle - threshold_min) / (
                threshold_start - threshold_min
            )
            scaled_speed = float(min_speed + scale * (max_speed - min_speed))

        # if scaled_steering_angle > 0:
        #     scaled_steering_angle = 0.4189
        # else:
        #     scaled_steering_angle = -0.4189

        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}"
        )

    # best so far try with v0.11
    # TODO: works well with v0.11
    def send_drive_command_7(
        self, speed: float, steering_angle: float, scan: List[float], odom_speed: float
    ):

        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (0.7, 3.5))

        # self.csv_data.append(
        #     (time.time(), scaled_speed, scaled_steering_angle, list(scan), odom_speed)
        # )

        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}"
        )

    # same as above but with ttc
    # TODO: works well with v0.11 + ttc
    def send_drive_command_8(
        self,
        speed: float,
        steering_angle: float,
        scan: List[float],
        odom_speed: float,
        odom_yaw_rate: float,
    ):
        closest_obstacle = min(scan[500:1300])
        min_speed = 0.7
        max_speed = 6.0

        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (min_speed, max_speed))

        ttc, ttc_scaled_speed = self._calculate_ttc(
            odom_speed,
            closest_obstacle,
            0.1,
            0.01,
            min_speed,
            max_speed,
            odom_yaw_rate,
        )

        scaled_speed = min(ttc_scaled_speed, scaled_speed)

        msg = AckermannDriveStamped()
        msg.drive.speed = float(scaled_speed)
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}, TTC Speed={ttc_scaled_speed}"
        )

    # best so far try with v0.12
    # TODO: works well with v0.12
    def send_drive_command_9(self, speed: float, steering_angle: float):

        scaled_steering_angle = self._map_value(
            steering_angle, (0.0, +1.0), (-0.4189, 0.4189)
        )
        scaled_speed = self._map_value(speed, (0.0, +1.0), (0.7, 4.0))
        msg = AckermannDriveStamped()
        msg.drive.speed = scaled_speed
        msg.drive.steering_angle = scaled_steering_angle
        self.drive_publisher.publish(msg)
        self.get_logger().info(
            f"Sent drive command: Speed={scaled_speed}, Steering Angle={scaled_steering_angle}"
        )

    def form_data(self):
        lidar_data = self.get_lidar_data()
        odom_data = self.get_odometry_data()
        if lidar_data and odom_data:
            patched_lidar_ranges = self.patch(lidar_data.ranges)
            self.data["scan"] = patched_lidar_ranges / 30.0
            # self.data["scan"] = np.array(lidar_data.ranges[:-1])
            self.data["linear_vel_x"] = odom_data.twist.twist.linear.x / 5.0
            # self.data["linear_vel_y"] = odom_data.twist.twist.linear.y
            # self.data["angular_vel_z"] = odom_data.twist.twist.angular.z
            self.odom_turning = odom_data.twist.twist.angular.z

        if (
            lidar_data
            and "scan" in self.data
            and not np.array_equal(self.data["scan"], self.last_lidar_scan)
        ):
            self.last_lidar_scan = self.data["scan"]
            self.times.append(time.time())

    def send_actions(self):
        self.form_data()

        if not len(self.data.keys()):
            return
        self.get_logger().info(f"{self.data}")

        if self.data["scan"] is None:
            return
        if self.data["linear_vel_x"] is None:
            return
        # if self.data["linear_vel_y"] is None:
        #     return
        # if self.data["angular_vel_z"] is None:
        #     return

        # self.get_logger().info(f"{self.data}")
        action, _ = self.model.predict(self.data, deterministic=True)

        self.get_logger().info(f"{action}")
        self.send_drive_command_7(
            float(action[1]),
            float(action[0]),
            self.data["scan"],
            self.data["linear_vel_x"],
        )
        # self.send_drive_command_8(
        #     float(action[1]),
        #     float(action[0]),
        #     self.data["scan"],
        #     self.data["linear_vel_x"],
        #     self.odom_turning,
        # )

    def shutdown_gracefully(self, signum, frame):
        # with open("data.csv", "w", newline="") as file:
        #     writer = csv.writer(file)
        #     writer.writerow(
        #         ["time", "scaled_speed", "scaled_angles", "scan", "odom_speed"]
        #     )
        #     writer.writerows(self.csv_data)

        average_time = 0.0
        for i in range(2, len(self.times)):
            average_time += self.times[i] - self.times[i - 1]

        self.get_logger().info(f"Average time={average_time / (len(self.times) - 2)}")

        self.send_drive_command(0.0, 0.0)
        self.get_logger().info("KeyboardInterrupt received! Stopping node execution...")
        raise KeyboardInterrupt

    def _map_value(self, value, current_range, desired_range):
        cur_min, cur_max = current_range
        des_min, des_max = desired_range
        return des_min + (value - cur_min) * (des_max - des_min) / (cur_max - cur_min)

    def patch(self, data, t=30):
        patched = np.zeros_like(data)

        started_patch = None

        for i, n in enumerate(data):
            if started_patch is None:
                if n < t:
                    patched[i] = n
                else:
                    started_patch = i
            else:
                if n < t:
                    patched[started_patch : i + 1] = (data[started_patch - 1] + n) / 2
                    started_patch = None
                else:
                    pass

        if started_patch is not None:
            patched[started_patch:] = patched[started_patch - 1]

        return patched

    def _calculate_ttc(
        self,
        odom_speed: float,
        closest_obstacle: float,
        safe_ttc: float,
        emergency_ttc: float,
        min_speed: float,
        max_speed: float,
        yaw_rate: float,
    ):
        yaw_factor = 1.0 + abs(yaw_rate) / 2.5  # Scale factor based on turn speed

        adjusted_safe_ttc = safe_ttc * yaw_factor
        adjusted_emergency_ttc = emergency_ttc * yaw_factor

        ttc = (closest_obstacle * 30.0) / (odom_speed * 5.0 + 1e-6)

        self.get_logger().info(
            f"TTC={ttc}, Closest Obstacle={closest_obstacle * 30.0}, Odom Speed={odom_speed * 5.0}, Odom Turning={yaw_rate}, Adjusted Emergency TTC={adjusted_emergency_ttc}, Adjusted Safe TTC={adjusted_safe_ttc}"
        )

        if ttc < adjusted_emergency_ttc:
            return ttc, min_speed
        elif ttc < adjusted_safe_ttc:
            scale = (ttc - adjusted_emergency_ttc) / (
                adjusted_safe_ttc - adjusted_emergency_ttc
            )
            return ttc, min_speed + scale * (max_speed - min_speed)
        else:
            return ttc, max_speed


def main():
    rclpy.init()

    # Define topics
    lidar_scan_topic = "/scan"
    # odometry_topic = "/ego_racecar/odom"
    odometry_topic = "/odom"
    # drive_topic = "/drive"
    drive_topic = "/ackermann_cmd"

    # Create an instance of the node
    node = F1TenthROSNode(lidar_scan_topic, odometry_topic, drive_topic, "v0.11")

    try:
        rclpy.spin(node)  # Keep node running
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
