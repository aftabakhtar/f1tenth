import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.animation import FuncAnimation
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarOdomPlotter(Node):
    def __init__(self):
        super().__init__("lidar_odom_plotter")

        # LIDAR Subscription
        self.subscription_lidar = self.create_subscription(
            LaserScan, "/scan", self.lidar_callback, 10
        )
        self.ranges = []
        self.angle_min = 0.0
        self.angle_increment = 0.0

        # Odom Subscription
        self.subscription_odom = self.create_subscription(
            Odometry, "/odom", self.odom_callback, 10
        )
        self.odom_x_data = []
        self.time_stamps = []
        self.start_time = None

        # Initialize Matplotlib Figure
        self.fig, axs = plt.subplots(2, 1, gridspec_kw={"height_ratios": [3, 1]})

        # LIDAR Plot (Polar)
        self.ax_lidar = self.fig.add_subplot(2, 1, 1, projection="polar")
        self.sc = self.ax_lidar.scatter([], [], s=10, c="red")
        self.ax_lidar.set_title("Real-Time LIDAR Scan")
        self.ax_lidar.set_ylim(0, 10)

        # Odometry Plot (Time-Series)
        self.ax_odom = axs[1]
        (self.odom_line,) = self.ax_odom.plot([], [], "b-")
        self.ax_odom.set_title("Odometry Velocity (linear.x)")
        self.ax_odom.set_xlabel("Time (s)")
        self.ax_odom.set_ylabel("Velocity (m/s)")
        self.ax_odom.set_xlim(0, 10)  # Time window
        self.ax_odom.set_ylim(0, 5)  # Adjust based on expected velocities

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def lidar_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def odom_callback(self, msg):
        linear_x = msg.twist.twist.linear.x
        current_time = self.get_clock().now().seconds_nanoseconds()[0]

        if self.start_time is None:
            self.start_time = current_time

        elapsed_time = current_time - self.start_time
        self.odom_x_data.append(linear_x)
        self.time_stamps.append(elapsed_time)

        # Keep only the last 100 data points (for a sliding window effect)
        if len(self.odom_x_data) > 100:
            self.odom_x_data.pop(0)
            self.time_stamps.pop(0)

    def update_plot(self, frame):
        if len(self.ranges) > 0:
            # Update LIDAR Plot
            angles = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment
            valid_mask = np.isfinite(self.ranges)
            angles, ranges = angles[valid_mask], self.ranges[valid_mask]
            self.sc.set_offsets(np.c_[angles, ranges])

        # Update Odometry Plot
        if len(self.time_stamps) > 1:
            self.odom_line.set_data(self.time_stamps, self.odom_x_data)
            self.ax_odom.set_xlim(
                max(0, self.time_stamps[-1] - 10), self.time_stamps[-1]
            )

    def start_plot(self):
        plt.tight_layout()
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    plotter = LidarOdomPlotter()

    try:
        from threading import Thread

        thread = Thread(target=rclpy.spin, args=(plotter,))
        thread.start()
        plotter.start_plot()

    except KeyboardInterrupt:
        pass
    finally:
        plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
