import matplotlib.pyplot as plt
import numpy as np
import rclpy
from matplotlib.animation import FuncAnimation
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class LidarPlotter(Node):
    def __init__(self):
        super().__init__("lidar_plotter")
        self.subscription = self.create_subscription(
            LaserScan, "/scan", self.listener_callback, 10
        )
        self.ranges = []
        self.angle_min = 0.0
        self.angle_increment = 0.0

        # Initialize Matplotlib Plot
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "polar"})
        self.sc = self.ax.scatter([], [], s=10, c="red")  # Initial empty scatter plot
        self.ax.set_title("Real-Time LIDAR Scan")
        self.ax.set_ylim(0, 10)  # Set max range (adjust as needed)

        # Start animation
        self.ani = FuncAnimation(self.fig, self.update_plot, interval=100)

    def listener_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def update_plot(self, frame):
        if len(self.ranges) == 0:
            return

        # Generate angles based on angle_min and angle_increment
        angles = self.angle_min + np.arange(len(self.ranges)) * self.angle_increment

        # Mask invalid data (infinite or NaN values)
        valid_mask = np.isfinite(self.ranges)
        angles = angles[valid_mask]
        ranges = self.ranges[valid_mask]

        # Update scatter plot
        self.sc.set_offsets(np.c_[angles, ranges])

    def start_plot(self):
        plt.show()


def main(args=None):
    rclpy.init(args=args)
    lidar_plotter = LidarPlotter()

    try:
        # Spin in a separate thread to keep both ROS 2 and Matplotlib alive
        from threading import Thread

        thread = Thread(target=rclpy.spin, args=(lidar_plotter,))
        thread.start()

        # Start the Matplotlib plot
        lidar_plotter.start_plot()

    except KeyboardInterrupt:
        pass
    finally:
        lidar_plotter.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
