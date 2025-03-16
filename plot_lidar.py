import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rclpy
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
        self.time_stamps = []
        self.start_time = None

        # Start animation
        self.fig, self.ax = plt.subplots(subplot_kw={"projection": "polar"})

        self.ani = animation.FuncAnimation(self.fig, self.plot_polar, interval=30)

    def lidar_callback(self, msg):
        self.ranges = np.array(msg.ranges)[:-1]
        self.angle_min = msg.angle_min
        self.angle_increment = msg.angle_increment

    def plot_polar(self, frame):
        self.ax.clear()
        data = self.patch(self.ranges)

        if data.shape[0] != 1080:
            raise ValueError("Input array must have exactly 1080 elements")

        angles = np.linspace(-2.3499999046325684, 2.3499999046325684, 1080)

        self.ax.set_theta_zero_location("N")
        self.ax.set_theta_direction(1)  # Counterclockwise
        line = self.ax.plot(angles, data, c="b")

        return line

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

    def start_plot(self):
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
