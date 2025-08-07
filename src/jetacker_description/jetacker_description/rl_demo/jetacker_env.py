import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np

class JetackerEnv(Node):
    def __init__(self):
        super().__init__('jetacker_rl_env')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.state = None
        self.done = False

    def odom_callback(self, msg):
        # TODO: Extract position, orientation, etc.
        pass

    def scan_callback(self, msg):
        # TODO: Extract laser scan data
        pass

    def step(self, action):
        # TODO: Send velocity command, get new state, calculate reward, check done
        pass

    def reset(self):
        # TODO: Reset simulation/robot position
        pass

    def get_obs(self):
        # TODO: Return current observation
        pass