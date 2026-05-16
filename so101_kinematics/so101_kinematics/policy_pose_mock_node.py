"""Mock policy node that publishes TCP pose targets.

This node mimics a Cartesian policy output by publishing geometry_msgs/PoseStamped
messages on a configurable topic. By default it publishes to /servo_target so the
existing cartesian_motion_node can consume the target directly.

The position traces a lemniscate-like figure in the Y-Z plane while X stays fixed.
Quaternion orientation is fixed and configurable through parameters.

Real-time feedback:
- Target TCP pose (reference trajectory)
- Current TCP pose (from FK of joint states)
- Pose error (target - current) — logged to console
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState

from robokin.placo import PlacoKinematics, PlacoConfig
from robokin.robot_model import load_robot_description


EE_FRAME = "gripper_frame_link"
DT = 1.0 / 50.0


def get_lemniscate_keypoint(t, a=0.2):
    """Return a lemniscate-like point in the Y-Z plane."""
    y = a * np.cos(t) / (1 + np.sin(t) ** 2)
    z = y * np.sin(t)
    return y, z


class PolicyPoseMockNode(Node):
    def __init__(self):
        super().__init__("policy_pose_mock_node")

        self.declare_parameter("target_topic", "/servo_target")
        self.declare_parameter("joints_topic", "/follower/joint_states")
        self.declare_parameter("base_frame", "follower/base_link")
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("period_sec", 8.0)
        self.declare_parameter("x_offset", 0.20)
        self.declare_parameter("y_offset", 0.0)
        self.declare_parameter("z_offset", 0.20)
        self.declare_parameter("width", 0.06)
        self.declare_parameter("qx", 0.0)
        self.declare_parameter("qy", 0.70710678)
        self.declare_parameter("qz", 0.0)
        self.declare_parameter("qw", 0.70710678)
        self.declare_parameter("log_interval_sec", 1.0)

        self._target_topic = str(self.get_parameter("target_topic").value)
        self._joints_topic = str(self.get_parameter("joints_topic").value)
        self._base_frame = str(self.get_parameter("base_frame").value)
        self._rate_hz = float(self.get_parameter("rate_hz").value)
        self._period_sec = float(self.get_parameter("period_sec").value)
        self._x_offset = float(self.get_parameter("x_offset").value)
        self._y_offset = float(self.get_parameter("y_offset").value)
        self._z_offset = float(self.get_parameter("z_offset").value)
        self._width = float(self.get_parameter("width").value)
        self._qx = float(self.get_parameter("qx").value)
        self._qy = float(self.get_parameter("qy").value)
        self._qz = float(self.get_parameter("qz").value)
        self._qw = float(self.get_parameter("qw").value)
        self._log_interval_sec = float(self.get_parameter("log_interval_sec").value)

        # Load robot model for FK
        model = load_robot_description("so_arm101_description")
        urdf_path = str(model.urdf_path)
        self.solver = PlacoKinematics(
            urdf_path=urdf_path,
            ee_frame=EE_FRAME,
            cfg=PlacoConfig(dt=DT),
        )
        self.joint_names = self.solver.joint_names

        # State tracking
        self._q_current: Optional[np.ndarray] = None
        self._T_target: Optional[np.ndarray] = None
        self._T_current: Optional[np.ndarray] = None
        self._last_log_ns = self.get_clock().now().nanoseconds

        # ROS I/O
        self._publisher = self.create_publisher(PoseStamped, self._target_topic, 10)

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            JointState, self._joints_topic, self._on_joints, sensor_qos
        )

        self._start_time = self.get_clock().now().nanoseconds

        assert self._rate_hz > 0
        timer_period = 1.0 / self._rate_hz
        self._timer = self.create_timer(timer_period, self._on_timer)

        self.get_logger().info(
            f"policy_pose_mock_node publishing to {self._target_topic} "
            f"at {self._rate_hz:.1f} Hz"
        )

    def _on_joints(self, msg: JointState):
        """Update current joint state and compute FK."""
        name_to_pos = dict(zip(msg.name, msg.position))
        q = np.zeros(len(self.joint_names))
        for i, name in enumerate(self.joint_names):
            if name in name_to_pos:
                q[i] = float(name_to_pos[name])
        self._q_current = q
        self.solver.set_joint_state(q)
        self._T_current = self.solver.current_pose().copy()
        self._log_if_due()

    def _on_timer(self):
        """Generate target pose and publish it."""
        now = self.get_clock().now().nanoseconds
        elapsed_sec = (now - self._start_time) / 1e9
        if self._period_sec > 0.0:
            phase = 2.0 * math.pi * ((elapsed_sec % self._period_sec) / self._period_sec)
        else:
            phase = 2.0 * math.pi * elapsed_sec

        y, z = get_lemniscate_keypoint(phase, a=self._width)

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._base_frame
        msg.pose.position.x = self._x_offset
        msg.pose.position.y = self._y_offset + float(y)
        msg.pose.position.z = self._z_offset + float(z)
        msg.pose.orientation.x = self._qx
        msg.pose.orientation.y = self._qy
        msg.pose.orientation.z = self._qz
        msg.pose.orientation.w = self._qw
        self._publisher.publish(msg)

        # Store target for error calculation
        from tf_transformations import quaternion_matrix
        T = np.eye(4)
        q = [self._qx, self._qy, self._qz, self._qw]
        T[:3, :3] = quaternion_matrix(q)[:3, :3]
        T[0, 3] = self._x_offset
        T[1, 3] = self._y_offset + float(y)
        T[2, 3] = self._z_offset + float(z)
        self._T_target = T

    def _log_if_due(self):
        """Periodically log pose information."""
        now = self.get_clock().now().nanoseconds
        elapsed_ns = now - self._last_log_ns
        if elapsed_ns < self._log_interval_sec * 1e9:
            return
        self._last_log_ns = now

        if self._T_target is not None:
            target_pos = self._T_target[:3, 3]
            self.get_logger().info(
                f"Target TCP:  x={target_pos[0]:.4f}  y={target_pos[1]:.4f}  z={target_pos[2]:.4f}"
            )

        if self._T_current is not None:
            current_pos = self._T_current[:3, 3]
            self.get_logger().info(
                f"Current TCP: x={current_pos[0]:.4f}  y={current_pos[1]:.4f}  z={current_pos[2]:.4f}"
            )

        if self._T_target is not None and self._T_current is not None:
            pos_error = np.linalg.norm(self._T_target[:3, 3] - self._T_current[:3, 3])
            self.get_logger().info(
                f"Position Error: {pos_error * 1000.0:.2f} mm"
            )


def main(args=None):
    rclpy.init(args=args)
    node = PolicyPoseMockNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()