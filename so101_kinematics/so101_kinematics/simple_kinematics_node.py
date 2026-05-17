import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import MultiThreadedExecutor
import rclpy.duration

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from scipy.spatial.transform import Rotation

from robokin.placo import PlacoKinematics, PlacoConfig
from robokin.robot_model import load_robot_description


class SimpleKinematicsNode(Node):
    def __init__(self):
        super().__init__("simple_kinematics_node")

        # Parameters
        self.declare_parameter("joints_topic", "/follower/joint_states")
        self.declare_parameter("ee_target_topic", "/go_to_ee_target")
        self.declare_parameter(
            "traj_cmd_topic", "/follower/trajectory_controller/joint_trajectory"
        )
        self.declare_parameter("joint_action_topic", "/go_to_joints")

        joints_topic = self.get_parameter("joints_topic").value
        ee_target_topic = self.get_parameter("ee_target_topic").value
        traj_cmd_topic = self.get_parameter("traj_cmd_topic").value
        joint_action_topic = self.get_parameter("joint_action_topic").value

        # Robot Model & Solver
        model = load_robot_description("so_arm101_description")
        self.solver = PlacoKinematics(
            urdf_path=str(model.urdf_path),
            ee_frame="gripper_frame_link",
            cfg=PlacoConfig(dt=0.01),  # dt doesn't matter much for static IK here
        )
        self.joint_names = self.solver.joint_names

        # Internal State
        self.q_measured = np.zeros(len(self.joint_names))
        self.has_joint_feedback = False

        # Rest configuration
        self.Q_REST = self.solver.make_configuration(
            {
                "shoulder_pan": 0.0,
                "shoulder_lift": -np.pi / 2,
                "elbow_flex": np.pi / 2,
                "wrist_flex": np.deg2rad(42.97),
                "wrist_roll": 0.0,
            }
        )

        # looking at cubes configuration
        self.Q_NOMINAL = self.solver.make_configuration(
            {
                "elbow_flex": -0.09357282806102411,
                "shoulder_lift": -0.6688156235181395,
                "shoulder_pan": -0.0260776733940559,
                "wrist_flex": 1.7303303287350031,
                "wrist_roll": -0.03681553890925539,
            }
        )

        # Callback Group for concurrent execution
        self.cb_group = ReentrantCallbackGroup()

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # Subscribers
        self.joint_sub = self.create_subscription(
            JointState,
            joints_topic,
            self.joint_state_cb,
            sensor_qos,
            callback_group=self.cb_group,
        )
        self.ee_target_sub = self.create_subscription(
            PoseStamped,
            ee_target_topic,
            self.ee_target_cb,
            sensor_qos,
            callback_group=self.cb_group,
        )

        # Publishers
        self.traj_pub = self.create_publisher(JointTrajectory, traj_cmd_topic, 10)
        self.fk_pub = self.create_publisher(PoseStamped, "/ee_pose_fk", 10)

        # Action Server
        self.action_server = ActionServer(
            self,
            FollowJointTrajectory,
            joint_action_topic,
            execute_callback=self.execute_go_to_joints,
            callback_group=self.cb_group,
        )

        # Timer for FK (100 Hz)
        self.timer = self.create_timer(
            1.0 / 100.0, self.fk_loop, callback_group=self.cb_group
        )
        self.get_logger().info("Simple Kinematics Node Started.")

    def joint_state_cb(self, msg: JointState):
        q = np.zeros(len(self.joint_names))
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]
        self.q_measured = q
        self.has_joint_feedback = True

    def fk_loop(self):
        if not self.has_joint_feedback:
            return

        T = self.solver.fk(self.q_measured)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"

        msg.pose.position.x = T[0, 3]
        msg.pose.position.y = T[1, 3]
        msg.pose.position.z = T[2, 3]

        quat = Rotation.from_matrix(T[:3, :3]).as_quat()
        msg.pose.orientation.x = quat[0]
        msg.pose.orientation.y = quat[1]
        msg.pose.orientation.z = quat[2]
        msg.pose.orientation.w = quat[3]

        self.fk_pub.publish(msg)

    def ee_target_cb(self, msg: PoseStamped):
        if not self.has_joint_feedback:  # only checks if we ever received
            self.get_logger().warn(
                "Cannot compute IK, no joint feedback yet.", throttle_duration_sec=2.0
            )
            return

        pos = msg.pose.position
        quat = msg.pose.orientation
        T_goal = np.eye(4)
        T_goal[:3, 3] = [pos.x, pos.y, pos.z]
        T_goal[:3, :3] = Rotation.from_quat(
            [quat.x, quat.y, quat.z, quat.w]
        ).as_matrix()

        q_target = self.solver.solve_goal(
            self.q_measured, T_goal, n_iters=20
        )  # TODO: change this number to allow fulfillment
        self.publish_trajectory(
            q_target, duration_sec=0.1  # TODO: hardcoded for now
        )  # 10Hz target -> 0.1s transition

    def publish_trajectory(self, q_target, joint_names=None, duration_sec=0.1):
        if joint_names is None:
            joint_names = self.joint_names

        delta_q = np.array(q_target) - self.q_measured
        required_velocities = delta_q / duration_sec

        # Max safe velocity for SO101 (rad/s)
        V_MAX = 1.5
        max_req_v = np.max(np.abs(required_velocities))

        if max_req_v > V_MAX:
            # Scale down the movement so the fastest joint moves exactly at V_MAX
            scale_factor = V_MAX / max_req_v
            delta_q = delta_q * scale_factor
            q_target = self.q_measured + delta_q  # only move at the place you can
            required_velocities = delta_q / duration_sec
            self.get_logger().warn(f"Velocity clamped! Scaled by {scale_factor:.2f}")

        traj_msg = JointTrajectory()
        traj_msg.joint_names = joint_names
        traj_msg.header.stamp = self.get_clock().now().to_msg()
        point = JointTrajectoryPoint()
        point.positions = list(q_target)

        # For a smooth transition, we set the time_from_start
        point.time_from_start = rclpy.duration.Duration(seconds=duration_sec).to_msg()

        traj_msg.points.append(point)
        self.traj_pub.publish(traj_msg)

    def get_current_q_mapped(self, requested_names):
        """Map self.q_measured to the order requested by an action."""
        current_mapped = []
        for name in requested_names:
            if name in self.joint_names:
                idx = self.joint_names.index(name)
                current_mapped.append(self.q_measured[idx])
            else:
                current_mapped.append(0.0)
        return current_mapped

    def execute_go_to_joints(self, goal_handle):
        self.get_logger().info("Executing go_to_joints action...")
        trajectory = goal_handle.request.trajectory

        if not trajectory.points:
            self.get_logger().warn("Empty trajectory received.")
            goal_handle.abort()
            return FollowJointTrajectory.Result()

        target_point = trajectory.points[-1]
        target_positions = target_point.positions
        duration_sec = (
            target_point.time_from_start.sec
            + target_point.time_from_start.nanosec * 1e-9
        )
        if duration_sec <= 0.0:
            duration_sec = 1.0  # default duration if not specified

        # Send to controller
        self.publish_trajectory(
            target_positions,
            joint_names=trajectory.joint_names,
            duration_sec=duration_sec,
        )

        feedback_msg = FollowJointTrajectory.Feedback()
        feedback_msg.joint_names = trajectory.joint_names

        start_time = time.time()

        while rclpy.ok():
            current_q = self.get_current_q_mapped(trajectory.joint_names)
            error = np.array(target_positions) - np.array(current_q)

            feedback_msg.actual.positions = current_q
            feedback_msg.desired.positions = target_positions
            feedback_msg.error.positions = error.tolist()
            goal_handle.publish_feedback(feedback_msg)

            # Check if reached
            if np.linalg.norm(error) < 0.05:
                self.get_logger().info("Target reached.")
                goal_handle.succeed()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
                return result

            # Timeout
            if time.time() - start_time > duration_sec + 2.0:
                self.get_logger().warn("Action timeout.")
                goal_handle.abort()
                result = FollowJointTrajectory.Result()
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                return result

            time.sleep(0.05)  # Loop at 20 Hz for feedback

        goal_handle.abort()
        return FollowJointTrajectory.Result()


def main(args=None):
    rclpy.init(args=args)
    node = SimpleKinematicsNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
