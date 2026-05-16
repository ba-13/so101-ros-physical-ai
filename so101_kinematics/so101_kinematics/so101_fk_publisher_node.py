import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
import numpy as np
from scipy.spatial.transform import Rotation
import yourdfpy

from robokin.placo import PlacoKinematics, PlacoConfig
from robokin.robot_model import load_robot_description

EE_FRAME = "gripper_frame_link"
DT = 1.0 / 50.0


class FKSO101Node(Node):
    def __init__(self):
        super().__init__("so101_fk_publisher_node")

        self.declare_parameter("joints_topic", "/follower/joint_states")
        self.declare_parameter("pose_topic", "/follower/end_effector_pose")

        joints_topic = self.get_parameter("joints_topic").value
        pose_topic = self.get_parameter("pose_topic").value

        # Load robot model
        model = load_robot_description("so_arm101_description")
        urdf_path = str(model.urdf_path)
        urdf = yourdfpy.URDF.load(urdf_path)

        # Placo solver for FK
        self.solver = PlacoKinematics(
            urdf_path=urdf_path,
            ee_frame=EE_FRAME,
            cfg=PlacoConfig(dt=DT),
        )
        self.joint_names = self.solver.joint_names

        # Subscribers and Publishers
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.joint_sub = self.create_subscription(
            JointState, joints_topic, self._joint_state_cb, sensor_qos
        )

        self.pose_pub = self.create_publisher(PoseStamped, pose_topic, 10)
        self.get_logger().info(
            f"FK Publisher node started. Listening to '{joints_topic}' and publishing to '{pose_topic}'"
        )

    def _joint_state_cb(self, msg: JointState):
        q = np.zeros(len(self.joint_names))
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                q[i] = msg.position[idx]

        # Calculate FK
        T = self.solver.fk(q)

        # Extract translation and rotation
        translation = T[:3, 3]
        rotation = Rotation.from_matrix(T[:3, :3]).as_euler(seq="xyz")  # HACK

        # Create message
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        # Ensure to check what the base link frame id is typically called in this project
        pose_msg.header.frame_id = "base_link"

        pose_msg.pose.position.x = float(translation[0])
        pose_msg.pose.position.y = float(translation[1])
        pose_msg.pose.position.z = float(translation[2])

        pose_msg.pose.orientation.x = float(rotation[0])
        pose_msg.pose.orientation.y = float(rotation[1])
        pose_msg.pose.orientation.z = float(rotation[2])
        pose_msg.pose.orientation.w = float(0)

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = FKSO101Node()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
