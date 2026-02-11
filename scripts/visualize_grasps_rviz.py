#!/usr/bin/env python3
"""
ROS 2 Node to visualize grasps in RViz using line segments (wireframe).
Subscribes to /grasp/poses and publishes visualization_msgs/MarkerArray.
"""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from geometry_msgs.msg import PoseArray, Point, PoseStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from scipy.spatial.transform import Rotation
from pathlib import Path

# Add GraspGen to Python path
GRASPGEN_ROOT = Path(__file__).resolve().parent.parent
if str(GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_ROOT))

# from grasp_gen.grasp_server import load_grasp_cfg
import omegaconf
from grasp_gen.utils.meshcat_utils import load_visualization_gripper_points, get_color_from_score

def load_grasp_cfg(gripper_config: str):
    """
    Loads the grasp configuration file and updates the checkpoint paths to be relative to the gripper config file.
    Assumes that the checkpoint paths are in the same directory as the gripper config file.
    """
    cfg = omegaconf.OmegaConf.load(gripper_config)
    ckpt_root_dir = Path(gripper_config).parent
    cfg.eval.checkpoint = str(ckpt_root_dir / cfg.eval.checkpoint)
    cfg.discriminator.checkpoint = str(ckpt_root_dir / cfg.discriminator.checkpoint)
    # assert (
    #     cfg.data.gripper_name
    #     == cfg.diffusion.gripper_name
    #     == cfg.discriminator.gripper_name
    # )
    return cfg


class GraspVisualizerRviz(Node):
    def __init__(self):
        super().__init__('grasp_visualizer_rviz')
        
        # Parameters
        self.declare_parameter('gripper_config', 
            '/home/booster/Workspace/GraspGen/GraspGenModels/checkpoints/graspgen_booster_gripper.yml')
        self.declare_parameter('grasp_topic', '/grasp/poses')
        self.declare_parameter('marker_topic', '/grasp/markers')
        self.declare_parameter('working_grasp_topic', '/grasp/working_grasp_pose')
        self.declare_parameter('working_marker_topic', '/grasp/working_grasp_marker')
        self.declare_parameter('top_k', 5)  # Default to top 5, user can set to 1
        
        gripper_config_path = self.get_parameter('gripper_config').value
        grasp_topic = self.get_parameter('grasp_topic').value
        marker_topic = self.get_parameter('marker_topic').value
        working_grasp_topic = self.get_parameter('working_grasp_topic').value
        working_marker_topic = self.get_parameter('working_marker_topic').value
        self.top_k = self.get_parameter('top_k').value
        
        # Load gripper configuration
        self.get_logger().info(f'Loading gripper config: {gripper_config_path}')
        try:
            self.grasp_cfg = load_grasp_cfg(gripper_config_path)
            self.gripper_name = self.grasp_cfg.data.gripper_name
            self.get_logger().info(f'Gripper Name: {self.gripper_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to load gripper config: {e}')
            sys.exit(1)

        # Load gripper geometry (control points for visualization)
        # Returns list of arrays, each [4, N] (homogeneous coordinates)
        self.gripper_points = load_visualization_gripper_points(self.gripper_name)
        self.get_logger().info(f'Loaded {len(self.gripper_points)} segments for gripper visualization')

        # QoS for latched topics
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscriber
        self.grasp_sub = self.create_subscription(
            PoseArray,
            grasp_topic,
            self.grasp_callback,
            latched_qos
        )
        
        self.working_grasp_sub = self.create_subscription(
            PoseStamped,
            working_grasp_topic,
            self.working_grasp_callback,
            latched_qos
        )
        
        # Publisher
        self.marker_pub = self.create_publisher(
            MarkerArray,
            marker_topic,
            latched_qos
        )
        
        self.working_marker_pub = self.create_publisher(
            MarkerArray,
            working_marker_topic,
            latched_qos
        )
        
        self.get_logger().info(f'Subscribed to {grasp_topic} and {working_grasp_topic}')
        self.get_logger().info(f'Publishing markers to {marker_topic} (Top {self.top_k}) and {working_marker_topic}')

    def _create_gripper_marker(self, pose, ns, id, r, g, b, header):
        # Create a Marker for this grasp
        marker = Marker()
        marker.header = header
        marker.ns = ns
        marker.id = id
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        
        # Use the pose directly as the marker pose
        marker.pose = pose
        
        # Set scale (line width)
        marker.scale.x = 0.002 # 2mm thick lines
        
        marker.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
        
        # Add points
        for segment_homog in self.gripper_points:
            # segment_homog is [4, N]
            points_3d = segment_homog[:3, :].T # [N, 3]
            
            num_pts = points_3d.shape[0]
            if num_pts < 2:
                continue
                
            for j in range(num_pts - 1):
                p_start = Point(x=points_3d[j, 0], y=points_3d[j, 1], z=points_3d[j, 2])
                p_end = Point(x=points_3d[j+1, 0], y=points_3d[j+1, 1], z=points_3d[j+1, 2])
                
                marker.points.append(p_start)
                marker.points.append(p_end)
        return marker

    def working_grasp_callback(self, msg: PoseStamped):
        self.get_logger().info('Received working grasp pose')
        
        marker_array = MarkerArray()
        
        # Delete previous marker
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Create marker (Green for working grasp)
        marker = self._create_gripper_marker(
            msg.pose, "working_grasp", 0, 0.0, 1.0, 0.0, msg.header
        )
        marker_array.markers.append(marker)
        
        self.working_marker_pub.publish(marker_array)

    def grasp_callback(self, msg: PoseArray):
        num_grasps_received = len(msg.poses)
        grasps_to_visualize = min(num_grasps_received, self.top_k)
        
        self.get_logger().info(f'Received {num_grasps_received} grasps, visualizing top {grasps_to_visualize}')
        
        marker_array = MarkerArray()
        
        # Delete all previous markers
        delete_marker = Marker()
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)
        
        # Generate colors based on index (assuming sorted by score/confidence)
        # Gradient from Red (best, index 0) to Blue (worst)
        # Using 1.0 down to 0.0 for "score" effect
        simulated_scores = np.linspace(1.0, 0.0, num_grasps_received)
        colors_rgb = get_color_from_score(simulated_scores, use_255_scale=True)
        
        # Iterate only up to top_k
        for i in range(grasps_to_visualize):
            pose = msg.poses[i]
            
            # Set color
            c = colors_rgb[i]
            r, g, b = c[0]/255.0, c[1]/255.0, c[2]/255.0
            if i % 10 == 0:  # Debug log every 10th grasp
                self.get_logger().info(f'Grasp {i} Color: R={r:.2f}, G={g:.2f}, B={b:.2f}')
                
            marker = self._create_gripper_marker(
                pose, "grasps", i, r, g, b, msg.header
            )
            
            marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
        self.get_logger().info('Published markers')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = GraspVisualizerRviz()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
