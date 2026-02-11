#!/usr/bin/env python3
"""
Minimal script to visualize grasps from ROS topic.
Subscribes to /segmentation/object_pointcloud and /grasp/poses to visualize grasps in meshcat.
"""

import sys
from pathlib import Path

# Add GraspGen to Python path
GRASPGEN_ROOT = Path(__file__).resolve().parent.parent
if str(GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_ROOT))

POINTNET2_OPS = GRASPGEN_ROOT / "pointnet2_ops"
if str(POINTNET2_OPS) not in sys.path:
    sys.path.insert(0, str(POINTNET2_OPS))

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray
import struct
from scipy.spatial.transform import Rotation
import meshcat
import meshcat.geometry as g
import meshcat.transformations as mtf

from grasp_gen.grasp_server import load_grasp_cfg
from grasp_gen.utils.meshcat_utils import (
    create_visualizer,
    get_color_from_score,
    visualize_grasp,
    visualize_pointcloud,
    load_visualization_gripper_points,
    rgb2hex
)


def pointcloud2_to_numpy(msg: PointCloud2) -> tuple:
    """Convert PointCloud2 to numpy array with colors if available."""
    num_points = msg.width * msg.height
    
    if num_points == 0:
        return np.array([]).reshape(0, 3), None
    
    # Parse XYZ
    points = np.zeros((num_points, 3), dtype=np.float32)
    colors = None
    
    # Check if RGB data exists
    has_rgb = any(field.name == 'rgb' for field in msg.fields)
    if has_rgb:
        colors = np.zeros((num_points, 3), dtype=np.uint8)
    
    for i in range(num_points):
        offset = i * msg.point_step
        x = struct.unpack_from('f', msg.data, offset + 0)[0]
        y = struct.unpack_from('f', msg.data, offset + 4)[0]
        z = struct.unpack_from('f', msg.data, offset + 8)[0]
        points[i] = [x, y, z]
        
        if has_rgb:
            # RGB is typically at offset 16 as a packed uint32
            rgb = struct.unpack_from('I', msg.data, offset + 16)[0]
            colors[i] = [(rgb >> 16) & 0xFF, (rgb >> 8) & 0xFF, rgb & 0xFF]
    
    # Filter invalid points
    valid_mask = np.isfinite(points).all(axis=1)
    points = points[valid_mask]
    if colors is not None:
        colors = colors[valid_mask]
    
    return points, colors


def pose_to_matrix(pose) -> np.ndarray:
    """Convert geometry_msgs/Pose to 4x4 transformation matrix."""
    matrix = np.eye(4)
    
    # Set translation
    matrix[0, 3] = pose.position.x
    matrix[1, 3] = pose.position.y
    matrix[2, 3] = pose.position.z
    
    # Set rotation
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    rotation = Rotation.from_quat(quat).as_matrix()
    matrix[:3, :3] = rotation
    
    return matrix


def rotation_matrix_from_y(target_vec):
    """Compute rotation matrix that rotates Y axis to align with target_vec."""
    target_vec = target_vec / np.linalg.norm(target_vec)
    y_axis = np.array([0, 1, 0])
    
    # Rotation axis is cross product
    rot_axis = np.cross(y_axis, target_vec)
    rot_axis_norm = np.linalg.norm(rot_axis)
    
    if rot_axis_norm < 1e-6:
        # Parallel or anti-parallel
        if np.dot(y_axis, target_vec) > 0:
            return np.eye(3)
        else:
            # 180 degrees around X
            return mtf.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
            
    rot_angle = np.arccos(np.dot(y_axis, target_vec))
    
    # Create rotation matrix
    R_homog = mtf.rotation_matrix(rot_angle, rot_axis)
    return R_homog[:3, :3]


def visualize_thick_grasp(
    vis: meshcat.Visualizer,
    name: str,
    transform: np.ndarray,
    color: list = [255, 0, 0],
    gripper_name: str = "franka_panda",
    radius: float = 0.005,
):
    """Visualize grasp using cylinders for thickness."""
    if vis is None:
        return
        
    points_list = load_visualization_gripper_points(gripper_name)
    color_hex = rgb2hex(tuple(color))
    material = g.MeshLambertMaterial(color=color_hex)
    
    # Create a clean folder for this grasp
    vis[name].delete()
    
    for i, pts_homog in enumerate(points_list):
        pts = pts_homog[:3, :] # [3, N]
        num_pts = pts.shape[1]
        
        for j in range(num_pts - 1):
            p1 = pts[:, j]
            p2 = pts[:, j+1]
            diff = p2 - p1
            length = np.linalg.norm(diff)
            
            if length < 1e-6: continue
            
            cyl_name = f"seg_{i}_{j}"
            
            # Create cylinder
            # Cylinder is Y-aligned, centered at 0
            vis[name][cyl_name].set_object(g.Cylinder(height=length, radius=radius), material)
            
            # Compute local transform
            center = (p1 + p2) / 2.0
            direction = diff / length
            
            R = rotation_matrix_from_y(direction)
            T_cyl = np.eye(4)
            T_cyl[:3, :3] = R
            T_cyl[:3, 3] = center
            
            vis[name][cyl_name].set_transform(T_cyl)
            
    # Apply global grasp transform
    vis[name].set_transform(transform.astype(float))


class GraspVisualizer(Node):
    def __init__(self):
        super().__init__('grasp_visualizer')
        
        # Parameters
        self.declare_parameter('gripper_config', 
            '/home/booster/Workspace/GraspGen/GraspGenModels/checkpoints/graspgen_booster_gripper.yml')
        self.declare_parameter('grasp_topic', '/grasp/poses')
        
        gripper_config = self.get_parameter('gripper_config').value
        grasp_topic = self.get_parameter('grasp_topic').value
        
        # Target grasp to highlight (position and orientation)
        self.target_position = np.array([0.5645343065261841, -0.17354044318199158, 0.1595679521560669])
        self.target_orientation = np.array([-0.33687750970239905, 0.8305481785941432, 0.26329425501853976, 0.35690251016894436])
        self.position_tolerance = 0.01  # 1cm tolerance
        self.orientation_tolerance = 0.05  # quaternion distance tolerance
        
        # Load gripper config (only for gripper name, not for inference)
        self.get_logger().info(f'Loading gripper config: {gripper_config}')
        self.grasp_cfg = load_grasp_cfg(gripper_config)
        self.gripper_name = self.grasp_cfg.data.gripper_name
        
        # Create visualizer
        self.vis = create_visualizer()
        self.get_logger().info('Meshcat visualizer created')
        
        # State
        self.latest_pointcloud = None
        self.latest_grasps = None
        
        # QoS profile for latched messages (transient local)
        latched_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Subscribe to pointcloud with latched QoS
        self.pc_sub = self.create_subscription(
            PointCloud2,
            '/segmentation/object_pointcloud',
            self.pointcloud_callback,
            latched_qos
        )
        
        # Subscribe to grasp poses with latched QoS
        self.grasp_sub = self.create_subscription(
            PoseArray,
            grasp_topic,
            self.grasp_callback,
            latched_qos
        )
        
        self.get_logger().info(f'Ready! Subscribed to:')
        self.get_logger().info(f'  - /segmentation/object_pointcloud')
        self.get_logger().info(f'  - {grasp_topic}')
    
    def pointcloud_callback(self, msg: PointCloud2):
        """Store latest pointcloud."""
        self.latest_pointcloud = msg
        self.get_logger().info(f'Received pointcloud with {msg.width * msg.height} points')
        self.update_visualization()
    
    def grasp_callback(self, msg: PoseArray):
        """Store latest grasps."""
        self.latest_grasps = msg
        self.get_logger().info(f'Received {len(msg.poses)} grasp poses')
        self.update_visualization()
    
    def update_visualization(self):
        """Update meshcat visualization with latest data."""
        if self.latest_pointcloud is None or self.latest_grasps is None:
            return
        
        self.get_logger().info('Updating visualization...')
        
        # Clear previous visualization
        self.vis.delete()
        
        # Convert pointcloud
        pc, pc_colors = pointcloud2_to_numpy(self.latest_pointcloud)
        
        if len(pc) < 100:
            self.get_logger().warn(f'Too few points: {len(pc)}')
            return
        
        # Visualize pointcloud
        if pc_colors is not None:
            visualize_pointcloud(self.vis, "object_pc", pc, pc_colors, size=0.005)
        else:
            # Default orange color
            default_colors = np.tile([255, 165, 0], (len(pc), 1))
            visualize_pointcloud(self.vis, "object_pc", pc, default_colors, size=0.005)
        
        # Convert grasp poses to matrices
        num_grasps = len(self.latest_grasps.poses)
        
        # Generate colors based on grasp index (higher index = lower confidence)
        # Simulate confidence scores from 1.0 to 0.5
        simulated_conf = np.linspace(1.0, 0.5, num_grasps)
        scores = get_color_from_score(simulated_conf, use_255_scale=True)
        
        # First, always visualize the target grasp in bright magenta
        target_pose_msg = type('Pose', (), {
            'position': type('Point', (), {
                'x': self.target_position[0],
                'y': self.target_position[1],
                'z': self.target_position[2]
            })(),
            'orientation': type('Quaternion', (), {
                'x': self.target_orientation[0],
                'y': self.target_orientation[1],
                'z': self.target_orientation[2],
                'w': self.target_orientation[3]
            })()
        })()
        
        target_matrix = pose_to_matrix(target_pose_msg)
        visualize_thick_grasp(
            self.vis,
            "target_grasp",
            target_matrix,
            color=[255, 0, 255],  # Bright magenta
            gripper_name=self.gripper_name,
            radius=0.005,  # 1cm diameter 
        )
        self.get_logger().info('🎯 Target grasp visualized in MAGENTA (thick cylinders)')
        
        # Visualize all other grasps
        target_grasp_found = False
        for i, (pose, score) in enumerate(zip(self.latest_grasps.poses, scores)):
            grasp_matrix = pose_to_matrix(pose)
            
            # Check if this is the target grasp
            is_target = self._is_target_grasp(pose)
            
            if is_target:
                target_grasp_found = True
                self.get_logger().info(f'✅ Target grasp found in list at index {i} (skipping duplicate viz)')
                # Skip visualization to avoid z-fighting with the magenta target grasp
                continue
            else:
                visualize_grasp(
                    self.vis,
                    f"grasps/{i:03d}",
                    grasp_matrix,
                    color=score,
                    gripper_name=self.gripper_name,
                    linewidth=1.5,
                )
        
        if not target_grasp_found:
            self.get_logger().info('ℹ️  Target grasp not in current list (but visualized separately)')
        
        self.get_logger().info(f'Visualized {num_grasps} grasps in meshcat')
    
    def _is_target_grasp(self, pose) -> bool:
        """Check if a pose matches the target grasp within tolerance."""
        # Compare position
        pose_position = np.array([pose.position.x, pose.position.y, pose.position.z])
        position_diff = np.linalg.norm(pose_position - self.target_position)
        
        # Compare orientation (quaternion)
        pose_quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        
        # Quaternion distance (accounting for q and -q representing same rotation)
        quat_dot = np.abs(np.dot(pose_quat, self.target_orientation))
        quat_distance = 1.0 - quat_dot
        
        return position_diff < self.position_tolerance and quat_distance < self.orientation_tolerance


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspVisualizer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
