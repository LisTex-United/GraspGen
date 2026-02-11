#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# ROS2 GraspGen Node - Generates 6-DOF grasp poses from segmented object point clouds
# Uses Grounded-SAM object_pointcloud topic and outputs grasp poses for Robotiq gripper

import sys
from pathlib import Path

# Add GraspGen and pointnet2_ops to Python path
GRASPGEN_ROOT = Path(__file__).resolve().parent.parent
if str(GRASPGEN_ROOT) not in sys.path:
    sys.path.insert(0, str(GRASPGEN_ROOT))

# Add pointnet2_ops (compiled CUDA extension)
POINTNET2_OPS = GRASPGEN_ROOT / "pointnet2_ops"
if str(POINTNET2_OPS) not in sys.path:
    sys.path.insert(0, str(POINTNET2_OPS))

import numpy as np
import torch
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseArray, Pose, PoseStamped, TransformStamped
from std_msgs.msg import Header
import struct
from typing import Optional, Tuple
import time
from tf2_ros import Buffer, TransformListener
import tf2_py as tf2

# GraspGen imports
from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg
from grasp_gen.utils.point_cloud_utils import point_cloud_outlier_removal

# Custom Interface
try:
    from graspgen_interfaces.srv import GenerateGrasps
except ImportError:
    print("WARNING: generate_grasps interface not found. Make sure to build graspgen_interfaces.")



def pointcloud2_to_numpy(msg: PointCloud2) -> np.ndarray:
    """
    Convert sensor_msgs/PointCloud2 to Nx3 numpy array.
    Assumes XYZ float32 format.
    """
    # Get number of points
    num_points = msg.width * msg.height
    
    if num_points == 0:
        return np.array([]).reshape(0, 3)
    
    # Parse point cloud data
    points = np.zeros((num_points, 3), dtype=np.float32)
    
    for i in range(num_points):
        offset = i * msg.point_step
        x = struct.unpack_from('f', msg.data, offset + 0)[0]
        y = struct.unpack_from('f', msg.data, offset + 4)[0]
        z = struct.unpack_from('f', msg.data, offset + 8)[0]
        points[i] = [x, y, z]
    
    # Filter out invalid points (NaN, Inf)
    valid_mask = np.isfinite(points).all(axis=1)
    
    return points[valid_mask]


def matrix_to_pose(matrix: np.ndarray) -> Pose:
    """Convert 4x4 transformation matrix to geometry_msgs/Pose."""
    from scipy.spatial.transform import Rotation
    
    pose = Pose()
    
    # Extract translation
    pose.position.x = float(matrix[0, 3])
    pose.position.y = float(matrix[1, 3])
    pose.position.z = float(matrix[2, 3])
    
    # Extract rotation and convert to quaternion
    rotation_matrix = matrix[:3, :3]
    r = Rotation.from_matrix(rotation_matrix)
    quat = r.as_quat()  # Returns [x, y, z, w]
    
    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])
    
    return pose


def transform_point_cloud(points: np.ndarray, transform: TransformStamped) -> np.ndarray:
    """
    Transform Nx3 point cloud using TF2 transform.
    
    Args:
        points: Nx3 numpy array of points
        transform: geometry_msgs/TransformStamped
    
    Returns:
        Nx3 transformed point cloud
    """
    # Extract translation and rotation from transform
    translation = np.array([
        transform.transform.translation.x,
        transform.transform.translation.y,
        transform.transform.translation.z
    ])
    
    # Convert quaternion to rotation matrix
    from scipy.spatial.transform import Rotation
    quat = [
        transform.transform.rotation.x,
        transform.transform.rotation.y,
        transform.transform.rotation.z,
        transform.transform.rotation.w
    ]
    rotation = Rotation.from_quat(quat).as_matrix()
    
    # Apply transformation: R * p + t
    transformed_points = (rotation @ points.T).T + translation
    
    return transformed_points


class GraspGenNode(Node):
    def __init__(self):
        super().__init__('graspgen_node')
        
        # Declare parameters
        self.declare_parameter('gripper_config', 
            '/home/booster/Workspace/GraspGen/GraspGenModels/checkpoints/graspgen_booster_gripper.yml')
        self.declare_parameter('grasp_threshold', 0.7)
        self.declare_parameter('num_grasps', 500)
        self.declare_parameter('topk_num_grasps', 10)
        self.declare_parameter('min_points', 100)
        self.declare_parameter('remove_outliers', True)
        
        # Get parameters
        gripper_config = self.get_parameter('gripper_config').value
        self.grasp_threshold = self.get_parameter('grasp_threshold').value
        self.num_grasps = self.get_parameter('num_grasps').value
        self.topk_num_grasps = self.get_parameter('topk_num_grasps').value
        self.min_points = self.get_parameter('min_points').value
        self.remove_outliers = self.get_parameter('remove_outliers').value
        
        self.get_logger().info(f'Loading GraspGen with config: {gripper_config}')
        
        # Initialize TF2 for coordinate transformations
        self.tf_buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=60.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Initialize GraspGen
        try:
            self.grasp_cfg = load_grasp_cfg(gripper_config)
            self.grasp_sampler = GraspGenSampler(self.grasp_cfg)
            self.gripper_name = self.grasp_cfg.data.gripper_name
            self.get_logger().info(f'GraspGen initialized with gripper: {self.gripper_name}')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize GraspGen: {e}')
            raise
        
        # QoS profile for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # Service
        self.srv = self.create_service(
            GenerateGrasps,
            '/graspgen/generate_grasps',
            self.generate_grasps_callback
        )
        
        # Publishers (still useful for visualization)
        self.grasp_poses_pub = self.create_publisher(
            PoseArray,
            '/graspgen/grasp_poses',
            10
        )
        
        self.best_grasp_pub = self.create_publisher(
            PoseStamped,
            '/graspgen/best_grasp',
            10
        )
        
        # State
        self.scene_pc: Optional[np.ndarray] = None
        self.processing = False
        
        self.get_logger().info('GraspGen ROS2 node ready!')
        self.get_logger().info('  Service: /graspgen/generate_grasps')
        self.get_logger().info(f'  Grasp threshold: {self.grasp_threshold}')
        self.get_logger().info(f'  Top-K grasps: {self.topk_num_grasps}')
    
    def scene_pointcloud_callback(self, msg: PointCloud2):
        """Store scene point cloud for potential collision filtering."""
        # Only update occasionally to save CPU
        if time.time() - self.last_process_time > 1.0:
            self.scene_pc = pointcloud2_to_numpy(msg)
            self.last_process_time = time.time() # Update usage
    
    def generate_grasps_callback(self, request, response):
        """Process service request."""
        self.get_logger().info('Received grasp generation request!')
        
        msg = request.object_pointcloud
        
        try:
            # Convert PointCloud2 to numpy
            object_pc = pointcloud2_to_numpy(msg)
            
            if len(object_pc) < self.min_points:
                msg = f'Object point cloud too small: {len(object_pc)} points'
                self.get_logger().warn(msg)
                response.success = False
                response.message = msg
                return response

            # # Transform to Target Frame
            # source_frame = msg.header.frame_id
            # if source_frame == 'zed_camera_link': source_frame = 'head_point'
            
            # try:
            #     transform = self.tf_buffer.lookup_transform(
            #         self.target_frame,
            #         source_frame,
            #         rclpy.time.Time(seconds=0),
            #         timeout=rclpy.duration.Duration(seconds=1.0)
            #     )
            #     object_pc = transform_point_cloud(object_pc, transform)
            # except Exception as e:
            #     err_msg = f'Transform failed: {e}'
            #     self.get_logger().error(err_msg)
            #     response.success = False
            #     response.message = err_msg
            #     return response
            # Input is already in the correct frame - no transformation needed
            input_frame = msg.header.frame_id
            
            # Debug: Log input point cloud statistics
            self.get_logger().info(f"Input point cloud: {len(object_pc)} points in frame '{input_frame}'")
            self.get_logger().info(f"Point cloud X range: [{object_pc[:, 0].min():.3f}, {object_pc[:, 0].max():.3f}]")
            self.get_logger().info(f"Point cloud Y range: [{object_pc[:, 1].min():.3f}, {object_pc[:, 1].max():.3f}]")
            self.get_logger().info(f"Point cloud Z range: [{object_pc[:, 2].min():.3f}, {object_pc[:, 2].max():.3f}]")
            self.get_logger().info(f"Point cloud centroid: ({object_pc[:, 0].mean():.3f}, {object_pc[:, 1].mean():.3f}, {object_pc[:, 2].mean():.3f})")
            
            # Inference (Get ALL grasps first)
            t_start = time.time()
            grasps, grasp_conf = GraspGenSampler.run_inference(
                object_pc,
                self.grasp_sampler,
                grasp_threshold=self.grasp_threshold,
                num_grasps=self.num_grasps,
                topk_num_grasps=self.num_grasps,
                remove_outliers=self.remove_outliers,
            )
            
            if len(grasps) == 0:
                response.success = False
                response.message = "No grasps found above threshold"
                return response

            # Post-Process & Filter (Filter 2: Top-Down)
            grasps_np = grasps.cpu().numpy()
            grasp_conf_np = grasp_conf.cpu().numpy()
            grasps_np[:, 3, 3] = 1.0
            
            # Debug: Log grasp matrix structure
            self.get_logger().info(f"First grasp matrix shape: {grasps_np[0].shape}")
            self.get_logger().info(f"First grasp matrix:\n{grasps_np[0]}")
            
            approach_vectors_z = grasps_np[:, 2, 2]
            self.get_logger().info(f"Approach Z range: min={approach_vectors_z.min():.3f}, max={approach_vectors_z.max():.3f}")
            self.get_logger().info(f"Approach Z values (first 10): {approach_vectors_z[:10]}")
            
            # Also check z positions of grasps
            grasp_z_positions = grasps_np[:, 2, 3]
            self.get_logger().info(f"Grasp Z position range: min={grasp_z_positions.min():.3f}, max={grasp_z_positions.max():.3f}")
            
            top_down_mask = approach_vectors_z < -0.7
            self.get_logger().info(f"Top-down grasps (approach_z < -0.7): {np.sum(top_down_mask)}/{len(approach_vectors_z)}")
            
            if np.sum(top_down_mask) == 0:
                 response.success = False
                 response.message = "No top-down grasps found"
                 return response
                 
            grasps_np = grasps_np[top_down_mask]
            grasp_conf_np = grasp_conf_np[top_down_mask]
            
            # Sort by confidence (descending)
            conf_flat = grasp_conf_np.flatten()
            sorted_indices = np.argsort(conf_flat)[::-1]
            grasps_np = grasps_np[sorted_indices]
            grasp_conf_np = grasp_conf_np[sorted_indices]
                
            # Construct Response
            pose_array = PoseArray()
            pose_array.header = msg.header
            pose_array.header.frame_id = input_frame
            pose_array.header.stamp = self.get_clock().now().to_msg()
            
            for i, grasp_mat in enumerate(grasps_np):
                pose_array.poses.append(matrix_to_pose(grasp_mat))
            
            # Fill Response
            response.grasp_poses = pose_array
            
            if len(pose_array.poses) > 0:
                response.best_grasp.header = pose_array.header
                response.best_grasp.pose = pose_array.poses[0]
            
            response.success = True
            response.message = f"Found {len(grasps_np)} grasps"
            self.get_logger().info(f"Published {len(grasps_np)} top-down grasps (filtered from {len(grasps)} candidates)")
            
            # Also publish for viz
            self.grasp_poses_pub.publish(pose_array)
            self.best_grasp_pub.publish(response.best_grasp)
            
            return response
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback
            self.get_logger().error(traceback.format_exc())
            response.success = False
            response.message = str(e)
            return response
    



def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = GraspGenNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
