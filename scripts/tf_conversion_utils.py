#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# TF Conversion Utils - Transforms object positions between coordinate frames using TF2

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs


class TFConversionUtils(Node):
    def __init__(self):
        super().__init__('tf_conversion_utils')
        
        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Subscribe to object position
        self.create_subscription(
            PointStamped, 
            '/grounded_sam/object_position', 
            self.callback, 
            10
        )
        
        self.get_logger().info('TF Conversion Utils ready!')
        self.get_logger().info('  Listening to: /grounded_sam/object_position')
    
    def callback(self, msg):
        try:
            # Replace zed_camera_link with head_point if needed
            source_frame = msg.header.frame_id
            if source_frame == 'zed_camera_link':
                source_frame = 'head_point'
            
            # Create a new PointStamped with time=0 to use latest available transform
            point_stamped = PointStamped()
            point_stamped.header.frame_id = source_frame
            point_stamped.header.stamp = rclpy.time.Time().to_msg()
            point_stamped.point = msg.point
            
            # Transform to target frame
            target_frame = 'Trunk'
            transformed_point = self.tf_buffer.transform(
                point_stamped, 
                target_frame, 
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            
            self.get_logger().info(
                f"[{source_frame} -> {target_frame}] "
                f"x={transformed_point.point.x:.3f}, "
                f"y={transformed_point.point.y:.3f}, "
                f"z={transformed_point.point.z:.3f}"
            )
            
        except Exception as e:
            self.get_logger().error(f'Transform failed: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TFConversionUtils()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
