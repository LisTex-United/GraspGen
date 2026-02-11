import torch
import numpy as np
import trimesh
import os
from grasp_gen.robot import load_control_points_core, load_default_gripper_config
from pathlib import Path


class GripperModel(object):
    """Custom gripper model - uses a simple box mesh for collision."""
    def __init__(self, data_root_dir=None):
        # Create a simple box mesh representing the gripper
        # Width: 8.5cm, Height: 2cm, Depth: 6cm
        self.mesh = trimesh.creation.box(extents=[0.085, 0.02, 0.06])
        # Translate so base is at origin
        self.mesh.apply_translation([0, 0, 0.03])

    def get_gripper_collision_mesh(self):
        return self.mesh

    def get_gripper_visual_mesh(self):
        return self.mesh


def get_gripper_offset_bins():
    """Offset bins for the custom gripper with 8.5cm width."""
    # Generate 11 bins from 0 to half-width (0.0425)
    half_width = 0.0425
    offset_bins = np.linspace(0, half_width, 11).tolist()
    
    # Uniform weights
    offset_bin_weights = [1.0] * 10
    return offset_bins, offset_bin_weights


def load_control_points() -> torch.Tensor:
    """
    Load the control points for the gripper, used for training.
    Returns a tensor of shape (4, N) where N is the number of control points.
    """
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    control_points = np.vstack([control_points, np.zeros(3)])
    control_points = np.hstack([control_points, np.ones([len(control_points), 1])])
    control_points = torch.from_numpy(control_points).float()
    return control_points.T


def load_control_points_for_visualization():
    gripper_config = load_default_gripper_config(Path(__file__).stem)
    control_points = load_control_points_core(gripper_config)
    
    mid_point = (control_points[0] + control_points[1]) / 2
    
    control_points = [
        control_points[-2], control_points[0], mid_point,
        [0, 0, 0], mid_point, control_points[1], control_points[-1]
    ]
    return [control_points, ]
