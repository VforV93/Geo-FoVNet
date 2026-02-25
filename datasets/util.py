"""Utility functions for dataset processing.

Provides functions for bounding box computation, centering, scaling,
and rotation transformations for B-Rep models.
"""

import pathlib

import numpy as np
import torch


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape((-1, 3))
    mask = inp[..., 6].reshape(-1)
    point_indices_inside_faces = mask == 1
    pts = pts[point_indices_inside_faces, :]
    return bounding_box_pointcloud(pts)

def bounding_box_pointcloud(pts: torch.Tensor):
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    box = [[x.min(), y.min(), z.min()], [x.max(), y.max(), z.max()]]
    return torch.tensor(box)


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    if return_center_scale:
        return inp, center, scale
    return inp

def get_random_rotation_matrix():
    """Generate a random 3D rotation matrix using axis-angle (Rodrigues' formula) - CPU version."""
    # Sample a random unit vector (axis)
    axis = torch.randn(3, dtype=torch.float32)
    axis = axis / axis.norm()
    # Sample a random rotation angle in [0, 2pi)
    theta = torch.rand(1, dtype=torch.float32) * 2 * np.pi
    # Rodrigues' rotation formula
    K = torch.tensor([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ], dtype=torch.float32)
    I = torch.eye(3, dtype=torch.float32)
    R = I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)
    return R

def rotate_face_features(ndata, rotation_matrix):
    """
    Rotate face UV-grid features for input shape (num_nodes, height, width, channels).
    Args:
        face_uv: (num_nodes, height, width, channels) tensor
        rotation_matrix: (3, 3) rotation matrix
    Returns:
        Rotated features with same shape (modifies in-place for memory efficiency)
    """
    # Rotate coordinates (channels 0-2) and normals (channels 3-5) in place
    num_nodes, height, width, channels = ndata["x"].shape
    # Reshape to (N*H*W, 3) for efficient batch matrix multiply
    coords = ndata["x"][..., 0:3].reshape(-1, 3)
    normals = ndata["x"][..., 3:6].reshape(-1, 3)
    # Apply rotation: (N*H*W, 3) @ (3, 3).T = (N*H*W, 3)
    ndata["x"][..., 0:3] = torch.matmul(coords, rotation_matrix.T).reshape(num_nodes, height, width, 3)
    ndata["x"][..., 3:6] = torch.matmul(normals, rotation_matrix.T).reshape(num_nodes, height, width, 3)

    # also rotate channels 7,8 and 9 of face_feat
    if ndata["face_feat"] is not None and ndata["face_feat"].shape[1] >= 10:
        # Shape: (num_nodes, channels)
        coords_feat = ndata["face_feat"][:, 7:10]  # (num_nodes, 3)
        ndata["face_feat"][:, 7:10] = torch.matmul(coords_feat, rotation_matrix.T)  # (num_nodes, 3)*
    return ndata["x"]
def rotate_edge_features(edata, rotation_matrix):
    """
    Rotate edge UV-grid features - optimized for minimal memory and computation.

    Args:
        x: (num_edges, 6, num_samples) tensor
        rotation_matrix: (3, 3) rotation matrix

        Returns:
        Rotated features with same shape (modifies in-place for memory efficiency)
    """
    # Shape: (num_edges, num_samples, channels)
    num_edges, num_samples, channels = edata["x"].shape
    # Process first 6 channels (two 3D vectors) efficiently
    if channels >= 6:
        # Reshape to (E*N, 3) for efficient batch matrix multiply
        v1 = edata["x"][..., 0:3].reshape(-1, 3)
        v2 = edata["x"][..., 3:6].reshape(-1, 3)
        # Apply rotation: (E*N, 3) @ (3, 3).T = (E*N, 3)
        edata["x"][..., 0:3] = torch.matmul(v1, rotation_matrix.T).reshape(num_edges, num_samples, 3)
        edata["x"][..., 3:6] = torch.matmul(v2, rotation_matrix.T).reshape(num_edges, num_samples, 3)
    return edata["x"]

def get_filenames(root_dir, filelist=None, ext=".bin"):
    """Get list of files in root_dir, optionally filtered by filelist (txt file with stems)."""
    root_dir = pathlib.Path(root_dir)
    if filelist:
        with open(str(root_dir / filelist), "r") as f:
            file_list = set(x.strip().replace("_rotated", "") for x in f.readlines())
        stem_to_path = {x.stem.replace("_rotated", ""): x for x in root_dir.rglob(f"*{ext}")}
        files = [stem_to_path[stem] for stem in file_list if stem in stem_to_path]
        return files
    return list(root_dir.rglob(f"*{ext}"))


def build_class_mapping(file_paths, split_token="_", idx=0, lowercase=True):
    """Build a mapping from class name (from file stem) to integer label."""
    class_names = sorted({
        fn.stem.split(split_token, 1)[idx].lower() if lowercase else fn.stem.split(split_token, 1)[idx] 
        for fn in file_paths
    })
    return {name: i for i, name in enumerate(class_names)}


def fractional_sample(file_paths, fraction):
    """Randomly sample a fraction of file_paths."""
    if fraction is None or fraction >= 1.0:
        return file_paths
    n = len(file_paths)
    k = max(1, int(n * fraction))
    indices = torch.randperm(n)[:k].tolist()
    return [file_paths[i] for i in indices]