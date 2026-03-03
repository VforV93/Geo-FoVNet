"""Utility functions: bounding-box, centering/scaling, rotation, file listing."""

import pathlib
import numpy as np
import torch


def bounding_box_uvgrid(inp: torch.Tensor):
    pts = inp[..., :3].reshape(-1, 3)
    mask = inp[..., 6].reshape(-1) == 1
    return bounding_box_pointcloud(pts[mask])


def bounding_box_pointcloud(pts: torch.Tensor):
    return torch.stack([pts.min(0).values, pts.max(0).values])


def center_and_scale_uvgrid(inp: torch.Tensor, return_center_scale=False):
    bbox = bounding_box_uvgrid(inp)
    diag = bbox[1] - bbox[0]
    scale = 2.0 / max(diag[0], diag[1], diag[2])
    center = 0.5 * (bbox[0] + bbox[1])
    inp[..., :3] -= center
    inp[..., :3] *= scale
    return (inp, center, scale) if return_center_scale else inp


def get_random_rotation_matrix():
    """Random 3D rotation via axis-angle (Rodrigues)."""
    axis = torch.randn(3, dtype=torch.float32)
    axis /= axis.norm()
    theta = torch.rand(1, dtype=torch.float32) * 2 * np.pi
    K = torch.tensor([[0, -axis[2], axis[1]],
                       [axis[2], 0, -axis[0]],
                       [-axis[1], axis[0], 0]], dtype=torch.float32)
    return torch.eye(3) + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def _rotate_3d_columns(data, slc, R, shape_suffix):
    """In-place rotate a (..., 3) slice of data by R."""
    flat = data[slc].reshape(-1, 3)
    data[slc] = torch.matmul(flat, R.T).reshape(*shape_suffix)


def rotate_face_features(ndata, R):
    """Rotate UV-grid face features (coords ch0-2, normals ch3-5) in-place."""
    n, h, w, _ = ndata["x"].shape
    s = (n, h, w, 3)
    _rotate_3d_columns(ndata["x"], (..., slice(0, 3)), R, s)
    _rotate_3d_columns(ndata["x"], (..., slice(3, 6)), R, s)
    if ndata["face_feat"] is not None and ndata["face_feat"].shape[1] >= 10:
        ndata["face_feat"][:, 7:10] = torch.matmul(ndata["face_feat"][:, 7:10], R.T)


def rotate_edge_features(edata, R):
    """Rotate edge features (two 3-vectors in first 6 channels) in-place."""
    ne, ns, _ = edata["x"].shape
    s = (ne, ns, 3)
    _rotate_3d_columns(edata["x"], (..., slice(0, 3)), R, s)
    _rotate_3d_columns(edata["x"], (..., slice(3, 6)), R, s)


def get_filenames(root_dir, filelist=None, ext=".bin"):
    """List files in root_dir, optionally filtered by a filelist txt."""
    root_dir = pathlib.Path(root_dir)
    if filelist:
        with open(str(root_dir / filelist)) as f:
            stems = {x.strip().replace("_rotated", "") for x in f}
        lookup = {p.stem.replace("_rotated", ""): p for p in root_dir.rglob(f"*{ext}")}
        return [lookup[s] for s in stems if s in lookup]
    return list(root_dir.rglob(f"*{ext}"))


def build_class_mapping(file_paths, split_token="_", idx=0, lowercase=True):
    names = sorted({
        (fn.stem.split(split_token, 1)[idx].lower() if lowercase
         else fn.stem.split(split_token, 1)[idx])
        for fn in file_paths
    })
    return {n: i for i, n in enumerate(names)}


def fractional_sample(file_paths, fraction):
    if fraction is None or fraction >= 1.0:
        return file_paths
    k = max(1, int(len(file_paths) * fraction))
    return [file_paths[i] for i in torch.randperm(len(file_paths))[:k].tolist()]
