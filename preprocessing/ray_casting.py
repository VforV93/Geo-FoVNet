"""
Ray Casting Module

Key Features:
    - Hemisphere-based directional sampling
    - Bidirectional ray casting (normal and opposite directions)
    - Geometric feature computation (occupancy, distance, surface normals)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAPI import GeomAPI_ProjectPointOnSurf
from OCC.Core.gp import gp_Dir, gp_Lin, gp_Pnt, gp_Vec
from OCC.Core.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_FORWARD, TopAbs_REVERSED

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def load_step_file(filepath: str):
    """Load a STEP file and return the shape."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)
    if status != 1:
        raise RuntimeError(f"Failed to read STEP file '{filepath}'. Reader status: {status}")
    reader.TransferRoots()
    return reader.OneShape()

def hemisphere_grid_sampling(
    axes: np.ndarray, num_elev: int = 8, num_azim: int = 16
) -> np.ndarray:
    """
    Generate hemisphere directions on a grid.

    Args:
        axes: 3x3 array with columns (x_axis, y_axis, z_axis)
        num_elev: elevation divisions (from near 0 to pi/2)
        num_azim: azimuth divisions (0..2pi)

    Returns:
        dirs: (num_elev*num_azim, 3) numpy array of unit directions
    """
    x_axis = axes[:, 0].astype(np.float64)
    y_axis = axes[:, 1].astype(np.float64)
    z_axis = axes[:, 2].astype(np.float64)
    # Special-case: if both elevations and azimuths are 1, return the face normal 
    if int(num_elev) == 1 and int(num_azim) == 1:
        z = z_axis.copy()
        nrm = np.linalg.norm(z)
        if nrm > 0:
            z /= nrm
        return z.reshape(1, 3)


    # Vectorized grid
    i = np.arange(num_elev)
    j = np.arange(num_azim)
    elev = (np.pi / 2) * (i[:, None] + 0.5) / float(num_elev)  # shape (num_elev, 1)
    az = 2 * np.pi * j[None, :] / float(num_azim)              # shape (1, num_azim)


    # Precompute trigonometric values
    cos_elev = np.cos(elev)
    sin_elev = np.sin(elev)
    cos_az = np.cos(az)
    sin_az = np.sin(az)

    # Compute directions for all grid points (fully vectorized)
    dirs = (cos_elev[..., None] * z_axis +
            sin_elev[..., None] * (cos_az[..., None] * x_axis + sin_az[..., None] * y_axis))
    # Normalize all directions
    norm = np.linalg.norm(dirs, axis=2, keepdims=True)
    dirs = np.divide(dirs, norm, out=np.zeros_like(dirs), where=norm!=0).reshape(-1, 3)

    return dirs

def raycast_hemisphere(
    shape,
    center: np.ndarray,
    axes: np.ndarray,
    num_elev: int = 8,
    num_azim: int = 16,
    compute_dot: bool = False,
    max_dist: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Perform hemisphere ray casting from a face center and extract features.
    """
    # tolerances
    intersection_tolerance = 1e-2
    ray_tolerance = 1e-4
    max_ray_dist = max_dist if max_dist is not None else 1e6

    # sample directions for both hemispheres
    dirs = hemisphere_grid_sampling(axes, num_elev, num_azim)
    dirs_opposite = hemisphere_grid_sampling(-axes, num_elev, num_azim)

    # initialize intersector
    inter = IntCurvesFace_ShapeIntersector()
    try:
        inter.Load(shape, intersection_tolerance)
    except Exception as exc:
        logger.warning("Failed to load shape into intersector: %s", exc)
        # Build safe fallback outputs (all zeros / no hits)
        N = num_elev * num_azim
        hits_fallback = [ (dirs[i % N].astype(np.float32), None, None) for i in range(2*N) ]
        grids = {
            "occupancy_grid": np.zeros((num_elev, num_azim), dtype=np.float32),
            "distance_grid": np.zeros((num_elev, num_azim), dtype=np.float32),
            "occupancy_grid_opposite": np.zeros((num_elev, num_azim), dtype=np.float32),
            "distance_grid_opposite": np.zeros((num_elev, num_azim), dtype=np.float32),
        }
        if compute_dot:
            grids["dot_grid"] = np.zeros((num_elev, num_azim), dtype=np.float32)
            grids["dot_grid_opposite"] = np.zeros((num_elev, num_azim), dtype=np.float32)

        return center, hits_fallback, grids

    # Preallocate arrays for results
    hits = np.empty((num_elev * num_azim, 3), dtype=object)  # (dir, dist, face)
    hits_opposite = np.empty((num_elev * num_azim, 3), dtype=object)
    occupancy_grid = np.zeros((num_elev, num_azim), dtype=np.float32)
    distance_grid = np.zeros((num_elev, num_azim), dtype=np.float32)
    dot_grid = np.zeros((num_elev, num_azim), dtype=np.float32) if compute_dot else None
    occupancy_grid_opposite = np.zeros((num_elev, num_azim), dtype=np.float32)
    distance_grid_opposite = np.zeros((num_elev, num_azim), dtype=np.float32)
    dot_grid_opposite = np.zeros((num_elev, num_azim), dtype=np.float32) if compute_dot else None
    unique_faces = set()


    def cast_rays(dirs, occupancy_grid, distance_grid, dot_grid, hits_array, unique_faces, is_opposite=False):

        origin = gp_Pnt(*center)
        # Preallocate arrays for vectorized dot product
        ray_normals = np.zeros((dirs.shape[0], 3), dtype=np.float32) if compute_dot else None
        ray_pts = np.zeros((dirs.shape[0], 3), dtype=np.float32) if compute_dot else None

        for idx, d in enumerate(dirs):
            elev_idx = idx // num_azim
            azim_idx = idx % num_azim
            ray = gp_Lin(origin, gp_Dir(*d))
            pt = None
            hit_face = None
            dist = None
            
            try:
                inter.Perform(ray, ray_tolerance, max_ray_dist)
            except RuntimeError:
                hits_array[idx, 0] = d
                hits_array[idx, 1] = None
                hits_array[idx, 2] = None
                occupancy_grid[elev_idx, azim_idx] = 0.0
                distance_grid[elev_idx, azim_idx] = 0.0

                # removed optional hitpoint/normal fallback
                continue
            
            if inter.NbPnt() > 0:
                try:
                    pt = inter.Pnt(1)
                    hit_face = inter.Face(1)
                    if pt is not None:
                        dist = np.linalg.norm(np.array([pt.X(), pt.Y(), pt.Z()]) - center)
                except Exception:
                    pt = None
                    hit_face = None
                    dist = None
            hits_array[idx, 0] = d
            hits_array[idx, 1] = dist
            hits_array[idx, 2] = hit_face
            occupancy_grid[elev_idx, azim_idx] = 1.0 if dist is not None else 0.0
            distance_grid[elev_idx, azim_idx] = dist if dist is not None else 0.0
            if hit_face is not None:
                unique_faces.add(hit_face)
                if compute_dot and pt is not None and dot_grid is not None:
                    try:
                        surf = BRepAdaptor_Surface(hit_face)
                        geom_surf = surf.Surface().Surface()
                        projector = GeomAPI_ProjectPointOnSurf(pt, geom_surf)
                        if projector.NbPoints() > 0:
                            u, v = projector.LowerDistanceParameters()
                            
                            P, D1U, D1V = gp_Pnt(), gp_Vec(), gp_Vec()
                            geom_surf.D1(u, v, P, D1U, D1V)
                            n_vec = D1U.Crossed(D1V)
                            if (not is_opposite and hit_face.Orientation() == TopAbs_REVERSED) or (is_opposite and hit_face.Orientation() == TopAbs_FORWARD):
                                n_vec = -n_vec
                            if n_vec.Magnitude() > 0:
                                n_vec.Normalize()
                                n = np.array([n_vec.X(), n_vec.Y(), n_vec.Z()])
                                ray_normals[idx] = n
                                ray_pts[idx] = np.array([pt.X(), pt.Y(), pt.Z()])
                    except Exception:
                        pass
        
        # Vectorized dot product calculation for all rays
        if compute_dot and dot_grid is not None and ray_normals is not None:
            valid_mask = np.linalg.norm(ray_normals, axis=1) > 0
            dot_products = np.einsum('ij,ij->i', dirs, ray_normals)
            for idx in range(dirs.shape[0]):
                elev_idx = idx // num_azim
                azim_idx = idx % num_azim
                if valid_mask[idx]:
                    dot_grid[elev_idx, azim_idx] = dot_products[idx]


    # Cast rays for both hemispheres
    cast_rays(dirs, occupancy_grid, distance_grid, dot_grid, hits, unique_faces, is_opposite=False)
    cast_rays(dirs_opposite, occupancy_grid_opposite, distance_grid_opposite, dot_grid_opposite, hits_opposite, unique_faces, is_opposite=True)

    grids = {
        'occupancy_grid': occupancy_grid,
        'distance_grid': distance_grid,
        'occupancy_grid_opposite': occupancy_grid_opposite,
        'distance_grid_opposite': distance_grid_opposite
    }
    if compute_dot:
        grids['dot_grid'] = dot_grid
        grids['dot_grid_opposite'] = dot_grid_opposite

    return center, grids