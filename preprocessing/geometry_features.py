"""
Geometric Feature Extraction module

Extracts face attributes, UV grids, vision grids, and local coordinate frames.
"""

import numpy as np
from typing import List, Optional, Tuple

# OpenCASCADE
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface, GeomAbs_BezierSurface, GeomAbs_Cone, GeomAbs_Cylinder, GeomAbs_Plane, GeomAbs_Sphere, GeomAbs_Torus
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.TopAbs import TopAbs_REVERSED
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.gp import gp_Pnt2d

# occwl
from occwl.face import Face
from occwl.uvgrid import uvgrid

# Local imports
try:
    from preprocessing.ray_casting import raycast_hemisphere
except ImportError:
    import sys
    import pathlib
    sys.path.append(str(pathlib.Path(__file__).parent.parent))
    from preprocessing.ray_casting import raycast_hemisphere

# Constants
ZERO_THRESHOLD = 1e-6
ANGLE_TOLERANCE_RADS = 0.0872664626  # 5 degrees


def _as_topods(entity) -> TopoDS_Shape:
    """Extract TopoDS_Shape from occwl wrapper or return as-is."""
    return entity.topods_shape() if hasattr(entity, "topods_shape") else entity


def scale_solid_to_unit_box(solid):
    """Scale solid to [-1,1]³ box centered at origin. Handles compound solids."""
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
    from OCC.Core.gp import gp_Pnt, gp_Trsf, gp_Vec
    from occwl.solid import Solid
    
    bbox = solid.box()
    center = bbox.center()
    max_size = max(bbox.x_length(), bbox.y_length(), bbox.z_length())
    
    # Translate to origin then scale
    translate_trsf = gp_Trsf()
    translate_trsf.SetTranslation(gp_Vec(-center[0], -center[1], -center[2]))
    translated_shape = BRepBuilderAPI_Transform(solid.topods_shape(), translate_trsf, True).Shape()
    
    scale_trsf = gp_Trsf()
    scale_trsf.SetScale(gp_Pnt(0, 0, 0), 2.0 / max_size)
    scaled_shape = BRepBuilderAPI_Transform(translated_shape, scale_trsf, True).Shape()
    
    return Solid(scaled_shape, allow_compound=True)


def extract_step_face_labels(file_path: str) -> list:
    """Extract integer face labels from STEP file ADVANCED_FACE entities."""
    import re
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Pattern: #17 = ADVANCED_FACE('24',(#18),#32,.F.);
        pattern = r'#\d+\s*=\s*ADVANCED_FACE\(\'([^\']*)\''
        matches = re.findall(pattern, content)
        
        # Convert to int, skip invalid
        return [int(label) for label in matches if label.strip() and label.isdigit()]
    except Exception as e:
        print(f"Warning: Could not extract labels from {file_path}: {e}")
        return []

def extract_face_attributes(
    face,
    attribute_list: List[str],
    points: Optional[np.ndarray] = None,
    solid_bbox_diag: Optional[float] = None
) -> List[float]:
    """Extract geometric attributes from face (surface type, area, centroid, etc.)."""
    if face is None or not attribute_list:
        return [0.0] * sum([3 if a == "FaceCentroidAttribute" else 1 for a in attribute_list])

    try:
        face_topods = _as_topods(face)
        surf = BRepAdaptor_Surface(face_topods)
        surf_type = surf.GetType()
    except Exception:
        return [0.0] * sum([3 if a == "FaceCentroidAttribute" else 1 for a in attribute_list])

    # Compute geometry properties once if needed
    geom_props = None
    if any(attr in {"FaceAreaAttribute", "FaceCentroidAttribute"} for attr in attribute_list):
        try:
            geom_props = GProp_GProps()
            brepgprop_SurfaceProperties(face_topods, geom_props)
        except Exception:
            geom_props = None

    def get_rational_nurbs():
        try:
            if surf_type == GeomAbs_BSplineSurface:
                return 1.0 if surf.BSpline().IsURational() or surf.BSpline().IsVRational() else 0.0
            elif surf_type == GeomAbs_BezierSurface:
                return 1.0 if surf.Bezier().IsURational() or surf.Bezier().IsVRational() else 0.0
        except Exception:
            pass
        return 0.0

    def get_face_area():
        if geom_props is None:
            return 0.0
        try:
            area = abs(float(geom_props.Mass()))
            if points is not None and solid_bbox_diag is not None:
                flat_points = points.reshape(-1, 3)
                if flat_points.shape[0] > 1:
                    min_threshold = (solid_bbox_diag * 1e-6) ** 2
                    max_threshold = solid_bbox_diag ** 2 * 10
                    if area < min_threshold or area > max_threshold:
                        return 0.0
            return area
        except Exception:
            return 0.0

    def get_centroid():
        if geom_props is None:
            return (0.0, 0.0, 0.0)
        try:
            c = geom_props.CentreOfMass()
            centroid = (float(c.X()), float(c.Y()), float(c.Z()))
            if points is not None and solid_bbox_diag is not None:
                flat_points = points.reshape(-1, 3)
                if flat_points.shape[0] > 1:
                    grid_centroid = np.mean(flat_points, axis=0)
                    centroid_distance = np.linalg.norm(np.array(centroid) - grid_centroid)
                    if centroid_distance > solid_bbox_diag * 2:
                        return tuple(grid_centroid)
            return centroid
        except Exception:
            return (0.0, 0.0, 0.0)

    attribute_map = {
        "Plane": lambda: float(surf_type == GeomAbs_Plane),
        "Cylinder": lambda: float(surf_type == GeomAbs_Cylinder),
        "Cone": lambda: float(surf_type == GeomAbs_Cone),
        "SphereFaceAttribute": lambda: float(surf_type == GeomAbs_Sphere),
        "TorusFaceAttribute": lambda: float(surf_type == GeomAbs_Torus),
        "FaceAreaAttribute": get_face_area,
        "RationalNurbsFaceAttribute": get_rational_nurbs,
        "FaceCentroidAttribute": get_centroid,
    }

    result = []
    for attr_name in attribute_list:
        val = attribute_map[attr_name]()
        result.extend(val) if attr_name == "FaceCentroidAttribute" else result.append(val)
    return result

def extract_face_vision_features(
    shape: TopoDS_Shape,
    num_elev: int = 12,
    num_azim: int = 12,
    max_dist: Optional[float] = None,
    center: Optional[np.ndarray] = None,
    axes: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Cast hemisphere rays from face to capture spatial context (6-channel vision grid)."""
    _, grids = raycast_hemisphere(
        shape, center, axes, num_elev, num_azim,
        max_dist=max_dist, compute_dot=True
    )
    
    default_2d = (num_elev, num_azim)
    zero_2d = np.zeros(default_2d, dtype=np.float32)
    
    # Stack 6 channels: occupancy(2) + distance(2) + dot(2)
    grid = np.stack([
        grids['occupancy_grid'],
        grids['distance_grid'],
        grids.get('dot_grid', zero_2d),
        grids.get('occupancy_grid_opposite', zero_2d),
        grids.get('distance_grid_opposite', zero_2d),
        grids.get('dot_grid_opposite', zero_2d)
    ], axis=-1)
    
    return grid


def compute_local_frame(
    face_shape: TopoDS_Shape,
    points: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    uv_points: Optional[np.ndarray] = None,
    num_u: int = 10,
    num_v: int = 10,
    file_path: Optional[str] = None,
    untrimmed_center: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute face-local coordinate frame (center, axes) aligned with UV parametrization."""
    if not isinstance(face_shape, TopoDS_Shape):
        face_shape = face_shape.topods_shape()
    
    surf = BRepAdaptor_Surface(face_shape)
    face = Face(face_shape)
    
    # Parametric midpoint
    u_mid = (surf.FirstUParameter() + surf.LastUParameter()) * 0.5
    v_mid = (surf.FirstVParameter() + surf.LastVParameter()) * 0.5
    
    props_mid = BRepLProp_SLProps(surf, 1, ZERO_THRESHOLD * 1e-3)
    props_mid.SetParameters(u_mid, v_mid)
    orig_center_pnt = props_mid.Value()
    orig_center = np.array([orig_center_pnt.X(), orig_center_pnt.Y(), orig_center_pnt.Z()])
    
    # Choose center location
    if untrimmed_center and face._trimmed.Perform(gp_Pnt2d(u_mid, v_mid)) not in [0, 2]:
        if points is None or mask is None or uv_points is None:
            points, uv_points = uvgrid(face, method="point", uvs=True, num_u=num_u, num_v=num_v)
            visibility, _ = uvgrid(face, method="visibility_status", uvs=True, num_u=num_u, num_v=num_v)
            mask = (visibility == 0) | (visibility == 2)
        
        valid_pts = points.reshape(-1, 3)[mask.ravel()]
        valid_uvs = uv_points.reshape(-1, 2)[mask.ravel()]
        
        if valid_pts.shape[0] > 0:
            geo_center = valid_pts.mean(axis=0)
            best_idx = np.argmin(np.sum((valid_pts - geo_center) ** 2, axis=1))
            u_mid, v_mid = valid_uvs[best_idx]
            props_mid = BRepLProp_SLProps(surf, 1, ZERO_THRESHOLD * 1e-3)
            props_mid.SetParameters(float(u_mid), float(v_mid))
    
    # Extract center and normal
    center_pnt = props_mid.Value()
    normal_vec = props_mid.Normal()
    center = np.array([center_pnt.X(), center_pnt.Y(), center_pnt.Z()])

    # Define Z-axis as surface normal
    z_axis = np.array([normal_vec.X(), normal_vec.Y(), normal_vec.Z()])
    norm_z = np.linalg.norm(z_axis)
    if norm_z > ZERO_THRESHOLD:
        z_axis /= norm_z

    if face_shape.Orientation() == TopAbs_REVERSED:
        z_axis = -z_axis

    # Project U-tangent onto tangent plane to get X-axis
    u_deriv_vec = surf.Surface().DN(float(u_mid), float(v_mid), 1, 0)
    u_deriv = np.array([u_deriv_vec.X(), u_deriv_vec.Y(), u_deriv_vec.Z()])
    x_axis = u_deriv - np.dot(u_deriv, z_axis) * z_axis
    norm_x = np.linalg.norm(x_axis)
    if norm_x > ZERO_THRESHOLD:
        x_axis /= norm_x

    # Y-axis via cross product
    y_axis = np.cross(z_axis, x_axis)
    
    axes = np.stack([x_axis, y_axis, z_axis], axis=1)
    return center, axes, orig_center


def process_single_face(args):
    """Process a single face for parallel execution."""
    (
        face_idx, node_id, face_shape, step_face_labels, segmentation,
        include_uv_face, include_face_attributes, include_vision_grids,
        surf_num_u_samples, surf_num_v_samples, vision_grid_elev, vision_grid_azim,
        face_attribute_list, max_ray_dist, diag, solid_shape, file_path
    ) = args
    
    result = {
        'face_idx': face_idx,
        'face_shape': face_shape,
        'face_type': None,
        'surface_area': 0.0,
        'is_curved': False,
        'has_holes': False,
    }
    
    # Handle segmentation labels
    if segmentation:
        label = -1
        if step_face_labels and face_idx < len(step_face_labels):
            try:
                label = step_face_labels[face_idx]
            except (IndexError, TypeError):
                pass
        result['label'] = label

    # Extract UV grid geometry
    points, normals, mask, uv_points, center, axes = None, None, None, None, None, None
    try:
        points, uv_points = uvgrid(
            face_shape, method="point", uvs=True, num_u=surf_num_u_samples, num_v=surf_num_v_samples
        )
    except Exception as e:
        print(f"Error extracting UV points for face {face_idx}: {e}")
        return result

    if include_uv_face:
        try:
            normals = uvgrid(
                face_shape, method="normal", num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            visibility, _ = uvgrid(
                face_shape, method="visibility_status", uvs=True, num_u=surf_num_u_samples, num_v=surf_num_v_samples
            )
            if visibility.ndim > 2:
                visibility = visibility.squeeze()
            mask = (visibility == 0) | (visibility == 2)
        except Exception as e:
            print(f"Error extracting UV features for face {face_idx}: {e}")
            return result

    # Compute local frame
    try:
        center, axes, _ = compute_local_frame(
            face_shape, points=points, mask=mask, uv_points=uv_points, file_path=file_path
        )
    except Exception as e:
        print(f"Error computing local frame for face {face_idx}: {e}")
        return result

    if include_uv_face and points is not None and normals is not None and mask is not None:
        try:
            if mask.ndim == 2:
                mask_3d = mask[..., np.newaxis].astype(np.float32)
            else:
                mask_3d = mask.astype(np.float32)
            
            face_feat = np.concatenate((points, normals, mask_3d), axis=-1)
            result['face_features'] = face_feat

            # Compute local coordinates
            pts_flat = points.reshape(-1, 3)
            nrms_flat = normals.reshape(-1, 3)
            
            center_broadcast = center[np.newaxis, :]
            vecs = pts_flat - center_broadcast
            local_pts = vecs @ axes
            local_nrms = nrms_flat @ axes
            
            local_pts_reshaped = local_pts.reshape(surf_num_u_samples, surf_num_v_samples, 3)
            local_nrms_reshaped = local_nrms.reshape(surf_num_u_samples, surf_num_v_samples, 3)
            local_features = np.concatenate([local_pts_reshaped, local_nrms_reshaped, mask_3d], axis=-1)
            result['face_features_local'] = local_features
        except Exception as e:
            print(f"Error processing UV features for face {face_idx}: {e}")

    if include_face_attributes:
        try:
            attr_values = extract_face_attributes(
                face_shape, face_attribute_list, points=points, solid_bbox_diag=diag
            )
            result['face_attributes'] = attr_values
        except Exception as e:
            print(f"Error extracting face attributes for face {face_idx}: {e}")

    if include_vision_grids:
        try:
            vision_grids = extract_face_vision_features(
                solid_shape, vision_grid_elev, vision_grid_azim,
                max_dist=max_ray_dist, center=center, axes=axes,
            )
            result['vision_grid'] = vision_grids
        except Exception as e:
            print(f"Error computing vision features for face {face_idx}: {e}")
    
    return result