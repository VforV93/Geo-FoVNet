"""
FOVNet Feature Extraction - Main processing

This module processes STEP files to extract the features from geometry_features.py and ray_casting.py
"""

# Standard library
import argparse
import contextlib
import multiprocessing
import os
import pathlib
import warnings
from typing import Any, List, Optional, Tuple

# Third-party
import dgl
import numpy as np
import torch
from tqdm import tqdm

from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepGProp import brepgprop_LinearProperties
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.GeomAbs import GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Line
from OCC.Core.GProp import GProp_GProps
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Extend import TopologyUtils

# occwl
from occwl.compound import Compound
from occwl.edge import Edge
from occwl.edge_data_extractor import EdgeConvexity, EdgeDataExtractor
from occwl.face import Face
from occwl.graph import face_adjacency
from occwl.solid import Solid

# Local geometry features module
from geometry_features import (
    process_single_face,
    scale_solid_to_unit_box,
    extract_step_face_labels,
)

# Configuration
np.set_printoptions(precision=3)
torch.set_printoptions(precision=3, sci_mode=False)
torch.set_num_threads(1)
np.seterr(all='ignore')
warnings.filterwarnings('ignore')

DEFAULT_FACE_ATTRIBUTES = [
    "Plane", "Cylinder", "Cone", "SphereFaceAttribute", "TorusFaceAttribute",
    "FaceAreaAttribute", "RationalNurbsFaceAttribute",
]

DEFAULT_EDGE_ATTRIBUTES = [
    "Concave edge", "Convex edge", "Smooth", "EdgeLengthAttribute",
    "CircularEdgeAttribute", "ClosedEdgeAttribute", "EllipticalEdgeAttribute",
    "NonRationalBSplineEdgeAttribute", "RationalBSplineEdgeAttribute", "StraightEdgeAttribute",
]

ANGLE_TOLERANCE_RADS = 0.0872664626  # 5 degrees

@contextlib.contextmanager
def suppress_stdout_stderr_fd():
    """Suppress C library output at file descriptor level."""
    old_stdout, old_stderr = os.dup(1), os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stdout, 1)
        os.dup2(old_stderr, 2)
        os.close(old_stdout)
        os.close(old_stderr)
        os.close(devnull)


def _as_topods(entity):
    """Extract TopoDS_Shape from occwl wrapper or return as-is."""
    return entity.topods_shape() if hasattr(entity, "topods_shape") else entity


def _edge_convexity(edge_topods, faces_of_edge: List[Face]) -> Optional[EdgeConvexity]:
    """Compute edge convexity (concave/convex/smooth) from dihedral angle."""
    try:
        edge_data = EdgeDataExtractor(Edge(edge_topods), faces_of_edge, use_arclength_params=False)
        return edge_data.edge_convexity(ANGLE_TOLERANCE_RADS) if edge_data.good else None
    except Exception:
        return None


def extract_aag_edge_attributes(edge, attribute_list: List[str], topology_explorer: TopologyUtils.TopologyExplorer) -> List[float]:
    """Extract geometric attributes from edge (curve type, length, convexity, etc.)."""
    if edge is None or not isinstance(attribute_list, list) or not attribute_list or topology_explorer is None:
        return [0.0] * len(attribute_list)

    edge_topods = _as_topods(edge)
    try:
        faces_of_edge = [Face(f) for f in topology_explorer.faces_from_edge(edge_topods)]
    except Exception:
        return [0.0] * len(attribute_list)

    # Pre-compute only what's needed
    curve_adaptor = None
    curve_type = None
    edge_wrapper = None
    edge_curve_type = None
    edge_rational = False
    geometry_properties = None
    convexity = None

    needs_curve_adaptor = any(attr in {"CircularEdgeAttribute", "EllipticalEdgeAttribute", "StraightEdgeAttribute"}
                             for attr in attribute_list)
    needs_edge_wrapper = any(attr in {"HyperbolicEdgeAttribute", "ParabolicEdgeAttribute", "BezierEdgeAttribute",
                                    "NonRationalBSplineEdgeAttribute", "RationalBSplineEdgeAttribute", "OffsetEdgeAttribute"}
                           for attr in attribute_list)
    needs_geometry = "EdgeLengthAttribute" in attribute_list
    needs_convexity = any(attr in {"Concave edge", "Convex edge", "Smooth"} for attr in attribute_list)

    if needs_curve_adaptor:
        try:
            curve_adaptor = BRepAdaptor_Curve(edge_topods)
            curve_type = curve_adaptor.GetType()
        except Exception:
            pass

    if needs_edge_wrapper:
        try:
            edge_wrapper = Edge(edge_topods)
            edge_curve_type = edge_wrapper.curve_type()
            edge_rational = edge_wrapper.rational() if edge_curve_type == "bspline" else False
        except Exception:
            pass

    if needs_geometry:
        try:
            geometry_properties = GProp_GProps()
            brepgprop_LinearProperties(edge_topods, geometry_properties)
        except Exception:
            pass

    if needs_convexity:
        try:
            convexity = _edge_convexity(edge_topods, faces_of_edge)
        except Exception:
            pass

    convexity_mapping = {
        "Concave edge": EdgeConvexity.CONCAVE,
        "Convex edge": EdgeConvexity.CONVEX,
        "Smooth": EdgeConvexity.SMOOTH,
    }

    attribute_map = {
        "EdgeLengthAttribute": lambda: float(geometry_properties.Mass()) if geometry_properties else 0.0,
        "CircularEdgeAttribute": lambda: float(curve_type == GeomAbs_Circle) if curve_type is not None else 0.0,
        "ClosedEdgeAttribute": lambda: float(BRep_Tool().IsClosed(edge_topods)) if edge_topods else 0.0,
        "EllipticalEdgeAttribute": lambda: float(curve_type == GeomAbs_Ellipse) if curve_type is not None else 0.0,
        "StraightEdgeAttribute": lambda: float(curve_type == GeomAbs_Line) if curve_type is not None else 0.0,
        "HyperbolicEdgeAttribute": lambda: float(edge_curve_type == "hyperbola") if edge_curve_type else 0.0,
        "ParabolicEdgeAttribute": lambda: float(edge_curve_type == "parabola") if edge_curve_type else 0.0,
        "BezierEdgeAttribute": lambda: float(edge_curve_type == "bezier") if edge_curve_type else 0.0,
        "NonRationalBSplineEdgeAttribute": lambda: float(edge_curve_type == "bspline" and not edge_rational) if edge_curve_type else 0.0,
        "RationalBSplineEdgeAttribute": lambda: float(edge_curve_type == "bspline" and edge_rational) if edge_curve_type else 0.0,
        "OffsetEdgeAttribute": lambda: float(edge_curve_type == "offset") if edge_curve_type else 0.0,
    }

    attribute_values = []
    for attr_name in attribute_list:
        if attr_name in convexity_mapping:
            val = float(convexity == convexity_mapping[attr_name]) if convexity else 0.0
            attribute_values.append(val)
        elif attr_name in attribute_map:
            attribute_values.append(attribute_map[attr_name]())
        else:
            attribute_values.append(0.0)
    return attribute_values


def compute_edge_uv_grids(edge, faces_of_edge: List[Face], num_samples: int) -> np.ndarray:
    """Sample edge curve: points, tangents, left/right face normals."""
    edge_topods = _as_topods(edge)
    if num_samples <= 0:
        return np.zeros((0, 12), dtype=np.float32)

    try:
        edge_data = EdgeDataExtractor(
            Edge(edge_topods),
            faces_of_edge,
            num_samples=num_samples,
            use_arclength_params=True,
        )
    except Exception:
        edge_data = None

    if edge_data is None or not edge_data.good:
        return np.zeros((num_samples, 12), dtype=np.float32)

    return np.concatenate([
        edge_data.points,
        edge_data.tangents,
        edge_data.left_normals,
        edge_data.right_normals
    ], axis=1).astype(np.float32)

def build_graph(
    file_path: str,
    solid: Solid,
    surf_num_u_samples: int,
    surf_num_v_samples: int,
    curv_num_u_samples: int = 10,
    vision_grid_elev: int = 12,
    vision_grid_azim: int = 12,
    include_vision_grids: bool = True,
    include_uv_face: bool = True,
    include_uv_edge: bool = False,
    include_face_attributes: bool = False,
    include_edge_attributes: bool = False,
    face_attribute_list: Optional[List[str]] = None,
    edge_attribute_list: Optional[List[str]] = None,
    parallel_level: str = "file",
    num_processes: int = 22,
    segmentation: bool = False,
) -> Optional[dgl.DGLGraph]:
    """
    Build DGL graph from CAD solid with face adjacency and rich geometric/vision features.
    
    Returns DGL graph with:
        Node data: UV grids, vision grids, geometric attributes, labels
        Edge data: Curve samples, geometric attributes
    """
    try:
        graph = face_adjacency(solid)
    except Exception as e:
        print(f"Error in face_adjacency for file: {file_path}. Skipping. Reason: {e}")
        return None

    solid_shape = solid.topods_shape()
    topology_explorer = TopologyUtils.TopologyExplorer(solid_shape, ignore_orientation=True)

    # Extract face labels from the STEP file if segmentation is True
    step_face_labels = extract_step_face_labels(file_path) if segmentation else None

    node_ids = sorted(list(graph.nodes))  # Ensure deterministic ordering
    edge_keys = sorted(list(graph.edges))  # Ensure deterministic ordering
    num_faces = len(node_ids)
    num_edges = len(edge_keys)

    if num_faces == 0:
        return None

    box = Bnd_Box()
    brepbndlib_Add(solid_shape, box)
    xmin, ymin, zmin, xmax, ymax, zmax = box.Get()
    diag = np.sqrt((xmax - xmin) ** 2 + (ymax - ymin) ** 2 + (zmax - zmin) ** 2)
    max_ray_dist = diag * 2

    # Pre-compute face feature dimension (avoid sample uvgrid call)
    face_feat_dim = 3 * 2 + 1  # points (3) + normals (3) + mask (1) = 7

    face_attribute_list = list(face_attribute_list) if face_attribute_list is not None else DEFAULT_FACE_ATTRIBUTES
    edge_attribute_list = list(edge_attribute_list) if edge_attribute_list is not None else DEFAULT_EDGE_ATTRIBUTES

    # Pre-allocate arrays with known dimensions
    graph_face_points = None
    graph_face_points_local = None
    if include_uv_face:
        graph_face_points = np.zeros(
            (num_faces, surf_num_u_samples, surf_num_v_samples, face_feat_dim),
            dtype=np.float32,
        )
        graph_face_points_local = np.zeros_like(graph_face_points)

    graph_face_vision_grids = None
    graph_face_vision_features = None
    if include_vision_grids:
        graph_face_vision_grids = np.zeros(
            (num_faces, vision_grid_elev, vision_grid_azim, 6), dtype=np.float32
        )
        graph_face_vision_features = np.zeros((num_faces, 4), dtype=np.float32)

    graph_face_attr: Optional[List[List[float]]] = [] if include_face_attributes else None

    graph_face_labels: List[int] = []

    # Pre-allocate edge arrays
    edge_uv_channels = 12  # points(3) + tangents(3) + left_normals(3) + right_normals(3)
    graph_edge_uv = None
    if include_uv_edge:
        edge_samples = max(curv_num_u_samples, 0)
        graph_edge_uv = np.zeros((num_edges, edge_samples, edge_uv_channels), dtype=np.float32)

    graph_edge_attr: Optional[List[List[float]]] = [] if include_edge_attributes else None

    # Initialize face type tracking for timing metadata
    face_types = {}
    total_surface_area = 0.0

    # Process faces (with optional parallelization)
    if parallel_level == "face" and num_processes > 1:
        # Prepare arguments for parallel face processing
        face_args = []
        for face_idx, node_id in enumerate(node_ids):
            face_shape = graph.nodes[node_id]["face"]
            args = (
                face_idx, node_id, face_shape, step_face_labels, segmentation,
                include_uv_face, include_face_attributes, include_vision_grids,
                surf_num_u_samples, surf_num_v_samples, vision_grid_elev, vision_grid_azim,
                face_attribute_list, max_ray_dist, diag, solid_shape, file_path
            )
            face_args.append(args)
        
        # Process faces in parallel
        with multiprocessing.Pool(processes=min(num_processes, len(face_args))) as pool:
            face_results = pool.map(process_single_face, face_args)
        
        # Process results from parallel execution
        for result in face_results:
            face_idx = result['face_idx']
            
            # Collect face type statistics
            if result['face_type']:
                face_types[result['face_type']] = face_types.get(result['face_type'], 0) + 1
                total_surface_area += result['surface_area']
                if result['is_curved']:
                    has_curved_faces = True
            
            # Handle segmentation labels
            if segmentation and 'label' in result:
                graph_face_labels.append(result['label'])
            
            # Store face features
            if include_uv_face and 'face_features' in result:
                graph_face_points[face_idx] = result['face_features']
                if 'face_features_local' in result:
                    graph_face_points_local[face_idx] = result['face_features_local']
            
            # Store face attributes
            if include_face_attributes and 'face_attributes' in result:
                graph_face_attr.append(result['face_attributes'])
            
            # Store vision features
            if include_vision_grids and 'vision_grid' in result:
                graph_face_vision_grids[face_idx] = result['vision_grid']
    
    else:
        # Sequential processing - use process_single_face for consistency
        for face_idx, node_id in enumerate(node_ids):
            face_shape = graph.nodes[node_id]["face"]
            
            args = (
                face_idx, node_id, face_shape, step_face_labels, segmentation,
                include_uv_face, include_face_attributes, include_vision_grids,
                surf_num_u_samples, surf_num_v_samples, vision_grid_elev, vision_grid_azim,
                face_attribute_list, max_ray_dist, diag, solid_shape, file_path
            )
            
            result = process_single_face(args)
            face_idx = result['face_idx']
            
            # Collect face type statistics
            if result.get('face_type'):
                face_types[result['face_type']] = face_types.get(result['face_type'], 0) + 1
                total_surface_area += result.get('surface_area', 0.0)
            
            # Handle segmentation labels
            if segmentation and 'label' in result:
                graph_face_labels.append(result['label'])
            
            # Store face features
            if include_uv_face and 'face_features' in result:
                graph_face_points[face_idx] = result['face_features']
                if 'face_features_local' in result:
                    graph_face_points_local[face_idx] = result['face_features_local']
            
            # Store face attributes
            if include_face_attributes and 'face_attributes' in result:
                graph_face_attr.append(result['face_attributes'])
            
            # Store vision features
            if include_vision_grids and 'vision_grid' in result:
                graph_face_vision_grids[face_idx] = result['vision_grid']

    # Process edges
    for edge_idx, edge_key in enumerate(edge_keys):
        edge_entity = graph.edges[edge_key]["edge"]
        edge_topods = _as_topods(edge_entity)
        faces_of_edge = [Face(f) for f in topology_explorer.faces_from_edge(edge_topods)]

        # Edge UV grid extraction
        if include_uv_edge and graph_edge_uv is not None:
            global_grid = compute_edge_uv_grids(edge_entity, faces_of_edge, curv_num_u_samples)
            samples = min(graph_edge_uv.shape[1], global_grid.shape[0])
            if samples > 0:
                graph_edge_uv[edge_idx, :samples, :] = global_grid[:samples]

        # Edge attribute extraction
        if include_edge_attributes and graph_edge_attr is not None:
            attr_values = extract_aag_edge_attributes(edge_entity, edge_attribute_list, topology_explorer)
            graph_edge_attr.append(attr_values)

    # Create DGL graph with face-adjacency edges
    src = [e[0] for e in edge_keys]
    dst = [e[1] for e in edge_keys]
    dgl_graph = dgl.graph((src, dst), num_nodes=num_faces)

    # Optimized tensor creation with direct conversion to reduce memory copies
    if include_uv_face and graph_face_points is not None:
        dgl_graph.ndata["x"] = torch.from_numpy(graph_face_points.astype(np.float32, copy=False))
        dgl_graph.ndata["x_local"] = torch.from_numpy(graph_face_points_local.astype(np.float32, copy=False))

    if include_face_attributes and graph_face_attr is not None:
        face_attr_array = np.asarray(graph_face_attr, dtype=np.float32)
        dgl_graph.ndata["face_feat"] = torch.from_numpy(face_attr_array)

    if include_vision_grids and graph_face_vision_grids is not None:
        dgl_graph.ndata["vision_grids"] = torch.from_numpy(graph_face_vision_grids.astype(np.float32, copy=False))
        dgl_graph.ndata["vision_features"] = torch.from_numpy(graph_face_vision_features.astype(np.float32, copy=False))

    if segmentation:
        labels_array = np.asarray(graph_face_labels, dtype=np.int64)
        dgl_graph.ndata["y"] = torch.from_numpy(labels_array)

    if include_uv_edge and graph_edge_uv is not None:
        dgl_graph.edata["x"] = torch.from_numpy(graph_edge_uv.astype(np.float32, copy=False))

    if include_edge_attributes and graph_edge_attr is not None:
        edge_attr_array = np.asarray(graph_edge_attr, dtype=np.float32)
        dgl_graph.edata["edge_feat"] = torch.from_numpy(edge_attr_array)

    # Explicit cleanup to reduce memory pressure
    del graph, solid_shape, topology_explorer
    if graph_face_points is not None:
        del graph_face_points, graph_face_points_local
    if graph_face_vision_grids is not None:
        del graph_face_vision_grids, graph_face_vision_features
    if graph_edge_uv is not None:
        del graph_edge_uv

    return dgl_graph
    
# ============================================================================
# FILE PROCESSING FUNCTIONS
# ============================================================================

def save_compressed_graph(graph: dgl.DGLGraph, output_file: pathlib.Path):
    """Save graph with float16 compression. Trims vision_grids to first 12 channels."""
    if "vision_grids" in graph.ndata:
        vg = graph.ndata["vision_grids"]
        graph.ndata["vision_grids"] = vg[..., :12].half() if vg.shape[-1] > 12 else vg.half()
    
    # Convert all float32 to float16
    for key, val in graph.ndata.items():
        if key != "vision_grids" and val.dtype == torch.float32:
            graph.ndata[key] = val.half()
    
    for key, val in graph.edata.items():
        if val.dtype == torch.float32:
            graph.edata[key] = val.half()
    
    dgl.data.utils.save_graphs(str(output_file), [graph])
    
def process_single_file(
    file_path: pathlib.Path,
    output_dir: pathlib.Path,
    surf_num_u_samples: int,
    surf_num_v_samples: int,
    curv_num_u_samples: int = 10,
    vision_grid_elev: int = 12,
    vision_grid_azim: int = 12,
    include_vision_grids: bool = True,
    include_uv_face: bool = True,
    include_uv_edge: bool = False,
    include_face_attributes: bool = True,
    include_edge_attributes: bool = False,
    scale_body: bool = True,
    random_rotate_step: bool = False,
    create_rotated_step_files: bool = False,
    parallel_level: str = "file",
    segmentation: bool = True,
    compress: bool = False,
) -> Optional[dgl.DGLGraph]:
    """Process a single STEP file and extract geometric/vision features."""
    try:
        # Load compound and create solid
        compound = Compound.load_from_step(file_path)
        solids_list = list(compound.solids())
        if not solids_list:
            raise ValueError(f"No solids found in: {file_path}")
        
        solid = solids_list[0] if len(solids_list) == 1 else Solid(compound.topods_shape(), allow_compound=True)
        if scale_body:
            solid = scale_solid_to_unit_box(solid)
    except Exception as e:
        print(f"Error loading or preparing file {file_path.name}: {type(e).__name__}: {str(e)}")
        return None
    try:
        if random_rotate_step:
            def random_rotation_matrix_around_origin():
                """Generate random rotation around origin."""
                # Use file path as seed for deterministic rotation per file
                file_seed = hash(str(file_path)) % (2**32)
                rng = np.random.RandomState(file_seed)
                # Generate random rotation angles (0 to 2π for each axis)
                angles = rng.uniform(0, 2 * np.pi, size=3)
                
                # Create rotation transformations around the origin
                origin = gp_Pnt(0, 0, 0)
                
                Rx = gp_Trsf()
                Rx.SetRotation(gp_Ax1(origin, gp_Dir(1, 0, 0)), angles[0])
                
                Ry = gp_Trsf()
                Ry.SetRotation(gp_Ax1(origin, gp_Dir(0, 1, 0)), angles[1])
                
                Rz = gp_Trsf()
                Rz.SetRotation(gp_Ax1(origin, gp_Dir(0, 0, 1)), angles[2])
                
                # Combine rotations: Rz * Ry * Rx
                R = Rx.Multiplied(Ry).Multiplied(Rz)
                return R
            
            # Apply random rotation around origin
            rotation_trsf = random_rotation_matrix_around_origin()
            rotated_shape = BRepBuilderAPI_Transform(solid.topods_shape(), rotation_trsf, True).Shape()
            
            # Create new solid from rotated shape
            solid = Solid(rotated_shape, allow_compound=True)
            
            # Save rotated STEP file to split_rotated directory
            rotated_step_dir = output_dir.parent  # This is the split_rotated folder
            rotated_step_path = rotated_step_dir / (file_path.stem + "_rotated.stp")
            
            if create_rotated_step_files:
                with suppress_stdout_stderr_fd():
                    writer = STEPControl_Writer()
                    writer.Transfer(rotated_shape, STEPControl_AsIs)
                    writer.Write(str(rotated_step_path))
            
        # Build graph with vision grids
        face_attribute_list = DEFAULT_FACE_ATTRIBUTES if include_face_attributes else None
        edge_attribute_list = DEFAULT_EDGE_ATTRIBUTES if include_edge_attributes else None

        graph = build_graph(
            file_path,
            solid,
            surf_num_u_samples,
            surf_num_v_samples,
            curv_num_u_samples,
            vision_grid_elev,
            vision_grid_azim,
            include_vision_grids=include_vision_grids,
            include_uv_face=include_uv_face,
            include_uv_edge=include_uv_edge,
            include_face_attributes=include_face_attributes,
            include_edge_attributes=include_edge_attributes,
            face_attribute_list=face_attribute_list,
            edge_attribute_list=edge_attribute_list,
            parallel_level=parallel_level,
            segmentation=segmentation,
        )
        if not graph is None:
            # Save graph
            if random_rotate_step:
                output_file = output_dir / (file_path.stem + "_rotated.bin")
            else:
                output_file = output_dir / (file_path.stem + ".bin")

            if compress:
                save_compressed_graph(graph, output_file)
            else:
                dgl.data.utils.save_graphs(str(output_file), [graph])
        
        return graph
    except Exception as e:
        print(f"Error processing file {file_path.name}: {type(e).__name__}: {str(e)}")
        return None

def process_file_with_output_dir(args):
    (
        f,
        outdir,
        surf_num_u_samples,
        surf_num_v_samples,
        curv_num_u_samples,
        vision_grid_elev,
        vision_grid_azim,
        include_vision_grids,
        include_uv_face,
        include_uv_edge,
        include_face_attributes,
        include_edge_attributes,
        scale_body,
        random_rotate_step,
        create_rotated_step_files,
        parallel_level,
        segmentation,
        compress
    ) = args

    try:
        result = process_single_file(
            f,
            outdir,
            surf_num_u_samples,
            surf_num_v_samples,
            curv_num_u_samples,
            vision_grid_elev,
            vision_grid_azim,
            include_vision_grids,
            include_uv_face,
            include_uv_edge,
            include_face_attributes,
            include_edge_attributes,
            scale_body,
            random_rotate_step,
            create_rotated_step_files,
            parallel_level,
            segmentation,
            compress
        )
        return (True, f.name, None)
    except Exception as e:
        error_msg = f"Error processing {f.name}: {type(e).__name__}: {str(e)}"
        print(error_msg)
        return (False, f.name, str(e))
def process_multiple_files(
    input_dir: pathlib.Path,
    output_dir: pathlib.Path,
    surf_num_u_samples: int,
    surf_num_v_samples: int,
    curv_num_u_samples: int = 10,
    vision_grid_elev: int = 12,
    vision_grid_azim: int = 12,
    include_vision_grids: bool = True,
    include_uv_face: bool = True,
    include_uv_edge: bool = False,
    include_face_attributes: bool = True,
    include_edge_attributes: bool = False,
    scale_body: bool = True,
    num_processes: int = 22,
    skip_existing: bool = True,
    dataset: str = "bendfm",
    random_rotate_step: bool = False,
    create_rotated_step_files: bool = False,
    parallel_level: str = "file",
    segmentation: bool = True,
    compress: bool = False,
    max_files: Optional[int] = None,
) -> Tuple[List[Any], List[str]]:
    """Process multiple STEP files with vision grid features using parallel processing."""
    step_files = []
    file_output_dirs = []

    if dataset == "solidletters":
        # Pre-compile invalid fonts set for O(1) lookup
        INVALID_FONTS = frozenset([
            "Bokor", "Lao Muang Khong", "Lao Sans Pro", "MS Outlook", "Catamaran Black",
            "Dubai", "HoloLens MDL2 Assets", "Lao Muang Don", "Oxanium Medium", 
            "Rounded Mplus 1c", "Moul Pali", "Noto Sans Tamil", "Webdings", "Armata",
            "Koulen", "Yinmar", "Ponnala", "Chenla", "Lohit Devanagari", "Metal",
            "MS Office Symbol", "Cormorant Garamond Medium", "Chiller", "Give You Glory",
            "Hind Vadodara Light", "Libre Barcode 39 Extended", "Myanmar Sans Pro",
            "Scheherazade", "Segoe MDL2 Assets", "Siemreap", "Signika SemiBold",
            "Taprom", "Times New Roman TUR", "Playfair Display SC Black", "Poppins Thin",
            "Raleway Dots", "Raleway Thin", "Spectral SC ExtraLight", "Txt", "Uchen",
            "Almarai ExtraBold", "Fasthand", "Exo", "Freckle Face", "Montserrat Light",
            "Inter", "MS Reference Specialty", "Preah Vihear", "Sitara",
            "Barkerville Old Face", "Bodoni MT", "HoloLens MDL2 Assests", 
            "Libre Barcode 39", "Lohit Tamil", "Marlett", "MS outlook",
            "MS office Symbol Semilight", "MS office symbol regular",
            "Ms office symbol extralight", "Ms Reference speciality", "Symbol",
            "Wingdings", "Souliyo Unicode", "Aguafina Script", "Yantramanav Black"
        ])
        
        # Optimized file filtering with list comprehension and early font check
        step_extensions = ["*.stp", "*.step", "*.STP", "*.STEP"]
        step_files = []
        invalid = []
        for ext in step_extensions:
            for f in input_dir.glob(ext):
                try:
                    font_name = f.stem.split("_")[1]
                    if font_name not in INVALID_FONTS:
                        step_files.append(f)
                    else:
                        invalid.append(f.name)
                except IndexError:
                    # Skip files that don't follow expected naming convention
                    continue
        if invalid:
            print(f"Found {len(step_files)} valid files after filtering")
        file_output_dirs = [output_dir] * len(step_files)

    else:
        # Generic dataset handling with optimized file discovery
        step_extensions = ["*.stp", "*.step", "*.STP", "*.STEP"]
        step_files = []
        for ext in step_extensions:
            step_files.extend(input_dir.glob(ext))
        
        print(f"Found {len(step_files)} STEP files")
        file_output_dirs = [output_dir] * len(step_files)
        
    if skip_existing:
        # Batch existence check for better performance
        if random_rotate_step and create_rotated_step_files:
            valid_files = []
            for f in step_files:
                bin_exists = (output_dir / (f.stem + "_rotated.bin")).exists()
                stp_exists = (output_dir.parent / (f.stem + "_rotated.stp")).exists()
                if not (bin_exists and stp_exists):
                    valid_files.append((f, output_dir))
            step_files, file_output_dirs = zip(*valid_files) if valid_files else ([], [])
        else:
            valid_files = [(f, output_dir) for f in step_files 
                            if not (output_dir / (f.stem + ".bin")).exists()]
            step_files, file_output_dirs = zip(*valid_files) if valid_files else ([], [])
    
    # Limit number of files if max_files is specified
    if max_files is not None and len(step_files) > max_files:
        print(f"Limiting processing to {max_files} files (out of {len(step_files)} available)")
        step_files = step_files[:max_files]
        if isinstance(file_output_dirs, (list, tuple)):
            file_output_dirs = file_output_dirs[:max_files]
    
    if not step_files:
        print("No new files to process.")
        return [], []

    results, error_files = [], []

    file_args = [
        (
            f,
            outdir,
            surf_num_u_samples,
            surf_num_v_samples,
            curv_num_u_samples,
            vision_grid_elev,
            vision_grid_azim,
            include_vision_grids,
            include_uv_face,
            include_uv_edge,
            include_face_attributes,
            include_edge_attributes,
            scale_body,
            random_rotate_step,
            create_rotated_step_files,
            parallel_level,
            segmentation,
            compress
        )
        for f, outdir in zip(step_files, file_output_dirs)
    ]

    if parallel_level == "file":
        if num_processes is None or num_processes <= 1:
            for args in tqdm(file_args, desc="Processing files", total=len(file_args)):
                result = process_file_with_output_dir(args)
                success, filename, error = result
                if not success:
                    error_files.append(filename)
        else:
            total_files = len(file_args)
            with multiprocessing.Pool(processes=num_processes) as pool:
                for result in tqdm(pool.imap_unordered(process_file_with_output_dir, file_args), total=total_files, desc="Processing files"):
                    success, filename, error = result
                    if not success:
                        error_files.append(filename)
    elif parallel_level == "face":
        for args in tqdm(file_args, desc="Processing files (face-level)", total=len(file_args)):
            result = process_file_with_output_dir(args)
            success, filename, error = result
            if not success:
                error_files.append(filename)

    return results, error_files

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Feature extraction for STEP files with vision grids")
    
    parser.add_argument("--dataset", type=str, default="solidletters",
                        help="Dataset name")
    parser.add_argument("--folder", type=str, default="graphs",
                        help="Output folder name")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Base data directory path (overrides auto-detection)")
    parser.add_argument("--all", action="store_true", default=True,
                        help="Process all splits (train, val, test) instead of just one")
    parser.add_argument("--seg", action="store_true", default=True,
                        help="Set to True if dataset is for segmentation (face-level labels)")
    parser.add_argument("--rotate", action="store_true", default=False,
                        help="Enable random rotation of STEP files")
    parser.add_argument("--split", type=str, default="train",
                        help="Data split to process (train, val, test)")
    parser.add_argument("--az", type=int, default=12,
                        help="Number of azimuthal divisions for vision grids")
    parser.add_argument("--el", type=int, default=6,
                        help="Number of elevation levels for vision grids")
    parser.add_argument("--parallel_level", type=str, default="file", choices=["file", "face"],
                        help="Parallelization level: 'file' (default) or 'face'")
    parser.add_argument("--max_files", "-n", type=int, default=None,
                        help="Maximum number of files to process (default: process all files)")
    parser.add_argument("--num_processes", type=int, default=None,
                        help="Number of parallel processes (default: min(22, cpu_count))")
    parser.add_argument("--uv_samples", type=int, default=10,
                        help="Number of UV grid samples (default: 10)")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip files that already exist")
    parser.add_argument("--no_compress", action="store_true", default=False,
                        help="Disable float16 compression")
    parser.add_argument("--edge_info", action="store_true", default=False,
                        help="Include edge UV grids and edge attributes")
    parser.add_argument("--curv_samples", type=int, default=10,
                        help="Number of curve samples for edges (default: 10)")
    
    return parser.parse_args()


def main():
    """Main execution function for processing STEP files with vision grids."""
    args = parse_args()
    
    # Configuration from args
    num_processes = args.num_processes or min(22, multiprocessing.cpu_count())
    splits_to_run = ["train", "val", "test"] if args.all else [args.split]
    
    print(f"[INFO] Dataset: {args.dataset}, Splits: {splits_to_run}")
    print(f"[INFO] Processes: {num_processes}, Segmentation: {args.seg}")
    print(f"[INFO] Edge info: {args.edge_info}")
    if args.max_files:
        print(f"[INFO] Max files: {args.max_files}")

    # Resolve data directory
    base_data_dir = None
    if args.data_dir:
        base_data_dir = pathlib.Path(args.data_dir)
    else:
        # Search relative to script
        script_dir = pathlib.Path(__file__).parent
        base_data_dir = (script_dir / "../data").resolve()

    if not base_data_dir or not base_data_dir.exists():
        print(f"[ERROR] Data directory not found. Use --data_dir.")
        return
    
    print(f"[INFO] Data directory: {base_data_dir}")

    # Process splits
    for split in splits_to_run:
        step_path = base_data_dir / args.dataset / split
        
        if args.rotate:
            rotated_split = split + "_rotated"
            graph_path = base_data_dir / args.dataset / rotated_split / args.folder
            path = step_path
        else:
            graph_path = base_data_dir / args.dataset / split / args.folder
            path = step_path

        graph_path.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Processing split: {split}")

        try:
            results, errors = process_multiple_files(
                path,
                graph_path,
                args.uv_samples,
                args.uv_samples,
                args.curv_samples,
                args.el,
                args.az,
                include_vision_grids=True,
                include_uv_face=True,
                include_uv_edge=args.edge_info,
                include_face_attributes=True,
                include_edge_attributes=args.edge_info,
                scale_body=True,
                num_processes=num_processes,
                skip_existing=args.skip_existing,
                dataset=args.dataset,
                random_rotate_step=args.rotate,
                create_rotated_step_files=True,
                parallel_level=args.parallel_level,
                segmentation=args.seg,
                compress=not args.no_compress,
                max_files=args.max_files
            )
            
            print(f"[INFO] Split '{split}' complete. Errors: {len(errors)}")
            if errors:
                print(f"[WARNING] Failed: {errors[:10]}{'...' if len(errors) > 10 else ''}")
                
        except KeyboardInterrupt:
            print(f"[INFO] Interrupted during split '{split}'")
            break
        except Exception as e:
            print(f"[ERROR] Split '{split}' failed: {e}")
            continue
            
    print("[INFO] Processing complete.")

if __name__ == "__main__":
    main()