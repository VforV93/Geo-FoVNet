"""
Visualization script for ray casting
"""

# Standard library
import random
import logging
from matplotlib.colors import LinearSegmentedColormap

# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Wedge
from matplotlib.cm import ScalarMappable
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

# Suppress matplotlib debug logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# OpenCASCADE imports
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Vec, gp_Ax2
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Display.SimpleGui import init_display
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeCone
from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
from OCC.Core.AIS import AIS_Shape
from OCC.Core.Graphic3d import Graphic3d_MaterialAspect, Graphic3d_NOM_PLASTIC

from preprocessing.ray_casting import raycast_hemisphere, load_step_file
from preprocessing.geometry_features import compute_local_frame

MATERIAL = Graphic3d_NOM_PLASTIC

def get_face_signature(face):
    """
    Create a deterministic signature for a face based on its geometric properties.
    This helps ensure the same face gets the same identifier across runs.
    """
    try:
        # Get the local frame to extract geometric properties
        center, axes, orig_center = compute_local_frame(face)
        
        # Create a signature based on center coordinates and normal direction
        # Round to reduce floating point precision issues
        center_rounded = np.round(center, decimals=6)
        normal_rounded = np.round(axes[2], decimals=6)  # Z-axis is the normal
        
        # Create a hash-like signature
        signature = hash((tuple(center_rounded), tuple(normal_rounded)))
        return signature
    except:
        # Fallback to object id if geometric analysis fails
        return id(face)

def sort_faces_deterministically(faces):
    """Sort faces by their geometric signature for consistent ordering."""
    face_signatures = [(get_face_signature(face), i, face) for i, face in enumerate(faces)]
    face_signatures.sort(key=lambda x: x[0])  # Sort by signature
    
    # Return sorted faces and a mapping from new index to original index
    sorted_faces = [item[2] for item in face_signatures]
    index_mapping = {new_idx: item[1] for new_idx, item in enumerate(face_signatures)}
    
    return sorted_faces, index_mapping

def hex_to_rgb01(hex_str):
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return (r, g, b)

def visualize_grids(occupancy_grid: np.ndarray, distance_grid: np.ndarray, 
                   dot_grid: np.ndarray = None,
                   dot_grid_opposite: np.ndarray = None,
                   hit_color_hex=None, inner_hit_color_hex1=None, inner_hit_color_hex2=None):
    """Visualize the hemisphere grids and their circular projections side by side."""
    def plot_side_by_side(outer_grid, outer_circ, inner_grid, inner_circ, num_elev, num_azim):
        # Use higher DPI and tight layout for higher quality plots
        fig, axs = plt.subplots(2, 4, figsize=(12, 12), dpi=100)
        fig.patch.set_facecolor('none')

        # Custom colormaps for grids
        def light_color(rgb, lightness=0.6):
            return tuple(lightness * 1.0 + (1.0 - lightness) * c for c in rgb)

        outer_rgb = hex_to_rgb01(hit_color_hex) if hit_color_hex else (1, 0, 0)
        blue_rgb = (0, 0, 1)
        miss_rgb = hex_to_rgb01(MISS_COLOR_HEX) if 'MISS_COLOR_HEX' in globals() and MISS_COLOR_HEX else (0.6, 0.3, 0)
        light_outer_rgb = light_color(outer_rgb, lightness=0.6)
        inner_hit_color_rgb1 = hex_to_rgb01(inner_hit_color_hex1) if inner_hit_color_hex1 else (1, 0, 0)
        inner_hit_color_rgb2 = hex_to_rgb01(inner_hit_color_hex2) if inner_hit_color_hex2 else (0, 1, 0)
        outer_cmap = LinearSegmentedColormap.from_list('outer_cmap', [light_outer_rgb, outer_rgb])
        inner_cmap = LinearSegmentedColormap.from_list('inner_cmap', [inner_hit_color_rgb1, inner_hit_color_rgb2])
        # Custom diverging colormap: -1 blue, 0 black, 1 red
        polar_cmap = LinearSegmentedColormap.from_list(
            "custom_polar",
            [
                (0.0, "#0000ff"),   # blue at -1
                (0.5, "#000000"),   # black at 0
                (1.0, "#ff0000"),   # red at 1
            ]
        )

        # Outer 2D grid (distance)
        if outer_grid is not None:
            vmin_outer = np.min(outer_grid)
            vmax_outer = np.max(outer_grid)
            # Build a custom RGB(A) image so that distance==0 uses miss_rgb
            outer_img = np.empty(outer_grid.shape + (3,))
            for i in range(outer_grid.shape[0]):
                for j in range(outer_grid.shape[1]):
                    val = outer_grid[i, j]
                    if val == 0:
                        outer_img[i, j] = miss_rgb
                    else:
                        norm_val = (val - vmin_outer) / (vmax_outer - vmin_outer) if vmax_outer > vmin_outer else 1.0
                        outer_img[i, j] = outer_cmap(norm_val)[:3]
            axs[0, 0].imshow(outer_img, aspect='equal', origin='upper', vmin=0, vmax=1)
            axs[0, 0].set_title('Outer Vision Grid', fontsize=12, fontweight='bold')
            axs[0, 0].set_xlabel('Azimuth', fontsize=10)
            axs[0, 0].set_ylabel('Elevation', fontsize=10)
            axs[0, 0].set_xticks(np.arange(num_azim) + 0.5)
            axs[0, 0].set_yticks(np.arange(num_elev) + 0.5)
            axs[0, 0].set_xticklabels([str(i) for i in range(num_azim)], rotation=0, fontsize=11)
            axs[0, 0].set_yticklabels([str(i) for i in range(num_elev)], rotation=0, fontsize=11)
        else:
            axs[0, 0].axis('off')

        # Outer 2D grid (dot product) - use polar_cmap, vmin=-1, vmax=1
        if dot_grid is not None:
            dot_img = np.empty(dot_grid.shape + (4,))
            for i in range(dot_grid.shape[0]):
                for j in range(dot_grid.shape[1]):
                    val = dot_grid[i, j]
                    dot_img[i, j] = polar_cmap((val + 1) / 2)
            axs[0, 2].imshow(dot_img, aspect='equal', origin='upper', vmin=-1, vmax=1)
            axs[0, 2].set_title('Outer 2D Dot Product Grid', fontsize=12, fontweight='bold')
            axs[0, 2].set_xlabel('Azimuth', fontsize=10)
            axs[0, 2].set_ylabel('Elevation', fontsize=10)
            axs[0, 2].set_xticks(np.arange(num_azim))
            axs[0, 2].set_yticks(np.arange(num_elev))
            axs[0, 2].set_xticklabels([str(i) for i in range(num_azim)], rotation=0, fontsize=11)
            axs[0, 2].set_yticklabels([str(i) for i in range(num_elev)], rotation=0, fontsize=11)
        else:
            axs[0, 2].axis('off')

        # Inner 2D grid (distance)
        if inner_grid is not None:
            vmin_inner = np.min(inner_grid)
            vmax_inner = np.max(inner_grid)
            sns.heatmap(inner_grid, ax=axs[0, 1], cmap=inner_cmap, cbar=False, square=True, vmin=vmin_inner, vmax=vmax_inner,
                        xticklabels=True, yticklabels=True)
            axs[0, 1].set_title('Inner Vision Grid', fontsize=12, fontweight='bold')
            axs[0, 1].set_xlabel('Azimuth', fontsize=10)
            axs[0, 1].set_ylabel('Elevation', fontsize=10)
            axs[0, 1].set_xticks(np.arange(num_azim) + 0.5)
            axs[0, 1].set_yticks(np.arange(num_elev) + 0.5)
            axs[0, 1].set_xticklabels([str(i) for i in range(num_azim)], rotation=0, fontsize=11)
            axs[0, 1].set_yticklabels([str(i) for i in range(num_elev)], rotation=0, fontsize=11)
        else:
            axs[0, 1].axis('off')

        # Inner 2D grid (dot product) - use polar_cmap, vmin=-1, vmax=1
        if dot_grid_opposite is not None:
            dot_img = np.empty(dot_grid_opposite.shape + (4,))
            for i in range(dot_grid_opposite.shape[0]):
                for j in range(dot_grid_opposite.shape[1]):
                    val = dot_grid_opposite[i, j]
                    dot_img[i, j] = polar_cmap((val + 1) / 2)
            axs[0, 3].imshow(dot_img, aspect='equal', origin='upper', vmin=-1, vmax=1)
            axs[0, 3].set_title('Inner 2D Dot Product Grid', fontsize=12, fontweight='bold')
            axs[0, 3].set_xlabel('Azimuth', fontsize=10)
            axs[0, 3].set_ylabel('Elevation', fontsize=10)
            axs[0, 3].set_xticks(np.arange(num_azim))
            axs[0, 3].set_yticks(np.arange(num_elev))
            axs[0, 3].set_xticklabels([str(i) for i in range(num_azim)], rotation=0, fontsize=11)
            axs[0, 3].set_yticklabels([str(i) for i in range(num_elev)], rotation=0, fontsize=11)
        else:
            axs[0, 3].axis('off')


        # Circular Distance Grid - Outer
        ax = axs[1, 0]
        if outer_circ is not None:
            ax.set_facecolor('none')
            ax.set_title('Outer Circular Distance Grid', fontsize=12, fontweight='bold', pad=20)
            ax.axis('equal')
            ax.axis('off')
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            shrink_factor = 0.9
            radius = 0.5 * shrink_factor
            center_xy = (0, 0)
            vmin_outer = np.min(outer_grid) if outer_grid is not None else 0
            vmax_outer = np.max(outer_grid) if outer_grid is not None else 1
            for elev in range(num_elev):
                r0 = radius * elev / num_elev
                r1 = radius * (elev + 1) / num_elev
                for az in range(num_azim):
                    theta0 = 360 * az / num_azim
                    theta1 = 360 * (az + 1) / num_azim
                    val = outer_grid[elev, az]
                    if val == 0:
                        color = (*miss_rgb, 1)
                    else:
                        norm_val = (val - vmin_outer) / (vmax_outer - vmin_outer) if vmax_outer > vmin_outer else 1.0
                        color = outer_cmap(norm_val)
                    wedge = Wedge(center_xy, r1, theta0, theta1, width=r1 - r0, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            circle = Circle(center_xy, radius, fill=False, color='#333', linewidth=1.5)
            ax.add_patch(circle)
        else:
            ax.axis('off')

        # Circular Dot Product Grid - Outer (use polar_cmap, NO normalization, assume already in [-1,1])
        ax = axs[1, 2]
        if dot_grid_circ is not None:
            ax.set_facecolor('none')
            ax.set_title('Outer Circular Dot Product Grid', fontsize=12, fontweight='bold', pad=20)
            ax.axis('equal')
            ax.axis('off')
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            shrink_factor = 0.9
            radius = 0.5 * shrink_factor
            center_xy = (0, 0)
            for elev in range(num_elev):
                r0 = radius * elev / num_elev
                r1 = radius * (elev + 1) / num_elev
                for az in range(num_azim):
                    theta0 = 360 * az / num_azim
                    theta1 = 360 * (az + 1) / num_azim
                    val = dot_grid[elev, az]
                    color = polar_cmap((val + 1) / 2)
                    wedge = Wedge(center_xy, r1, theta0, theta1, width=r1 - r0, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            circle = Circle(center_xy, radius, fill=False, color='#333', linewidth=1.5)
            ax.add_patch(circle)
        else:
            ax.axis('off')

        # Circular Distance Grid - Inner
        ax = axs[1, 1]
        if inner_circ is not None:
            ax.set_facecolor('none')
            ax.set_title('Inner Circular Distance Grid', fontsize=12, fontweight='bold', pad=20)
            ax.axis('equal')
            ax.axis('off')
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            shrink_factor = 0.9
            radius = 0.5 * shrink_factor
            center_xy = (0, 0)
            vmin_inner = np.min(inner_grid) if inner_grid is not None else 0
            vmax_inner = np.max(inner_grid) if inner_grid is not None else 1
            for elev in range(num_elev):
                r0 = radius * elev / num_elev
                r1 = radius * (elev + 1) / num_elev
                for az in range(num_azim):
                    theta0 = 360 * az / num_azim
                    theta1 = 360 * (az + 1) / num_azim
                    val = inner_grid[elev, az]
                    norm_val = (val - vmin_inner) / (vmax_inner - vmin_inner) if vmax_inner > vmin_inner else 1.0
                    color = inner_cmap(norm_val)
                    wedge = Wedge(center_xy, r1, theta0, theta1, width=r1 - r0, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            circle = Circle(center_xy, radius, fill=False, color='#333', linewidth=1.5)
            ax.add_patch(circle)
        else:
            ax.axis('off')

        # Circular Dot Product Grid - Inner (use polar_cmap, NO normalization, assume already in [-1,1])
        ax = axs[1, 3]
        if dot_grid_opposite_circ is not None:
            ax.set_facecolor('none')
            ax.set_title('Inner Circular Dot Product Grid', fontsize=12, fontweight='bold', pad=20)
            ax.axis('equal')
            ax.axis('off')
            ax.set_xlim(-0.55, 0.55)
            ax.set_ylim(-0.55, 0.55)
            shrink_factor = 0.9
            radius = 0.5 * shrink_factor
            center_xy = (0, 0)
            for elev in range(num_elev):
                r0 = radius * elev / num_elev
                r1 = radius * (elev + 1) / num_elev
                for az in range(num_azim):
                    theta0 = 360 * az / num_azim
                    theta1 = 360 * (az + 1) / num_azim
                    val = dot_grid_opposite[elev, az]
                    color = polar_cmap((val + 1) / 2)
                    wedge = Wedge(center_xy, r1, theta0, theta1, width=r1 - r0, facecolor=color, edgecolor='none')
                    ax.add_patch(wedge)
            circle = Circle(center_xy, radius, fill=False, color='#333', linewidth=1.5)
            ax.add_patch(circle)
        else:
            ax.axis('off')

        # Colorbars
        sm_outer = ScalarMappable(cmap=outer_cmap, norm=plt.Normalize(vmin=vmin_outer, vmax=vmax_outer)) if outer_grid is not None else None
        sm_inner = ScalarMappable(cmap=inner_cmap, norm=plt.Normalize(vmin=vmin_inner, vmax=vmax_inner)) if inner_grid is not None else None
        # Dot product colorbars: use polar_cmap, NO normalization (norm=None)
        from matplotlib.colors import Normalize
        sm_dot = ScalarMappable(cmap=polar_cmap, norm=Normalize(vmin=-1, vmax=1)) if dot_grid is not None else None
        sm_dot_op = ScalarMappable(cmap=polar_cmap, norm=Normalize(vmin=-1, vmax=1)) if dot_grid_opposite is not None else None
        fig.subplots_adjust(right=0.97, wspace=0.18, hspace=0.18, bottom=0.18)

        # Colorbars: align horizontally under each column in the same order as the graphs
        # [0,0] Outer Distance, [0,1] Inner Distance, [0,2] Outer Dot, [0,3] Inner Dot
        # Get the bounding boxes of the axes to align colorbars
        fig.canvas.draw()  # ensure layout is computed
        cbar_h = 0.025
        cbar_y = 0.08
        for col, (sm, label, ticks, ticklabels) in enumerate([
            (sm_outer, 'Outer Distance', None, None),
            (sm_inner, 'Inner Distance', None, None),
            (sm_dot, 'Outer Dot Product', [-1, 0, 1], ['-1 (Blue)', '0 (Black)', '1 (Red)']),
            (sm_dot_op, 'Inner Dot Product', [-1, 0, 1], ['-1 (Blue)', '0 (Black)', '1 (Red)'])
        ]):
            if sm is not None:
                # Get the position of the top axes in this column
                ax = axs[0, col]
                bbox = ax.get_position()
                cbar_x = bbox.x0
                cbar_w = bbox.width
                cbar_ax = fig.add_axes([cbar_x, cbar_y, cbar_w, cbar_h])
                if ticks is not None:
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', ticks=ticks)
                    if ticklabels is not None:
                        cbar.ax.set_xticklabels(ticklabels)
                    # For dot product colorbars, show the color gradient from blue (-1) to black (0) to red (1)
                    cbar.ax.tick_params(labelsize=11)
                else:
                    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
                cbar.set_label(label, fontsize=12)
        plt.show()

    # This function now expects both outer and inner grids
    # (outer_occupancy, outer_distance, outer_circular, inner_distance, inner_circular)
    return plot_side_by_side


# ============================================================================
# VISUALIZATION
# ============================================================================
def make_arrow_shapes(start, vec, shaft_radius=5, head_radius=5, head_length=5):
    """Create arrow shapes (shaft and head) for a given vector."""
    # If head_radius or head_length are None, scale them by shaft_radius
    if head_radius is None:
        head_radius = 3 * shaft_radius
    if head_length is None:
        head_length = 6 * shaft_radius
    length = vec.Magnitude()
    if length < head_length * 1.2:
        # Too short for arrow, just draw a cylinder
        dir = gp_Dir(vec)
        shaft_ax = gp_Ax2(start, dir)
        shaft = BRepPrimAPI_MakeCylinder(shaft_ax, shaft_radius, length).Shape()
        return [shaft]
    dir = gp_Dir(vec)
    shaft_len = length - head_length
    shaft_ax = gp_Ax2(start, dir)
    shaft = BRepPrimAPI_MakeCylinder(shaft_ax, shaft_radius, shaft_len).Shape()
    head_start = gp_Pnt(start.XYZ() + dir.XYZ() * shaft_len)
    head_ax = gp_Ax2(head_start, dir)
    head = BRepPrimAPI_MakeCone(head_ax, head_radius, 0, head_length).Shape()
    return [shaft, head]

def display_arrow(display, start, end, color, shaft_radius=5, head_radius=5, head_length=5, transparency=None):
    """Display an arrow from start to end with the given color and optional transparency."""
    start = np.asarray(start, dtype=np.float64).reshape(3)
    end = np.asarray(end, dtype=np.float64).reshape(3)
    start_pnt = gp_Pnt(*start)
    end_pnt = gp_Pnt(*end)
    vec = gp_Vec(start_pnt, end_pnt)
    for part in make_arrow_shapes(gp_Pnt(*start), vec, shaft_radius, head_radius, head_length):
        ais = AIS_Shape(part)
        material_aspect = Graphic3d_MaterialAspect(Graphic3d_NOM_PLASTIC)
        ais.SetMaterial(material_aspect)
        ais.SetColor(color)
        if transparency is not None:
            ais.SetTransparency(transparency)
        display.Context.Display(ais, False)

def hex_to_Quantity_Color(hex_str):
    """Convert a hex color string (e.g. 'B45253' or '#B45253') to Quantity_Color (RGB, 0-1)."""
    hex_str = hex_str.lstrip('#')
    if len(hex_str) != 6:
        raise ValueError(f"Invalid hex color: {hex_str}")
    r = int(hex_str[0:2], 16) / 255.0
    g = int(hex_str[2:4], 16) / 255.0
    b = int(hex_str[4:6], 16) / 255.0
    return Quantity_Color(r, g, b, Quantity_TOC_RGB)

def visualize_rays(
    shape, center: np.ndarray, hits: list, 
    show_only_hits: bool = False, hits_opposite: list = None, model_transparency=0.0,
    hitpoints=None, normals=None, hitpoints_opposite=None, normals_opposite=None, show_normals=False,
    shaft_frac=0.01, head_frac=0.03, head_len_frac=0.06, unhit_frac=0.5, normal_frac=0.01,
    orig_center=None, miss_transparency=None,
    hit_color_hex=None, miss_color_hex=None, blue_hex=None, purple_hex=None,
    design_color_hex=None,
    face_color_hex=None,
    face_under_consideration=None,
    show_outer=True,
    show_inner=True,
    inner_hit_color_hex1=None,
    inner_hit_color_hex2=None,
    gradient: bool = True,
    dummy_gradient: bool = False,
):
    """
    Visualize ray casting results in 3D. Arrow sizes can be set as fractions of bbox diagonal. Colors can be specified as hex strings.
    
    Args:
        ...existing code...
        gradient (bool): If True, make the outer hits a gradient between hit and miss color. Else just use the color for a hit and miss respectively.
    """
    display, start, _, _ = init_display(backend_str="qt-pyqt5", background_gradient_color1=[255, 255, 255], background_gradient_color2=[255, 255, 255])
    # Use user-specified design color if provided
    model_color = hex_to_Quantity_Color(design_color_hex) if design_color_hex else Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)
    face_color = hex_to_Quantity_Color(face_color_hex) if face_color_hex else model_color  # default: orange
    display.DisplayShape(shape, update=True, material=Graphic3d_NOM_PLASTIC, color=model_color, transparency=model_transparency)

    
    if face_under_consideration is not None:
        display.DisplayShape(face_under_consideration, update=False, material=Graphic3d_NOM_PLASTIC, color=face_color, transparency=model_transparency)

    # Use user-specified colors if provided, else defaults
    hit_color = hex_to_Quantity_Color(hit_color_hex) if hit_color_hex else Quantity_Color(1, 0, 0, Quantity_TOC_RGB)
    miss_color = hex_to_Quantity_Color(miss_color_hex) if miss_color_hex else Quantity_Color(0.6, 0.3, 0, Quantity_TOC_RGB)
    purple = hex_to_Quantity_Color(purple_hex) if purple_hex else Quantity_Color(0.6, 0, 0.6, Quantity_TOC_RGB)
    # Inner hit color gradient setup
    if inner_hit_color_hex1:
        inner_hit_rgb1 = hex_to_rgb01(inner_hit_color_hex1)
    else:
        inner_hit_rgb1 = (0, 0, 1)  # default blue
    if inner_hit_color_hex2:
        inner_hit_rgb2 = hex_to_rgb01(inner_hit_color_hex2)
    else:
        inner_hit_rgb2 = (0.6, 0, 0.6)  # default purple

    # Compute bounding box diagonal for scaling
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bbox_diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    shaft_radius = shaft_frac * bbox_diag
    head_radius = head_frac * bbox_diag
    head_length = head_len_frac * bbox_diag
    unhit_ray_length = unhit_frac * bbox_diag
    # Compute min/max for distance grid for gradient
    distances = [dist for _, dist, _ in hits if dist is not None and dist != 0]
    vmin_outer = min(distances) if distances else 0
    vmax_outer = max(distances) if distances else 1
    # Define a lighter version of the main color for the gradient base (e.g., 80% main color + 20% white)
    def light_color(rgb, lightness=0.6):
        return tuple(lightness * 1.0 + (1.0 - lightness) * c for c in rgb)

    light_hit_rgb = light_color(hex_to_rgb01(hit_color_hex) if hit_color_hex else (1, 0, 0), lightness=0.6)
    light_blue_rgb = light_color(inner_hit_rgb1, lightness=0.6)

    # --- Draw U and V direction arrows ---
    if face_under_consideration is not None:
        # Compute local frame for the face
        center, axes, orig_center = compute_local_frame(face_under_consideration)
        # U direction: axes[0], V direction: axes[1]
        u_color = Quantity_Color(0, 0.7, 0, Quantity_TOC_RGB)  # green
        v_color = Quantity_Color(0, 0, 1, Quantity_TOC_RGB)    # blue
        n_color = Quantity_Color(1, 0, 0, Quantity_TOC_RGB)    # red
        arrow_len = 0.025 * bbox_diag
        display_arrow(display, orig_center, orig_center + axes[:,0] * arrow_len, u_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)
        display_arrow(display, orig_center, orig_center + axes[:,1] * arrow_len, v_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)
        display_arrow(display, orig_center, orig_center + axes[:,2] * arrow_len, n_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)
    if show_outer:
        for idx, (d, dist, hit_face) in enumerate(hits):
            if show_only_hits and (dist is None or dist == 0):
                continue
            ray_len = dist if dist else unhit_ray_length
            end_pt = center + d * ray_len
            transp = 0
            if dummy_gradient:
                color = Quantity_Color(*color_grid[idx], Quantity_TOC_RGB)
            elif dist and dist != 0:
                if gradient:
                    # Gradient between light_hit_rgb and hit_color
                    norm_val = (dist - vmin_outer) / (vmax_outer - vmin_outer) if vmax_outer > vmin_outer else 1.0
                    hit_rgb = hex_to_rgb01(hit_color_hex) if hit_color_hex else (1, 0, 0)
                    interp_rgb = tuple((1.0 - norm_val) * l + norm_val * c for l, c in zip(light_hit_rgb, hit_rgb))
                    color = Quantity_Color(*interp_rgb, Quantity_TOC_RGB)
                else:
                    # Use hit color directly
                    color = hit_color
                # display.DisplayShape(gp_Pnt(*end_pt), update=False, color=Quantity_Color(1, 1, 0, Quantity_TOC_RGB))
            else:
                color = miss_color
                transp = miss_transparency
            display_arrow(display, center, end_pt, color, shaft_radius, head_radius, head_length, transparency=transp)

    if show_inner and hits_opposite is not None:
        distances_op = [dist for _, dist, _ in hits_opposite if dist is not None and dist != 0]
        vmin_inner = min(distances_op) if distances_op else 0
        vmax_inner = max(distances_op) if distances_op else 1
        for d, dist, hit_face in hits_opposite:
            if show_only_hits and (dist is None or dist == 0):
                continue
            ray_len = dist if dist else unhit_ray_length
            end_pt = center + d * ray_len
            # For opposite hemisphere: distance==0 is purple, else interpolate between two user-specified colors
            if dist and dist != 0:
                norm_val = (dist - vmin_inner) / (vmax_inner - vmin_inner) if vmax_inner > vmin_inner else 1.0
                interp_rgb = tuple((1.0 - norm_val) * c1 + norm_val * c2 for c1, c2 in zip(inner_hit_rgb1, inner_hit_rgb2))
                color = Quantity_Color(*interp_rgb, Quantity_TOC_RGB)
                display_arrow(display, center, end_pt, color, shaft_radius, head_radius, head_length)
            else:
                display_arrow(display, center, end_pt, purple, shaft_radius, head_radius, head_length)

    if show_normals and hitpoints is not None and normals is not None:
        normal_color = Quantity_Color(0, 0, 0, Quantity_TOC_RGB)  # black
        normal_len = normal_frac
        num_elev, num_azim = hitpoints.shape[0], hitpoints.shape[1]
        for elev in range(num_elev):
            for azim in range(num_azim):
                pt = hitpoints[elev, azim]
                n = normals[elev, azim]
                if pt is not None and n is not None and np.any(pt):
                    end_pt = pt + n * normal_len
                    display_arrow(display, pt, end_pt, normal_color, shaft_radius*0.7, head_radius*0.7, head_length*0.7)
    if show_normals and hitpoints_opposite is not None and normals_opposite is not None:
        normal_color = Quantity_Color(0.6, 0, 0.6, Quantity_TOC_RGB)  # purple
        normal_len = normal_frac
        num_elev, num_azim = hitpoints_opposite.shape[0], hitpoints_opposite.shape[1]
        for elev in range(num_elev):
            for azim in range(num_azim):
                pt = hitpoints_opposite[elev, azim]
                n = normals_opposite[elev, azim]
                if pt is not None and n is not None and np.any(pt):
                    end_pt = pt + n * normal_len
                    display_arrow(display, pt, end_pt, normal_color, shaft_radius*0.7, head_radius*0.7, head_length*0.7)
    start()


def visualize_rays_all_faces(
    shape, faces, num_elev: int = 12, num_azim: int = 12, show_only_hits: bool = False, model_transparency: float = 0.0,
    show_outer: bool = True, show_inner: bool = True,
    shaft_frac: float = 0.01, head_frac: float = 0.03, head_len_frac: float = 0.06, unhit_frac: float = 0.5,
    hit_color_hex: str = None, miss_color_hex: str = None,
    inner_hit_color_hex1: str = None, inner_hit_color_hex2: str = None,
    gradient: bool = True, miss_transparency: float = 0.0, only_faces_with_hits: bool = False,
    design_color_hex: str = None
):
    """Visualize ray casting from all faces of a shape, with toggles for outer/inner hemisphere rays. Arrow sizes can be set as fractions of bbox diagonal."""
    bbox = Bnd_Box()
    brepbndlib.Add(shape, bbox)
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    bbox_diag = np.linalg.norm([xmax - xmin, ymax - ymin, zmax - zmin])
    shaft_radius = shaft_frac * bbox_diag
    head_radius = head_frac * bbox_diag
    head_length = head_len_frac * bbox_diag
    unhit_ray_length = unhit_frac * bbox_diag
    # Use user-specified design color if provided
    model_color = hex_to_Quantity_Color(design_color_hex) if design_color_hex else Quantity_Color(0.1, 0.1, 0.1, Quantity_TOC_RGB)

    display, start, _, _ = init_display(backend_str="qt-pyqt5", background_gradient_color1=[255, 255, 255], background_gradient_color2=[255, 255, 255])
    display.DisplayShape(shape, update=True, material = Graphic3d_NOM_PLASTIC, color=model_color, transparency=model_transparency)

    # --- Color Setup ---
    hit_color = hex_to_Quantity_Color(hit_color_hex) if hit_color_hex else Quantity_Color(1, 0, 0, Quantity_TOC_RGB)
    miss_color = hex_to_Quantity_Color(miss_color_hex) if miss_color_hex else Quantity_Color(0.2, 0.2, 0.2, Quantity_TOC_RGB)
    purple = hex_to_Quantity_Color("960096") # Default for inner misses, can be customized if needed

    # Inner hit color gradient setup
    inner_hit_rgb1 = hex_to_rgb01(inner_hit_color_hex1) if inner_hit_color_hex1 else (0, 0, 1)  # default blue
    inner_hit_rgb2 = hex_to_rgb01(inner_hit_color_hex2) if inner_hit_color_hex2 else (0.6, 0, 0.6)  # default purple

    # Outer hit gradient setup
    def light_color(rgb, lightness=0.6):
        return tuple(lightness * 1.0 + (1.0 - lightness) * c for c in rgb)
    light_hit_rgb = light_color(hex_to_rgb01(hit_color_hex) if hit_color_hex else (1, 0, 0), lightness=0.6)
    hit_rgb = hex_to_rgb01(hit_color_hex) if hit_color_hex else (1, 0, 0)


    for face in faces:
        center, axes, orig_center = compute_local_frame(face)

        # --- Draw U and V direction arrows for each face ---
        u_color = Quantity_Color(0, 0.7, 0, Quantity_TOC_RGB)  # green
        v_color = Quantity_Color(0, 0, 1, Quantity_TOC_RGB)    # blue
        n_color = Quantity_Color(1, 0, 0, Quantity_TOC_RGB)    # red
        arrow_len = 0.025 * bbox_diag
        print(axes.shape)
        display_arrow(display, center, center + axes[:,0] * arrow_len, u_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)
        display_arrow(display, center, center + axes[:,1] * arrow_len, v_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)
        display_arrow(display, center, center + axes[:,2] * arrow_len, n_color, shaft_radius*1.2, head_radius*1.2, head_length*1.2)

        center, hits, features = raycast_hemisphere(shape, center=center, axes=axes, num_elev=num_elev, num_azim=num_azim)
        if only_faces_with_hits:
            has_outer_hit = any((dist is not None and dist != 0) for _, dist, _ in hits)
            if not has_outer_hit:
                continue

        # --- Outer Hemisphere Visualization ---
        if show_outer:
            distances = [dist for _, dist, _ in hits if dist is not None and dist != 0]
            vmin_outer = min(distances) if distances else 0
            vmax_outer = max(distances) if distances else 1
            for d, dist, hit_face in hits:
                if show_only_hits and (dist is None or dist == 0):
                    continue
                ray_len = dist if dist else unhit_ray_length
                end_pt = center + d * ray_len
                transp = 0
                if dist and dist != 0:
                    if gradient:
                        norm_val = (dist - vmin_outer) / (vmax_outer - vmin_outer) if vmax_outer > vmin_outer else 1.0
                        interp_rgb = tuple((1.0 - norm_val) * l + norm_val * c for l, c in zip(light_hit_rgb, hit_rgb))
                        color = Quantity_Color(*interp_rgb, Quantity_TOC_RGB)
                    else:
                        color = hit_color
                else:
                    color = miss_color
                    transp = miss_transparency
                
                display_arrow(display, center, end_pt, color, shaft_radius, head_radius, head_length, transparency=transp)

        # --- Inner Hemisphere Visualization ---
        hits_opposite = features.get('hits_opposite', None)
        if show_inner and hits_opposite is not None:
            distances_op = [dist for _, dist, _ in hits_opposite if dist is not None and dist != 0]
            vmin_inner = min(distances_op) if distances_op else 0
            vmax_inner = max(distances_op) if distances_op else 1
            for d, dist, hit_face in hits_opposite:
                if show_only_hits and (dist is None or dist == 0):
                    continue
                ray_len = dist if dist else unhit_ray_length
                end_pt = center + d * ray_len
                if dist and dist != 0:
                    norm_val = (dist - vmin_inner) / (vmax_inner - vmin_inner) if vmax_inner > vmin_inner else 1.0
                    interp_rgb = tuple((1.0 - norm_val) * c1 + norm_val * c2 for c1, c2 in zip(inner_hit_rgb1, inner_hit_rgb2))
                    color = Quantity_Color(*interp_rgb, Quantity_TOC_RGB)
                    display_arrow(display, center, end_pt, color, shaft_radius, head_radius, head_length)
                else:
                    # Use purple for inner misses
                    display_arrow(display, center, end_pt, purple, shaft_radius, head_radius, head_length)

    start()

if __name__ == "__main__":
    # Gradient option for outer hits
    GRADIENT = False  # Set to False to disable gradient for outer hits
    DUMMY_GRADIENT = False  # Set to True to use dummy gradient colors for outer hits (for testing)
    # --- Config ---
    show_only_hits = True
    all_faces = True
    only_faces_with_hits = True  # visualize only faces with outer hits
    show_grids = False  # New option to visualize grids
    show_outer = False  # Show outer hemisphere rays
    show_inner = False  # Show inner hemisphere rays
    show_normals = False
    MISS_TRANSPARENCY = 0
    # Color hex values (set to None to use defaults)
    MISS_COLOR_HEX = "#79333F"    
    HIT_COLOR_HEX = "#3A000B"     
    DESIGN_COLOR_HEX = "#BDBCBC"  
    FACE_COLOR_HEX = None
    INNER_HIT_COLOR_HEX1 = "#0A312A" # "#46746B"
    INNER_HIT_COLOR_HEX2 = "#0A312A" 
    
    
    # Face selection: set to specific index (0-based) or None for random selection
    face_index = 11  
    shaft_frac = 0.002
    head_frac = 0.003
    head_len_frac = 0.02
    normal_frac = 0.6
    unhit_frac = 0.1
    num_elev, num_azim = 6,12
    model_transparency = 0.5

    filepath = "data/example/train/file_0.step" \

    shape = load_step_file(filepath)

    faces = list(TopologyExplorer(shape).faces())
    print(f"Shape has {len(faces)} faces (original order)")
    
    # Sort faces deterministically for consistent indexing
    faces_sorted, index_mapping = sort_faces_deterministically(faces)
    print(f"Faces sorted deterministically by geometric signature")
    
    if all_faces:
        visualize_rays_all_faces(
            shape, faces_sorted, num_elev=num_elev, num_azim=num_azim, show_only_hits=show_only_hits, 
            model_transparency=model_transparency,
            show_inner=show_inner, show_outer=show_outer,
            shaft_frac=shaft_frac, head_frac=head_frac, head_len_frac=head_len_frac, unhit_frac=unhit_frac,
            hit_color_hex=HIT_COLOR_HEX, miss_color_hex=MISS_COLOR_HEX,
            inner_hit_color_hex1=INNER_HIT_COLOR_HEX1, inner_hit_color_hex2=INNER_HIT_COLOR_HEX2,
            gradient=GRADIENT, miss_transparency=MISS_TRANSPARENCY, only_faces_with_hits=only_faces_with_hits,
            design_color_hex=DESIGN_COLOR_HEX
        )
    else:
        # Face selection logic
        selected_face = None
        selected_face_index = None
        original_face_index = None
        face_info = ""
        
        if face_index is not None:
            # Use specific face index (in sorted order)
            if 0 <= face_index < len(faces_sorted):
                selected_face = faces_sorted[face_index]
                selected_face_index = face_index
                original_face_index = index_mapping[face_index]
                face_info = f"Using specified face index {face_index} (original index: {original_face_index})"
                print(face_info)
            else:
                print(f"Warning: Face index {face_index} is out of range (0-{len(faces_sorted)-1}). Using random selection.")
                face_index = None
        
        if face_index is None:
            # Use random face selection (with optional constraint to find hits)
            if only_faces_with_hits:
                faces_indices = list(range(len(faces_sorted)))
                random.shuffle(faces_indices)
                found = False
                for i, face_idx in enumerate(faces_indices):
                    face = faces_sorted[face_idx]
                    center, axes, orig_center = compute_local_frame(face)
                    center, hits, features = raycast_hemisphere(shape, center=center, axes=axes, num_elev=num_elev, num_azim=num_azim, add_circular=True, compute_dot=True, return_hits=True)
                    if any((dist is not None and dist != 0) for _, dist, _ in hits):
                        selected_face = face
                        selected_face_index = face_idx
                        original_face_index = index_mapping[face_idx]
                        face_info = f"Found face with hits at index {face_idx} (original index: {original_face_index}, tried {i+1} faces)"
                        print(face_info)
                        found = True
                        break
                if not found:
                    raise RuntimeError("No faces with hits found.")
            else:
                selected_face_index = random.randint(0, len(faces_sorted) - 1)
                selected_face = faces_sorted[selected_face_index]
                original_face_index = index_mapping[selected_face_index]
                face_info = f"Using random face at index {selected_face_index} (original index: {original_face_index})"
                print(face_info)
        
        # Perform ray casting on selected face
        center, axes, orig_center = compute_local_frame(selected_face)
        center, hits, features = raycast_hemisphere(shape, center=center, axes=axes, num_elev=num_elev, num_azim=num_azim, add_circular=True, compute_dot=True, return_hits=True)
        print(f"Raycasting successful on face index {selected_face_index} (original index: {original_face_index})")
        visualize_rays(
            shape, center, hits, show_only_hits=show_only_hits,
            hits_opposite=features.get('hits_opposite'),
            model_transparency=model_transparency,
            hitpoints=features.get('hitpoints'),
            normals=features.get('normals'),
            hitpoints_opposite=features.get('hitpoints_opposite'),
            normals_opposite=features.get('normals_opposite'),
            show_normals=show_normals,
            shaft_frac=shaft_frac, head_frac=head_frac, head_len_frac=head_len_frac, unhit_frac=unhit_frac, normal_frac=normal_frac,
            orig_center=orig_center,
            miss_transparency=MISS_TRANSPARENCY,
            hit_color_hex=HIT_COLOR_HEX,
            miss_color_hex=MISS_COLOR_HEX,
            design_color_hex=DESIGN_COLOR_HEX,
            face_color_hex=FACE_COLOR_HEX,
            face_under_consideration=selected_face,
            show_outer=show_outer,
            show_inner=show_inner,
            inner_hit_color_hex1=INNER_HIT_COLOR_HEX1,
            inner_hit_color_hex2=INNER_HIT_COLOR_HEX2,
            gradient=GRADIENT,
            dummy_gradient=DUMMY_GRADIENT,  # if not gradient, use dummy gradient for testing
        )
        if show_grids:
            # Visualize both outer and inner hemisphere distance and dot product grids
            distance_grid_opposite = features.get('distance_grid_opposite')
            dot_grid = features.get('dot_grid')
            dot_grid_opposite = features.get('dot_grid_opposite')
            plot_grids = visualize_grids(
                features['occupancy_grid'], features['distance_grid'],
                features['circular_occupancy'], features['circular_distance'],
                dot_grid, dot_grid_opposite,
                hit_color_hex=HIT_COLOR_HEX, inner_hit_color_hex1=INNER_HIT_COLOR_HEX1, inner_hit_color_hex2=INNER_HIT_COLOR_HEX2
            )
            plot_grids(
                features['distance_grid'],
                distance_grid_opposite, 
                num_elev, num_azim
            )
