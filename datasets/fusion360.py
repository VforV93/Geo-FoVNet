"""Fusion360 Dataset: CAD models with face segmentation labels.

From the Fusion Gallery dataset (BRepNet, CVPR 2021).
Contains CAD models with 8-class face segmentation labels.
"""

import torch
import numpy as np

from datasets.base import BaseDataset


class Fusion360(BaseDataset):
    """Fusion360 dataset for B-Rep face segmentation (8 classes)."""
    
    @staticmethod
    def num_classes():
        return 8

    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None
    ):
        """Initialize Fusion360 dataset.
        
        References:
            BRepNet: A topological message passing system for solid models
            Lambourne et al., CVPR 2021
        
        Args:
            root_dir: Path to dataset directory
            center_and_scale: Whether to center and scale models
            random_rotate: Whether to apply random 3D rotations
            char2label: Not used (for API compatibility)
        """
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        
        # Locate segmentation label directory
        self.seg_path = self.path.parent.parent / "seg"
        
    def load_one_graph(self, file_path):
        """Load a single graph and attach per-face segmentation labels.
        
        Args:
            file_path: Path to .bin graph file
            
        Returns:
            dict: Sample with 'graph' and 'filename' keys (labels in graph.ndata['y'])
            None: If label count doesn't match face count
        """
        sample = super().load_one_graph(file_path)
        
        # Load segmentation labels from .seg file
        seg_file = str(self.seg_path.joinpath(file_path.stem + ".seg")).replace("_rotated", "")
        label = np.loadtxt(seg_file, dtype=int, ndmin=1)
        
        # Validate label count matches face count
        if sample["graph"].number_of_nodes() != label.shape[0]:
            return None  # Skip samples with mismatched labels
            
        sample["graph"].ndata["y"] = torch.tensor(label).long()
        return sample