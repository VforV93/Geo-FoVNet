"""SolidLetters Dataset: 3D alphabet characters for classification.

Contains 26 letter classes (A-Z) represented as 3D solids.
Each sample is a DGL graph with geometric features.
"""

import json
import string
import torch
from pathlib import Path

from datasets.base import BaseDataset

def _get_label_from_json(file_path, label_key = "y_tool_collision"):
    json_path = file_path.parent.parent / f"{file_path.stem}_labels.json"
    try:
        with open(json_path, "r") as f:
            data = json.load(f)
        return data[label_key]
    except FileNotFoundError as e:
        print(f"Error reading label from {json_path}: {e}")
    return -1

class BenDFM(BaseDataset):
    """BenDFM dataset for sheet metal bending manufacturability assessment."""
    
    @staticmethod
    def num_classes():
        return 2

    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None
    ):
        """Initialize SolidLetters dataset.
        
        Args:
            root_dir: Path to dataset directory
            center_and_scale: Whether to center and scale models
            random_rotate: Whether to apply random 3D rotations
            char2label: Optional character-to-label mapping (uses default if None)
        """
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        self.labels = [_get_label_from_json(path) for path in self.file_paths]

    def load_one_graph(self, file_path):
        """Load a single graph and attach its label.
        
        Args:
            file_path: Path to .bin graph file
            
        Returns:
            dict: Sample with 'graph', 'filename', and 'label' keys
        """
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([_get_label_from_json(file_path)], dtype=torch.long)
        return sample