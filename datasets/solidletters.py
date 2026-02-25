"""SolidLetters Dataset: 3D alphabet characters for classification.

Contains 26 letter classes (A-Z) represented as 3D solids.
Each sample is a DGL graph with geometric features.
"""

import string

import torch

from datasets.base import BaseDataset


# Mapping from characters to class labels
CHAR2LABEL = {char: i for (i, char) in enumerate(string.ascii_lowercase)}


def _char_to_label(char):
    """Convert character to integer label."""
    return CHAR2LABEL[char.lower()]


class SolidLetters(BaseDataset):
    """SolidLetters dataset for 3D letter classification (26 classes)."""
    
    @staticmethod
    def num_classes():
        return 26

    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None,
    ):
        """Initialize SolidLetters dataset.
        
        Args:
            root_dir: Path to dataset directory
            center_and_scale: Whether to center and scale models
            random_rotate: Whether to apply random 3D rotations
            char2label: Optional character-to-label mapping (uses default if None)
        """
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        
        # Extract labels from filenames (first character)
        self.labels = [_char_to_label(fn.stem[0]) for fn in self.file_paths]

    def load_one_graph(self, file_path):
        """Load a single graph and attach its label.
        
        Args:
            file_path: Path to .bin graph file
            
        Returns:
            dict: Sample with 'graph', 'filename', and 'label' keys
        """
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([_char_to_label(file_path.stem[0])], dtype=torch.long)
        return sample