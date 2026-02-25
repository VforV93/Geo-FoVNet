"""TraceParts Dataset: Industrial components classification.

Contains industrial CAD components for 6-class classification.
"""

import string
import torch

from datasets.base import BaseDataset

CHAR2LABEL = {char: i for (i, char) in enumerate(string.digits)}


def _char_to_label(char):
    return CHAR2LABEL[char]


class TraceParts(BaseDataset):
    @staticmethod
    def num_classes():
        return 6

    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None,
    ):
        """
        Initialize TraceParts dataset with file paths and labels only.
        Graphs are loaded on demand in __getitem__.
        """
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        
        self.labels = [_char_to_label(fn.stem[0]) for fn in self.file_paths]
        print(f"TraceParts dataset initialized with {len(self.file_paths)} samples.")

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([_char_to_label(file_path.stem[0])]).long()
        return sample