"""Fusion360: 8-class B-Rep face segmentation (Lambourne et al., CVPR 2021)."""

import numpy as np
import torch
from datasets.base import BaseDataset


class Fusion360(BaseDataset):
    @staticmethod
    def num_classes():
        return 8

    def __init__(self, root_dir, center_and_scale=False, random_rotate=False, char2label=None):
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        self.seg_path = self.path.parent.parent / "seg"

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        seg_file = str(self.seg_path / (file_path.stem + ".seg")).replace("_rotated", "")
        label = np.loadtxt(seg_file, dtype=int, ndmin=1)
        if sample["graph"].number_of_nodes() != label.shape[0]:
            return None
        sample["graph"].ndata["y"] = torch.tensor(label).long()
        return sample
