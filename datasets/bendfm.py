"""BenDFM: 2-class sheet metal bending manufacturability."""

import json
import torch
from datasets.base import BaseDataset


def _get_label(file_path, key="y_tool_collision"):
    json_path = file_path.parent.parent / f"{file_path.stem}_labels.json"
    try:
        with open(json_path) as f:
            return json.load(f)[key]
    except FileNotFoundError:
        return -1


class BenDFM(BaseDataset):
    @staticmethod
    def num_classes():
        return 2

    def __init__(self, root_dir, center_and_scale=False, random_rotate=False, char2label=None):
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        self.labels = [_get_label(p) for p in self.file_paths]

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([_get_label(file_path)], dtype=torch.long)
        return sample
