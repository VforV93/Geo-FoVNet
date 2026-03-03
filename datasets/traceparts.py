"""TraceParts: 6-class industrial component classification."""

import string
import torch
from datasets.base import BaseDataset

CHAR2LABEL = {c: i for i, c in enumerate(string.digits)}


class TraceParts(BaseDataset):
    @staticmethod
    def num_classes():
        return 6

    def __init__(self, root_dir, center_and_scale=False, random_rotate=False, char2label=None):
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        self.labels = [CHAR2LABEL[fn.stem[0]] for fn in self.file_paths]

    def load_one_graph(self, file_path):
        sample = super().load_one_graph(file_path)
        sample["label"] = torch.tensor([CHAR2LABEL[file_path.stem[0]]], dtype=torch.long)
        return sample
