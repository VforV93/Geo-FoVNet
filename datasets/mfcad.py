"""MFCAD++: 25-class mechanical part face segmentation."""

from datasets.base import BaseDataset


class MFCAD(BaseDataset):
    @staticmethod
    def num_classes():
        return 25

    def __init__(self, root_dir, center_and_scale=False, random_rotate=False, char2label=None):
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
