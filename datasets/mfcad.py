"""MFCAD++ Dataset: Mechanical parts with face segmentation labels.

Contains mechanical CAD parts with 25-class face segmentation labels.
"""

from datasets.base import BaseDataset


class MFCAD(BaseDataset):
    @staticmethod
    def num_classes():
        return 25

    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None
    ):
        """
        Load the MFCAD dataset.

        Args:
            root_dir (str): Root path of dataset
            split (str, optional): Data split to load. Defaults to "train".
            center_and_scale (bool, optional): Whether to center and scale the solid. Defaults to True.
            random_rotate (bool, optional): Whether to apply random rotations to the solid. Defaults to False.
        """
        super().__init__(root_dir, center_and_scale, random_rotate, char2label)
        print(f"MFCAD dataset initialized with {len(self.file_paths)} samples.")