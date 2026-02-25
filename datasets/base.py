"""
Base Dataset Classes for B-Rep Learning

This module provides the base dataset implementation for loading and processing
boundary representation (B-Rep) 3D models stored as DGL graphs.

Classes:
    BaseDataset: Abstract base class for all B-Rep datasets
    
Features:
    - Lazy loading of graph data from binary files
    - Optional data augmentation (random 3D rotations)
    - Centering and scaling transformations  
    - Batch collation for graph neural networks
"""

import pathlib
from abc import abstractmethod

import dgl
import torch
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from datasets import util


class BaseDataset(Dataset):
    """Abstract base class for B-Rep datasets.
    
    Provides common functionality for loading DGL graphs from disk,
    applying transformations, and batching for training.
    
    Subclasses must implement:
        - num_classes(): Static method returning number of classes
        - load_one_graph(): Method to load a single graph with labels
    """
    
    @staticmethod
    @abstractmethod
    def num_classes():
        """Return number of classes in the dataset."""
        pass
    
    def __init__(
        self,
        root_dir,
        center_and_scale=False,
        random_rotate=False,
        char2label=None
    ):
        """
        Base initialization for all datasets.
        Subclasses should call super().__init__() and then add dataset-specific logic.
        """
        self.path = pathlib.Path(root_dir)
        self.random_rotate = random_rotate
        self.center_and_scale = center_and_scale
        self.char2label = char2label
        
        # Load all .bin files from the directory
        self.file_paths = util.get_filenames(self.path)

    def load_graphs(self, file_paths, center_and_scale=True):
        self.data = []
        for fn in tqdm(file_paths):
            if not fn.exists():
                continue
            sample = self.load_one_graph(fn)
            if sample is None:
                continue
            if sample["graph"].edata["x"].size(0) == 0:
                # Catch the case of graphs with no edges
                continue
            self.data.append(sample)
        if center_and_scale:
            self.center_and_scale()
        self.convert_to_float32()
    
    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        sample = {"graph": graph, "filename": file_path.stem}
        return sample

    def center_and_scale(self):
        for i in range(len(self.data)):
            self.data[i]["graph"].ndata["x"], center, scale = util.center_and_scale_uvgrid(
                self.data[i]["graph"].ndata["x"], return_center_scale=True
            )
            if "vision_grids" in self.data[i]["graph"].ndata:
                # channel 0 is occupancy, channel 1 distance, channel 2 dot prod, channel 3 inner occupancy, channel 4 inner distance, channel 5 inner dot prod.
                # only scale the distances, vision grids having shape [num_nodes, elev, azim, channels]
                self.data[i]["graph"].ndata["vision_grids"][..., 1] *= scale
                self.data[i]["graph"].ndata["vision_grids"][..., 4] *= scale
                
            self.data[i]["graph"].edata["x"][..., :3] -= center
            self.data[i]["graph"].edata["x"][..., :3] *= scale

    def convert_to_float32(self):
        for i in range(len(self.data)):
            for key in self.data[i]["graph"].ndata:
                self.data[i]["graph"].ndata[key] = self.data[i]["graph"].ndata[key].float()
            for key in self.data[i]["graph"].edata:
                self.data[i]["graph"].edata[key] = self.data[i]["graph"].edata[key].float()

    def __len__(self):
        if hasattr(self, 'data'):
            return len(self.data)
        elif hasattr(self, 'file_paths'):
            return len(self.file_paths)
        else:
            raise NotImplementedError("Dataset must have either 'data' or 'file_paths' attribute")
    
    def _collate_with_labels(self, batch):
        """Collate function for classification datasets that have labels."""
        collated = self._collate(batch)
        collated["label"] = torch.cat([x["label"] for x in batch], dim=0)
        return collated

    def __getitem__(self, idx):
        if hasattr(self, 'file_paths'):
            file_path = self.file_paths[idx]
            try:
                sample = self.load_one_graph(file_path)
            except Exception as e:
                print(f"[Worker crash] Failed to load {file_path}: {e}")
                return {"graph": dgl.graph([])}  # fallback
        else:
            raise NotImplementedError("Dataset must have either 'data' or 'file_paths' attribute")
        
        # Apply center and scale if enabled
        if getattr(self, 'center_and_scale', False):
            sample["graph"].ndata["x"], center, scale = util.center_and_scale_uvgrid(
                sample["graph"].ndata["x"], return_center_scale=True
            )
            if "vision_grids" in sample["graph"].ndata:
                # channel 0 is occupancy, channel 1 distance, channel 2 dot prod, channel 3 inner occupancy, channel 4 inner distance, channel 5 inner dot prod.
                # only scale the distances, vision grids having shape [num_nodes, elev, azim, channels]
                sample["graph"].ndata["vision_grids"][..., 1] *= scale
                sample["graph"].ndata["vision_grids"][..., 4] *= scale
                sample["graph"].ndata["x_local"][..., :3] *= scale
                
            sample["graph"].edata["x"][..., :3] -= center
            sample["graph"].edata["x"][..., :3] *= scale
        
        # Convert to float32
        for key in sample["graph"].ndata:
            sample["graph"].ndata[key] = sample["graph"].ndata[key].float()
        for key in sample["graph"].edata:
            sample["graph"].edata[key] = sample["graph"].edata[key].float()
                
        # Apply random rotation if enabled
        if getattr(self, 'random_rotate', False):

            R = util.get_random_rotation_matrix()
            
            # Apply rotation to face and edge features
            util.rotate_face_features(sample["graph"].ndata, R)
            util.rotate_edge_features(sample["graph"].edata, R)
        
        return sample

    def _collate(self, batch):
        batched_graph = dgl.batch([sample["graph"] for sample in batch])
        batched_filenames = [sample["filename"] for sample in batch]
        return {"graph": batched_graph, "filename": batched_filenames}

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self._collate,
            num_workers=num_workers,  # Can be set to non-zero on Linux
            drop_last=False,
        )