"""Base dataset for loading B-Rep DGL graphs with optional transforms."""

import pathlib
from abc import abstractmethod

import dgl
import torch
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset, DataLoader

from datasets import util


class BaseDataset(Dataset):
    """Abstract base for B-Rep datasets (lazy graph loading).

    Subclasses must implement ``num_classes()`` and optionally
    override ``load_one_graph()``.
    """

    @staticmethod
    @abstractmethod
    def num_classes():
        pass

    def __init__(self, root_dir, center_and_scale=False, random_rotate=False, char2label=None):
        self.path = pathlib.Path(root_dir)
        self.random_rotate = random_rotate
        self.center_and_scale = center_and_scale
        self.char2label = char2label
        self.file_paths = util.get_filenames(self.path)

    # ── graph I/O ───────────────────────────────────────────────────────
    def load_one_graph(self, file_path):
        graph = load_graphs(str(file_path))[0][0]
        return {"graph": graph, "filename": file_path.stem}

    # ── transforms ──────────────────────────────────────────────────────
    @staticmethod
    def _apply_center_and_scale(sample):
        g = sample["graph"]
        g.ndata["x"], center, scale = util.center_and_scale_uvgrid(
            g.ndata["x"], return_center_scale=True
        )
        if "vision_grids" in g.ndata:
            # Vision grids: [ch0=OV_occ, ch1=OV_dist, ch2=OV_dot, ch3=IV_occ, ch4=IV_dist, ch5=IV_dot]
            # (OV=outer/forward vision, IV=inner/opposite vision). Scale distance channels only.
            g.ndata["vision_grids"][..., 1] *= scale
            g.ndata["vision_grids"][..., 4] *= scale
            g.ndata["x_local"][..., :3] *= scale
        g.edata["x"][..., :3] -= center
        g.edata["x"][..., :3] *= scale

    @staticmethod
    def _to_float32(graph):
        for key in graph.ndata:
            graph.ndata[key] = graph.ndata[key].float()
        for key in graph.edata:
            graph.edata[key] = graph.edata[key].float()

    # ── Dataset interface ───────────────────────────────────────────────
    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        try:
            sample = self.load_one_graph(self.file_paths[idx])
        except Exception as e:
            print(f"[Worker] Failed to load {self.file_paths[idx]}: {e}")
            return {"graph": dgl.graph([]), "filename": self.file_paths[idx].stem}

        if self.center_and_scale:
            self._apply_center_and_scale(sample)
        self._to_float32(sample["graph"])

        if self.random_rotate:
            R = util.get_random_rotation_matrix()
            util.rotate_face_features(sample["graph"].ndata, R)
            util.rotate_edge_features(sample["graph"].edata, R)
        return sample

    # ── collation / dataloader ──────────────────────────────────────────
    def _collate(self, batch):
        return {
            "graph": dgl.batch([s["graph"] for s in batch]),
            "filename": [s["filename"] for s in batch],
        }

    def _collate_with_labels(self, batch):
        c = self._collate(batch)
        c["label"] = torch.cat([s["label"] for s in batch])
        return c

    def get_dataloader(self, batch_size=128, shuffle=True, num_workers=0):
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle,
            collate_fn=self._collate, num_workers=num_workers, drop_last=False,
        )
