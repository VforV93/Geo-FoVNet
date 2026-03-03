"""Dataset modules for B-Rep learning tasks."""

from .base import BaseDataset
from .solidletters import SolidLetters
from .fusion360 import Fusion360
from .mfcad import MFCAD
from .traceparts import TraceParts
from .bendfm import BenDFM

__all__ = ["BaseDataset", "SolidLetters", "Fusion360", "MFCAD", "TraceParts", "BenDFM"]
