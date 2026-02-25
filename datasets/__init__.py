"""
Dataset modules for B-Rep learning tasks.

Provides dataset classes for loading and processing boundary representation
models from various CAD datasets.
"""

from .base import BaseDataset
from .solidletters import SolidLetters
from .fusion360 import Fusion360
from .mfcad import MFCAD
from .traceparts import TraceParts

__all__ = [
    'BaseDataset',
    'SolidLetters',
    'Fusion360',
    'MFCAD',
    'TraceParts',
]
