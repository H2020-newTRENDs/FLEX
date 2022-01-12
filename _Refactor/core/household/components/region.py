
from typing import Dict, Any
import numpy as np

from _Refactor.core.elements.component import Component


class Region(Component):
    def __init__(self):
        self.ID_Region = None
        self.temperature: np.array = None
        self.radiation_north: np.array = None
        self.radiation_east: np.array = None
        self.radiation_south: np.array = None
        self.radiation_west: np.array = None
