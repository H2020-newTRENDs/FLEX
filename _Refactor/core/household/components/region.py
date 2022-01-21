
from typing import Dict, Any
import numpy as np

from _Refactor.core.elements.component import Component


class Region(Component):
    def __init__(self, component_id):
        self.ID_Region = None
        self.temperature: np.array = None
        self.north: np.array = None
        self.east: np.array = None
        self.south: np.array = None
        self.west: np.array = None

        self.set_parameters(component_id)
