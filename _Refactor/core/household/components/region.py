
from typing import Dict, Any
import numpy as np

from _Refactor.core.elements.component import Component


class Region(Component):

    # id: int
    # temperature: np.array
    # radiation: np.array

    def __init__(self):
        self.id = None
        self.temperature: np.array = None
        self.radiation: np.array = None
