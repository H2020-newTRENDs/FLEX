
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class SpaceHeatingTank(Component):

    def __init__(self):
        self.size: float = None
        self.surface = None
        self.loss = None
        self.temperature_start = None
        self.temperature_max = None
        self.temperature_min = None
        self.temperature_outside = None

