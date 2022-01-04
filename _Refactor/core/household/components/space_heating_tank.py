
from typing import Dict, Any

from _Refactor.core.elements.component import Component

class SpaceHeatingTank(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.size: float
        self.surface = None
        self.loss = None
        self.temperature_start = None
        self.temperature_max = None
        self.temperature_min = None
        self.temperature_outside = None
        self.set_params(params_dict)

