
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class HotWaterTank(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.size: float
        self.size_unit: str
        self.surface_area: float
        self.surface_area_unit: str
        self.loss: float
        self.loss_unit: str
        self.temperature_start: float
        self.temperature_max: float
        self.temperature_min: float
        self.temperature_surrounding: float

        self.set_params(params_dict)

