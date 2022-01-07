
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class AirConditioner(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.ID_AirConditioner: int
        self.power: float
        self.power_unit: str

        self.set_params(params_dict)

