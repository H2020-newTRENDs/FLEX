
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Boiler(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.ID_Boiler: int
        self.name: str
        self.thermal_power_max: float
        self.thermal_power_max_unit: str
        self.heating_element_power: float
        self.heating_element_power_unit: str
        self.carnot_efficiency_factor: float

        self.set_params(params_dict)


