
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class Boiler(Component):

    def __init__(self,component_id):
        self.ID_Boiler: int = None
        self.name: str = None
        self.thermal_power_max: float = None
        self.thermal_power_max_unit: str = None
        self.heating_element_power: float = None
        self.heating_element_power_unit: str = None
        self.carnot_efficiency_factor: float = None

        self.set_parameters(component_id)
