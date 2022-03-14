
from typing import Dict, Any

from core.elements.component import Component


class Battery(Component):
    def __init__(self, component_id):
        self.ID_Battery: str = None
        self.capacity: float = None
        self.capacity_unit: str = None
        self.charge_efficiency: float = None
        self.discharge_efficiency: float = None
        self.charge_power_max: float = None
        self.charge_power_max_unit: str = None
        self.discharge_power_max: float = None
        self.discharge_power_max_unit: str = None

        self.set_parameters(component_id)


