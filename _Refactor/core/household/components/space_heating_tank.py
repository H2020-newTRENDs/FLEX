
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class SpaceHeatingTank(Component):
    def __init__(self, component_id):
        self.ID_SpaceHeatingTank: int = None
        self.size: float = None
        self.size_unit: float = None
        self.surface_area: float = None
        self.surface_area_unit: str = None
        self.loss: float = None
        self.loss_unit: str = None
        self.temperature_start: float = None
        self.temperature_max: float = None
        self.temperature_min: float = None
        self.temperature_surrounding: float = None

        self.set_parameters(component_id)

