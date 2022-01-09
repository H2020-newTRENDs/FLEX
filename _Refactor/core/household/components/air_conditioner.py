
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class AirConditioner(Component):

    def __init__(self):
        self.ID_AirConditioner: int = None
        self.efficiency: float = None
        self.power: float = None
        self.power_unit: str = None


