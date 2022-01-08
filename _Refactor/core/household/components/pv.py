
from typing import Dict, Any

from _Refactor.core.elements.component import Component


class PV(Component):

    def __init__(self):
        self.ID_PV: int = None
        self.peak_power: float = None
        self.peak_power_unit: str = None


