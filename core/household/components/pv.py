
import numpy as np
from core.elements.component import Component


class PV(Component):
    def __init__(self, component_id):
        self.ID_PV: int = None
        self.peak_power: float = None
        self.peak_power_unit: str = None
        self.power: np.array = None
        self.power_unit = None

        self.set_parameters(component_id)
