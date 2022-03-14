
from core.elements.component import Component


class AirConditioner(Component):

    def __init__(self, component_id):
        self.ID_AirConditioner: int = None
        self.efficiency: float = None
        self.power: float = None
        self.power_unit: str = None

        self.set_parameters(component_id)

