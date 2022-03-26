
from core.elements.component import Component
from dataclasses import dataclass


# class AirConditioner(Component):
#
#     def __init__(self, component_id):
#         self.ID_AirConditioner: int = None
#         self.efficiency: float = None
#         self.power: float = None
#         self.power_unit: str = None
#
#         self.set_parameters(component_id)


@dataclass
class AirConditioner(Component):
    component_id: int
    ID_AirConditioner: int = None
    efficiency: float = None
    power: float = None
    power_unit: str = None

    def __post_init__(self):
        self.set_parameters(self.component_id)