
from core.elements.component import Component


class ElectricityPrice(Component):

    def __init__(self, component_id):
        self.ID_ElectricityPrice: int = None
        self.name: str = None
        self.electricity_price = None
        self.unit = None

        self.set_parameters(component_id)


