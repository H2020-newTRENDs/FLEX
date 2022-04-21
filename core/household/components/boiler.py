
from core.elements.component import Component


class Boiler(Component):

    def __init__(self, component_id):
        self.ID_Boiler: int = None
        self.name: str = None
        self.electric_power_max: float = None
        self.electric_power_max_unit: str = None
        self.heating_element_power: float = None
        self.heating_element_power_unit: str = None
        self.carnot_efficiency_factor: float = None
        self.heating_supply_temperature = None
        self.hot_water_supply_temperature = None
        self.db_name: str = None

        self.set_parameters(component_id)
