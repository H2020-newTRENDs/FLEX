from dataclasses import dataclass
import itertools
from core.elements.component_setup import GeneralComponent
import pandas as pd

@dataclass
class Component:

    def set_parameters(self, component_id: int):
        # get the data from the root database:
        params = GeneralComponent(row_id=component_id, table_name=self.__class__.__name__).get_complete_data()

        for paramName, paramValue in params.items():
            if paramName in self.__dict__.keys():
                setattr(self, paramName, paramValue)


@dataclass
class Battery(Component):
    capacity: float
    capacity_unit: str
    charge_efficiency: float
    discharge_efficiency: float
    charge_power_max: float
    charge_power_max_unit: str
    discharge_power_max: float
    discharge_power_max_unit: str


@dataclass
class Boiler(Component):
    name: str
    thermal_power_max: float
    thermal_power_max_unit: str
    heating_element_power: float
    heating_element_power_unit: str
    carnot_efficiency_factor: float
    heating_supply_temperature: float
    hot_water_supply_temperature: float


