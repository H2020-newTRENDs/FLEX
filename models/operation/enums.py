from enum import Enum
from models.operation import components


class ScenarioEnum(Enum):
    Scenario = "scenario"
    Region = "region"
    Building = "building"
    Boiler = "boiler"  # TODO: this name is a bit confusing, better to be "SpaceHeatingTechnology"
    SpaceHeatingTank = "space_heating_tank"
    HotWaterTank = "hot_water_tank"
    SpaceCoolingTechnology = "space_cooling_technology"
    PV = "pv"
    Battery = "battery"
    Vehicle = "vehicle"
    EnergyPrice = "energy_price"
    Behavior = "behavior"

    @property
    def table_name(self):
        if self.name == 'Scenario':
            table_name = "Scenario"
        else:
            table_name = "Scenario_Component_" + self.name
        return table_name

    @property
    def id(self):
        return "ID_" + self.name

    @property
    def class_var(self):
        return components.__dict__[self.name]


class TableEnum(Enum):
    BehaviorProfile = 'Scenario_Profile_Behavior'
    EnergyPriceProfile = 'Scenario_Profile_EnergyPrice'
    RegionWeatherProfile = 'Scenario_Profile_RegionWeather'
