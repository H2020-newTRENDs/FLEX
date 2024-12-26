import sys
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd

from models.operation.components import Battery
from models.operation.components import Behavior
from models.operation.components import Boiler
from models.operation.components import Building
from models.operation.components import EnergyPrice
from models.operation.components import HeatingElement
from models.operation.components import HotWaterTank
from models.operation.components import PV
from models.operation.components import Region
from models.operation.components import SpaceCoolingTechnology
from models.operation.components import SpaceHeatingTank
from models.operation.components import Vehicle
from models.operation.constants import OperationScenarioComponent
from utils.config import Config
from utils.tables import InputTables


@dataclass
class OperationScenario:
    config: "Config"
    scenario_id: int
    input_tables: Optional[Dict[str, pd.DataFrame]]
    region: Optional["Region"] = None
    building: Optional["Building"] = None
    boiler: Optional["Boiler"] = None
    space_heating_tank: Optional["SpaceHeatingTank"] = None
    hot_water_tank: Optional["HotWaterTank"] = None
    space_cooling_technology: Optional["SpaceCoolingTechnology"] = None
    pv: Optional["PV"] = None
    battery: Optional["Battery"] = None
    vehicle: Optional["Vehicle"] = None
    energy_price: Optional["EnergyPrice"] = None
    behavior: Optional["Behavior"] = None
    heating_element: Optional["HeatingElement"] = None

    def __post_init__(self):
        self.setup_components()
        self.setup_region_weather_and_pv_generation()
        self.setup_energy_price()
        self.setup_behavior()

    def get_component_scenario_ids(self) -> dict:
        scenario_df = self.input_tables[InputTables.OperationScenario.name]
        scenario_df = scenario_df.set_index("ID_Scenario", drop=True)
        component_scenario_ids: dict = scenario_df.loc[scenario_df.index == self.scenario_id, :].to_dict()
        component_scenario_ids = {key: value for key, value in component_scenario_ids.items() if key.startswith('ID')}
        return component_scenario_ids

    def setup_components(self):
        for id_component, component_scenario_id in self.get_component_scenario_ids().items():
            component_info = OperationScenarioComponent.__dict__[id_component.replace("ID_", "")]
            if component_info.name in self.__dict__.keys():
                df = self.input_tables[component_info.table_name]
                row = df.loc[df.loc[:, component_info.id_name] == component_scenario_id[self.scenario_id], :].squeeze()
                instance = getattr(sys.modules[__name__], component_info.camel_name)()
                instance.set_params(row.to_dict())
                setattr(self, component_info.name, instance)

    def setup_region_weather_and_pv_generation(self):
        df = self.input_tables[InputTables.OperationScenario_RegionWeather.name]
        self.region.temperature = df["temperature"].to_numpy()
        self.region.radiation_north = df["radiation_north"].to_numpy()
        self.region.radiation_south = df["radiation_south"].to_numpy()
        self.region.radiation_east = df["radiation_east"].to_numpy()
        self.region.radiation_west = df["radiation_west"].to_numpy()
        self.pv.generation = df[f"pv_generation_optimal"].to_numpy() * self.pv.size

    def setup_energy_price(self):
        df = self.input_tables[InputTables.OperationScenario_EnergyPrice.name]
        for key, value in self.energy_price.__dict__.items():
            if key.startswith("id_") and value is not None:
                energy_carrier = key.replace("id_", "")
                energy_price_column = f"{energy_carrier}_{value}"
                self.energy_price.__dict__[energy_carrier] = df[energy_price_column].to_numpy()

    def setup_behavior(self):
        behavior_df = self.input_tables[InputTables.OperationScenario_BehaviorProfile.name]
        self.behavior.appliance_electricity_demand = self.setup_appliance_electricity_demand_profile()
        self.behavior.hot_water_demand = self.setup_hot_water_demand_profile()
        self.setup_target_temperature(behavior_df)
        self.setup_driving_profiles()

    def setup_appliance_electricity_demand_profile(self):
        behavior_df = self.input_tables[InputTables.OperationScenario_BehaviorProfile.name]
        shape = behavior_df[f'appliance_electricity_demand_profile_1'].to_numpy()
        shape_sum = shape.sum()
        return np.array([
            self.building.appliance_electricity_demand_per_person * self.building.person_num * shape_item / shape_sum
            for shape_item in shape
        ])

    def setup_hot_water_demand_profile(self):
        behavior_df = self.input_tables[InputTables.OperationScenario_BehaviorProfile.name]
        shape = behavior_df[f'hot_water_demand_profile_1'].to_numpy()
        shape_sum = shape.sum()
        return np.array([
            self.building.hot_water_demand_per_person * self.building.person_num * shape_item / shape_sum
            for shape_item in shape
        ])

    def setup_target_temperature(self, behavior: pd.DataFrame):

        def gen_target_temperature_range_array(
                at_home_array, at_home_max, at_home_min, not_at_home_max, not_at_home_min
        ) -> (np.ndarray, np.ndarray):
            hours_num = len(at_home_array)
            max_array = np.zeros(hours_num, )
            min_array = np.zeros(hours_num, )
            for hour in range(0, hours_num):
                if at_home_array[hour] == 1:
                    max_array[hour] = at_home_max
                    min_array[hour] = at_home_min
                else:
                    max_array[hour] = not_at_home_max
                    min_array[hour] = not_at_home_min
            return max_array, min_array

        column = f"occupancy_1"
        (
            self.behavior.target_temperature_array_max,
            self.behavior.target_temperature_array_min,
        ) = gen_target_temperature_range_array(
            behavior[column].to_numpy(),
            self.behavior.target_temperature_at_home_max,
            self.behavior.target_temperature_at_home_min,
            self.behavior.target_temperature_not_at_home_max,
            self.behavior.target_temperature_not_at_home_min,
        )

    def setup_driving_profiles(self):
        parking_home = self.input_tables[InputTables.OperationScenario_DrivingProfile_ParkingHome.name]
        distance = self.input_tables[InputTables.OperationScenario_DrivingProfile_Distance.name]

        if self.vehicle.capacity == 0:
            self.behavior.vehicle_at_home = np.zeros((8760,))
            self.behavior.vehicle_distance = np.zeros((8760,))
            self.behavior.vehicle_demand = np.zeros((8760,))
        else:
            self.behavior.vehicle_at_home = parking_home[str(self.vehicle.id_parking_at_home_profile)].to_numpy()
            self.behavior.vehicle_distance = distance[str(self.vehicle.id_distance_profile)].to_numpy()
            self.behavior.vehicle_demand = self.behavior.vehicle_distance * self.vehicle.consumption_rate
