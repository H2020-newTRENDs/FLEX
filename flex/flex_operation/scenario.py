import copy
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from flex.core.config import Config
from flex.core.db import create_db_conn
from flex.core.scenario import Scenario, ScenarioEnum
from .component import Region, Building, Boiler, SpaceHeatingTank, HotWaterTank, SpaceCoolingTechnology, \
    PV, Battery, Vehicle, SEMS, EnergyPrice, Behavior, OperationComponentEnum


@dataclass
class OperationScenario(Scenario):
    scenario_id: int
    config: 'Config'
    region: Optional['Region'] = None
    building: Optional['Building'] = None
    boiler: Optional['Boiler'] = None
    space_heating_tank: Optional['SpaceHeatingTank'] = None
    hot_water_tank: Optional['HotWaterTank'] = None
    space_cooling_technology: Optional['SpaceCoolingTechnology'] = None
    pv: Optional['PV'] = None
    battery: Optional['Battery'] = None
    vehicle: Optional['Vehicle'] = None
    sems: Optional['SEMS'] = None
    energy_price: Optional['EnergyPrice'] = None
    behavior: Optional['Behavior'] = None

    def __post_init__(self):
        self.db = create_db_conn(self.config)
        self.component_scenario_ids = self.get_component_scenario_ids()
        self.import_source_data()
        self.setup_components()
        self.setup_region_and_pv()
        self.setup_energy_price()
        self.setup_behavior()

    def import_source_data(self):
        pass

    def get_component_scenario_ids(self):
        component_scenario_ids = self.get_scenario_data_dict(
            OperationScenarioEnum.Scenario.table_name,
            scenario_filter={OperationScenarioEnum.Scenario.id: self.scenario_id}
        )
        del component_scenario_ids[OperationScenarioEnum.Scenario.id]
        return component_scenario_ids

    def setup_components(self):
        for id_component, component_scenario_id in self.component_scenario_ids.items():
            component_name_cap = id_component.replace("ID_", "")
            component_enum = OperationComponentEnum.__dict__[component_name_cap]
            if component_enum.value in self.__dict__.keys():
                component_params = self.get_scenario_data_dict(
                    component_enum.table_name,
                    scenario_filter={component_enum.id: component_scenario_id}
                )
                instance = getattr(sys.modules[__name__], component_name_cap)()
                instance.set_params(component_params)
                setattr(self, component_enum.value, instance)

    @staticmethod
    def get_float2vector(value: float):
        return value * np.ones(8760, )

    def setup_region_and_pv(self):
        df = self.db.read_dataframe("region_pv_gis", filter={"region": self.region.code,
                                                             "year": self.region.year})
        self.region.temperature = df["temperature"].to_numpy()
        self.region.radiation_north = df["radiation_north"].to_numpy()
        self.region.radiation_south = df["radiation_south"].to_numpy()
        self.region.radiation_east = df["radiation_east"].to_numpy()
        self.region.radiation_west = df["radiation_west"].to_numpy()
        self.pv.generation = df["pv_generation"].to_numpy() * self.pv.size

    def setup_energy_price(self):
        energy_carrier_prices: dict = copy.deepcopy(self.energy_price.__dict__)
        del energy_carrier_prices['price_unit']
        for energy_carrier, price in energy_carrier_prices.items():
            if self.energy_price.__dict__[energy_carrier] is not None:
                self.energy_price.__dict__[energy_carrier] = self.get_float2vector(price)

    @staticmethod
    def gen_target_temperature_range_array(at_home_array,
                                           at_home_max, at_home_min,
                                           not_at_home_max, not_at_home_min) -> (np.ndarray, np.ndarray):
        max_array = np.zeros(8760, )
        min_array = np.zeros(8760, )
        for hour in range(0, 8760):
            if at_home_array[hour] == 1:
                max_array[hour] = at_home_max
                min_array[hour] = at_home_min
            else:
                max_array[hour] = not_at_home_max
                min_array[hour] = not_at_home_min
        return max_array, min_array

    def setup_behavior_target_temperature(self, behavior: pd.DataFrame):
        self.behavior.target_temperature_array_max, self.behavior.target_temperature_array_min = \
            self.gen_target_temperature_range_array(behavior["people_at_home"].to_numpy(),
                                                    self.behavior.target_temperature_at_home_max,
                                                    self.behavior.target_temperature_at_home_min,
                                                    self.behavior.target_temperature_not_at_home_max,
                                                    self.behavior.target_temperature_not_at_home_min)

    @staticmethod
    def gen_profile_with_annual_amount(annual_amount: float, shape_array: np.ndarray) -> np.ndarray:
        profile = np.zeros(8760, )
        weight_sum = shape_array.sum()
        for index, hour_weight in enumerate(shape_array):
            profile[index] = annual_amount * hour_weight / weight_sum
        return profile

    def setup_behavior_vehicle(self, behavior: pd.DataFrame):
        self.behavior.vehicle_at_home = behavior["vehicle_at_home"].to_numpy()
        self.behavior.vehicle_distance = self.gen_profile_with_annual_amount(self.behavior.vehicle_distance_annual,
                                                                             behavior["vehicle_distance"].to_numpy())

    def setup_behavior_hot_water_demand(self, behavior: pd.DataFrame):
        self.behavior.hot_water_demand = self.gen_profile_with_annual_amount(self.behavior.hot_water_demand_annual,
                                                                             behavior["hot_water_demand"].to_numpy())

    def setup_behavior_appliance_electricity_demand(self, behavior: pd.DataFrame):
        self.behavior.appliance_electricity_demand = \
            self.gen_profile_with_annual_amount(self.behavior.appliance_electricity_demand_annual,
                                                behavior["appliance_electricity_demand"].to_numpy())

    def setup_behavior(self):
        behavior_df = self.db.read_dataframe("behavior")
        try:
            self.setup_behavior_target_temperature(behavior_df)
            self.setup_behavior_vehicle(behavior_df)
            self.setup_behavior_hot_water_demand(behavior_df)
            self.setup_behavior_appliance_electricity_demand(behavior_df)
        except Exception as e:
            print("There seems to be something wrong with the behavior table.")


class OperationScenarioEnum(ScenarioEnum):
    Scenario = "scenario"

    @property
    def table_name(self):
        return "Operation" + self.name
