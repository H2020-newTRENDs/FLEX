import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from basics.config import Config
from basics.db import create_db_conn
from models.operation.components import (
    Region,
    Building,
    Boiler,
    SpaceHeatingTank,
    HotWaterTank,
    SpaceCoolingTechnology,
    PV,
    Battery,
    Vehicle,
    EnergyPrice,
    Behavior,
)
from models.operation.enums import OperationScenarioComponent, OperationTable


@dataclass
class OperationScenario:
    scenario_id: int
    config: "Config"
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

    def __post_init__(self):
        self.db = create_db_conn(self.config)
        self.component_scenario_ids = self.get_component_scenario_ids()
        self.setup_components()
        self.setup_region_weather_and_pv_generation()
        self.setup_energy_price()
        self.setup_behavior()

    def get_component_scenario_ids(self):
        scenario_df = self.db.read_dataframe(
            OperationTable.Scenarios.value,
            filter={"ID_Scenario": self.scenario_id},
        )
        component_scenario_ids: dict = scenario_df.iloc[0].to_dict()
        del component_scenario_ids["ID_Scenario"]
        return component_scenario_ids

    def setup_components(self):
        for id_component, component_scenario_id in self.component_scenario_ids.items():
            component_name_cap = id_component.replace("ID_", "")
            component_enum = OperationScenarioComponent.__dict__[component_name_cap]
            if component_enum.value in self.__dict__.keys():
                df = self.db.read_dataframe(
                    component_enum.table_name,
                    filter={component_enum.id: component_scenario_id},
                )
                instance = getattr(sys.modules[__name__], component_name_cap)()
                instance.set_params(df.iloc[0].to_dict())
                setattr(self, component_enum.value, instance)

    def setup_region_weather_and_pv_generation(self):
        df = self.db.read_dataframe(
            OperationTable.RegionWeatherProfile.value,
            filter={"region": self.region.code, "year": self.region.year},
        )
        self.region.temperature = df["temperature"].to_numpy()
        self.region.radiation_north = df["radiation_north"].to_numpy()
        self.region.radiation_south = df["radiation_south"].to_numpy()
        self.region.radiation_east = df["radiation_east"].to_numpy()
        self.region.radiation_west = df["radiation_west"].to_numpy()
        self.pv.generation = df["pv_generation"].to_numpy() * self.pv.size

    def setup_energy_price(self):
        df = self.db.read_dataframe(
            OperationTable.EnergyPriceProfile.value,
            filter={"region": self.region.code, "year": self.region.year},
        )
        for key, value in self.energy_price.__dict__.items():
            if key.startswith("id_") and value is not None:
                energy_carrier = key.replace("id_", "")
                energy_price_column = f"{energy_carrier}_{value}"
                self.energy_price.__dict__[energy_carrier] = df[
                    energy_price_column
                ].to_numpy()
            else:
                pass

    @staticmethod
    def gen_target_temperature_range_array(
        at_home_array, at_home_max, at_home_min, not_at_home_max, not_at_home_min
    ) -> (np.ndarray, np.ndarray):
        hours_num = len(at_home_array)
        max_array = np.zeros(
            hours_num,
        )
        min_array = np.zeros(
            hours_num,
        )
        for hour in range(0, hours_num):
            if at_home_array[hour] == 1:
                max_array[hour] = at_home_max
                min_array[hour] = at_home_min
            else:
                max_array[hour] = not_at_home_max
                min_array[hour] = not_at_home_min
        return max_array, min_array

    def setup_behavior_target_temperature(self, behavior: pd.DataFrame):
        column = f"people_at_home_profile_{self.behavior.id_people_at_home_profile}"
        (
            self.behavior.target_temperature_array_max,
            self.behavior.target_temperature_array_min,
        ) = self.gen_target_temperature_range_array(
            behavior[column].to_numpy(),
            self.behavior.target_temperature_at_home_max,
            self.behavior.target_temperature_at_home_min,
            self.behavior.target_temperature_not_at_home_max,
            self.behavior.target_temperature_not_at_home_min,
        )

    @staticmethod
    def gen_profile_with_annual_amount(
        annual_amount: float, shape_array: np.ndarray
    ) -> np.ndarray:
        profile = np.zeros(
            len(shape_array),
        )
        weight_sum = shape_array.sum()
        for index, hour_weight in enumerate(shape_array):
            profile[index] = annual_amount * hour_weight / weight_sum
        return profile

    def setup_behavior_vehicle(self, behavior: pd.DataFrame):
        column_at_home = (
            f"vehicle_at_home_profile_{self.behavior.id_vehicle_at_home_profile}"
        )
        column_distance = (
            f"vehicle_distance_profile_{self.behavior.id_vehicle_distance_profile}"
        )
        self.behavior.vehicle_at_home = (
            behavior[column_at_home].to_numpy()
        )  # pyomo Problem with 0
        self.behavior.vehicle_distance = behavior[column_distance].to_numpy()
        self.behavior.vehicle_demand = (
            self.behavior.vehicle_distance * self.vehicle.consumption_rate
        )

    def setup_behavior_hot_water_demand(self, behavior: pd.DataFrame):
        column = f"hot_water_demand_profile_{self.behavior.id_hot_water_demand_profile}"
        annual_demand = self.behavior.hot_water_demand_annual * self.building.person_num
        self.behavior.hot_water_demand = self.gen_profile_with_annual_amount(
            annual_demand, behavior[column].to_numpy()
        )

    def setup_behavior_appliance_electricity_demand(self, behavior: pd.DataFrame):
        column = f"appliance_electricity_demand_profile_{self.behavior.id_appliance_electricity_demand_profile}"
        annual_demand = self.behavior.appliance_electricity_demand_annual * self.building.person_num
        self.behavior.appliance_electricity_demand = (
            self.gen_profile_with_annual_amount(
                annual_demand,
                behavior[column].to_numpy(),
            )
        )

    def setup_behavior(self):
        behavior_df = self.db.read_dataframe(OperationTable.BehaviorProfile.value)
        self.setup_behavior_target_temperature(behavior_df)
        self.setup_behavior_vehicle(behavior_df)
        self.setup_behavior_hot_water_demand(behavior_df)
        self.setup_behavior_appliance_electricity_demand(behavior_df)
