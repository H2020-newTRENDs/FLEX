import sys
from dataclasses import dataclass, fields
from typing import Optional

import numpy as np
import pandas as pd

from flex.config import Config
from flex.db import create_db_conn
from flex_operation.components import (
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
    HeatingElement,
)
from flex_operation.constants import OperationScenarioComponent, OperationTable


@dataclass
class MotherOperationScenario:
    config: "Config"

    def __post_init__(self):
        self.db = create_db_conn(self.config)
        self.component_table_names = self.get_component_table_names()
        self.get_component_tables()

        setattr(self, OperationTable.BehaviorProfile, self.db.read_dataframe(OperationTable.BehaviorProfile))
        setattr(self, OperationTable.EnergyPriceProfile, self.db.read_dataframe(OperationTable.EnergyPriceProfile))
        setattr(self, OperationTable.RegionWeatherProfile, self.db.read_dataframe(OperationTable.RegionWeatherProfile))
        setattr(self, OperationTable.Scenarios, self.db.read_dataframe(OperationTable.Scenarios))

    def get_component_table_names(self) -> list:
        scenario_df = self.db.read_dataframe(OperationTable.Scenarios)
        component_table_names = scenario_df.columns.drop("ID_Scenario").to_list()
        return component_table_names

    def get_component_tables(self):
        for table_name in self.component_table_names:
            component_info = OperationScenarioComponent.__dict__[table_name.replace("ID_", "")]
            df = self.db.read_dataframe(component_info.table_name)
            setattr(self, component_info.table_name, df)


@dataclass
class OperationScenario:
    scenario_id: int
    config: "Config"
    tables: "MotherOperationScenario"
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
        self.db = create_db_conn(self.config)
        self.component_scenario_ids = self.get_component_scenario_ids()
        self.setup_components()
        self.setup_region_weather_and_pv_generation()
        self.setup_energy_price()
        self.setup_behavior()
        self.setup_vehicle_profiles()

    def get_component_scenario_ids(self) -> dict:
        scenario_df = self.tables.__getattribute__(OperationTable.Scenarios)
        scenario_df = scenario_df.set_index("ID_Scenario", drop=True)
        component_scenario_ids: dict = scenario_df.loc[
                                       scenario_df.index == self.scenario_id, :].to_dict()
        return component_scenario_ids

    def setup_components(self):
        for id_component, component_scenario_id in self.component_scenario_ids.items():
            component_info = OperationScenarioComponent.__dict__[id_component.replace("ID_", "")]
            if component_info.name in self.__dict__.keys():
                df = self.tables.__getattribute__(component_info.table_name)
                row = df.loc[df.loc[:, component_info.id_name] == component_scenario_id[self.scenario_id], :].squeeze()
                instance = getattr(sys.modules[__name__], component_info.camel_name)()
                instance.set_params(row.to_dict())
                setattr(self, component_info.name, instance)

    def setup_region_weather_and_pv_generation(self):
        df = self.db.read_dataframe(OperationTable.RegionWeatherProfile)
        self.region.temperature = df["temperature"].to_numpy()
        self.region.radiation_north = df["radiation_north"].to_numpy()
        self.region.radiation_south = df["radiation_south"].to_numpy()
        self.region.radiation_east = df["radiation_east"].to_numpy()
        self.region.radiation_west = df["radiation_west"].to_numpy()
        self.pv.generation = df[f"pv_generation_{self.pv.orientation}"].to_numpy() * self.pv.size

    def setup_energy_price(self):
        df = self.db.read_dataframe(OperationTable.EnergyPriceProfile)
        for key, value in self.energy_price.__dict__.items():
            if key.startswith("id_") and value is not None:
                energy_carrier = key.replace("id_", "")
                energy_price_column = f"{energy_carrier}_{value}"
                self.energy_price.__dict__[energy_carrier] = df[energy_price_column].to_numpy()

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
        behavior_df = self.tables.__getattribute__(OperationTable.BehaviorProfile)
        self.setup_behavior_target_temperature(behavior_df)
        self.setup_behavior_hot_water_demand(behavior_df)
        self.setup_behavior_appliance_electricity_demand(behavior_df)

    def setup_vehicle_profiles(self):
        parking_home = self.db.read_dataframe(OperationTable.DrivingProfile_ParkingHome)
        distance = self.db.read_dataframe(OperationTable.DrivingProfile_Distance)

        if self.vehicle.capacity == 0:
            self.behavior.vehicle_at_home = np.zeros((8760,))
            self.behavior.vehicle_distance = np.zeros((8760,))
            self.behavior.vehicle_demand = np.zeros((8760,))
        else:
            self.behavior.vehicle_at_home = parking_home[str(self.vehicle.id_parking_at_home_profile)].to_numpy()
            self.behavior.vehicle_distance = distance[str(self.vehicle.id_distance_profile)].to_numpy()
            self.behavior.vehicle_demand = self.behavior.vehicle_distance * self.vehicle.consumption_rate
