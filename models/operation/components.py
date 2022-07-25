from dataclasses import dataclass
from typing import Optional

import numpy as np


class Component:
    def set_params(self, params: dict):
        for param_name, param_value in params.items():
            if param_name in self.__dict__.keys():
                setattr(self, param_name, param_value)


@dataclass
class Region(Component):
    code: Optional[str] = None
    year: Optional[int] = None
    temperature: Optional[np.ndarray] = None
    temperature_unit: Optional[str] = None
    radiation_north: Optional[np.ndarray] = None
    radiation_south: Optional[np.ndarray] = None
    radiation_east: Optional[np.ndarray] = None
    radiation_west: Optional[np.ndarray] = None
    radiation_unit: Optional[str] = None


@dataclass
class Building(Component):
    type: Optional[str] = None
    construction_period_start: Optional[int] = None
    construction_period_end: Optional[int] = None
    person_num: Optional[float] = None
    Af: Optional[float] = None
    Hop: Optional[float] = None
    Htr_w: Optional[float] = None
    Hve: Optional[float] = None
    CM_factor: Optional[float] = None
    Am_factor: Optional[float] = None
    internal_gains: Optional[float] = None
    effective_window_area_west_east: Optional[float] = None
    effective_window_area_south: Optional[float] = None
    effective_window_area_north: Optional[float] = None
    grid_power_max: Optional[float] = None


@dataclass
class Boiler(Component):
    type: Optional[str] = None
    power_max: Optional[float] = None
    power_max_unit: Optional[str] = None
    heating_element_power: Optional[float] = None
    heating_element_power_unit: Optional[str] = None
    carnot_efficiency_factor: Optional[float] = None
    heating_supply_temperature: Optional[float] = None
    hot_water_supply_temperature: Optional[float] = None


@dataclass
class HeatingElement(Component):
    power: Optional[float] = None
    power_unit: Optional[str] = None
    efficiency: Optional[float] = None


@dataclass
class SpaceHeatingTank(Component):
    size: Optional[float] = None
    size_unit: Optional[str] = None
    surface_area: Optional[float] = None
    surface_area_unit: Optional[str] = None
    loss: Optional[float] = None
    loss_unit: Optional[str] = None
    temperature_start: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    temperature_surrounding: Optional[float] = None
    temperature_unit: Optional[str] = None


@dataclass
class HotWaterTank(Component):
    size: Optional[float] = None
    size_unit: Optional[str] = None
    surface_area: Optional[float] = None
    surface_area_unit: Optional[str] = None
    loss: Optional[float] = None
    loss_unit: Optional[str] = None
    temperature_start: Optional[float] = None
    temperature_max: Optional[float] = None
    temperature_min: Optional[float] = None
    temperature_surrounding: Optional[float] = None
    temperature_unit: Optional[str] = None


@dataclass
class SpaceCoolingTechnology(Component):
    efficiency: Optional[float] = None
    power: Optional[float] = None
    power_unit: Optional[str] = None


@dataclass
class PV(Component):
    size: Optional[float] = None
    size_unit: Optional[str] = None
    generation: Optional[np.ndarray] = None
    generation_unit: Optional[str] = None


@dataclass
class Battery(Component):
    capacity: Optional[float] = None
    capacity_unit: Optional[str] = None
    charge_efficiency: Optional[float] = None
    discharge_efficiency: Optional[float] = None
    charge_power_max: Optional[float] = None
    charge_power_max_unit: Optional[str] = None
    discharge_power_max: Optional[float] = None
    discharge_power_max_unit: Optional[str] = None


@dataclass
class Vehicle(Component):
    type: Optional[str] = None
    capacity: Optional[float] = None
    capacity_unit: Optional[str] = None
    consumption_rate: Optional[float] = None
    consumption_rate_unit: Optional[str] = None
    charge_efficiency: Optional[float] = None
    charge_power_max: Optional[float] = None
    charge_power_max_unit: Optional[str] = None
    discharge_efficiency: Optional[float] = None
    discharge_power_max: Optional[float] = None
    discharge_power_max_unit: Optional[str] = None
    charge_bidirectional: Optional[float] = None


@dataclass
class EnergyPrice(Component):
    id_electricity: Optional[int] = None
    id_electricity_feed_in: Optional[int] = None
    id_solids: Optional[int] = None
    id_liquids: Optional[int] = None
    id_gases: Optional[int] = None
    id_district_heating: Optional[int] = None
    id_gasoline: Optional[int] = None
    price_unit: Optional[str] = None
    electricity: Optional[np.ndarray] = None
    electricity_feed_in: Optional[np.ndarray] = None
    solids: Optional[np.ndarray] = None
    liquids: Optional[np.ndarray] = None
    gases: Optional[np.ndarray] = None
    district_heating: Optional[np.ndarray] = None
    gasoline: Optional[np.ndarray] = None


@dataclass
class Behavior(Component):
    id_people_at_home_profile: Optional[int] = None
    target_temperature_at_home_max: Optional[float] = None
    target_temperature_at_home_min: Optional[float] = None
    target_temperature_not_at_home_max: Optional[float] = None
    target_temperature_not_at_home_min: Optional[float] = None
    shading_solar_reduction_rate: Optional[float] = None
    shading_threshold_temperature: Optional[float] = None
    temperature_unit: Optional[str] = None
    id_hot_water_demand_profile: Optional[int] = None
    hot_water_demand_annual: Optional[float] = None
    hot_water_demand_unit: Optional[str] = None
    id_appliance_electricity_demand_profile: Optional[int] = None
    appliance_electricity_demand_annual: Optional[float] = None
    appliance_electricity_demand_unit: Optional[str] = None
    id_vehicle_at_home_profile: Optional[int] = None
    id_vehicle_distance_profile: Optional[int] = None

    target_temperature_array_max: Optional[np.ndarray] = None
    target_temperature_array_min: Optional[np.ndarray] = None
    hot_water_demand: Optional[np.ndarray] = None
    appliance_electricity_demand: Optional[np.ndarray] = None
    vehicle_at_home: Optional[np.ndarray] = None
    vehicle_distance: Optional[np.ndarray] = None
    vehicle_demand: Optional[np.ndarray] = None
