from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from flex.core.component import Component, ComponentEnum


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
    db_name: Optional[str] = None


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
class SEMS(Component):
    adoption: Optional[int] = None


@dataclass
class EnergyPrice(Component):
    electricity_consumption: Optional[Union[float, np.ndarray]] = None  # TODO consumption??
    electricity_feed_in: Optional[Union[float, np.ndarray]] = None
    district_heating: Optional[Union[float, np.ndarray]] = None
    natural_gas: Optional[Union[float, np.ndarray]] = None
    heating_oil: Optional[Union[float, np.ndarray]] = None
    biomass: Optional[Union[float, np.ndarray]] = None
    gasoline: Optional[Union[float, np.ndarray]] = None
    price_unit: Optional[str] = None


@dataclass
class Behavior(Component):
    target_temperature_at_home_max: Optional[float] = None
    target_temperature_at_home_min: Optional[float] = None
    target_temperature_not_at_home_max: Optional[float] = None
    target_temperature_not_at_home_min: Optional[float] = None
    target_temperature_unit: Optional[str] = None
    vehicle_distance_annual: Optional[float] = None
    vehicle_distance_unit: Optional[str] = None
    hot_water_demand_annual: Optional[float] = None
    hot_water_demand_unit: Optional[str] = None
    appliance_electricity_demand_annual: Optional[float] = None
    appliance_electricity_demand_unit: Optional[str] = None
    target_temperature_array_max: Optional[np.ndarray] = None
    target_temperature_array_min: Optional[np.ndarray] = None
    vehicle_at_home: Optional[np.ndarray] = None
    vehicle_distance: Optional[np.ndarray] = None
    hot_water_demand: Optional[np.ndarray] = None
    appliance_electricity_demand: Optional[np.ndarray] = None


class OperationComponentEnum(ComponentEnum):
    Region = "region"
    Building = "building"
    Boiler = "boiler"
    SpaceHeatingTank = "space_heating_tank"
    HotWaterTank = "hot_water_tank"
    SpaceCoolingTechnology = "space_cooling_technology"
    PV = "pv"
    Battery = "battery"
    Vehicle = "vehicle"
    SEMS = "sems"
    EnergyPrice = "energy_price"
    Behavior = "behavior"

    @property
    def table_name(self):
        return "OperationScenario_" + self.name

