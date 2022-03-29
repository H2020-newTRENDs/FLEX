from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from flex.core.component import Component, ComponentEnum


@dataclass
class Region(Component):
    nuts: Optional[str] = None


@dataclass
class Year(Component):
    year: Optional[int] = None


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


@dataclass
class Boiler(Component):
    type: Optional[str] = None
    thermal_power_max: Optional[float] = None
    thermal_power_max_unit: Optional[str] = None
    heating_element_power: Optional[float] = None
    heating_element_power_unit: Optional[str] = None
    carnot_efficiency_factor: Optional[float] = None
    heating_supply_temperature: Optional[float] = None
    hot_water_supply_temperature: Optional[float] = None


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
    peak_power: Optional[float] = None
    peak_power_unit: Optional[str] = None
    direction: Optional[str] = None
    angle: Optional[float] = None
    angle_unit: Optional[str] = None


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


@dataclass
class SEMS(Component):
    adoption: Optional[int] = None


@dataclass
class Weather(Component):
    region: Optional[List[str]] = None
    temperature: Optional[np.ndarray] = None
    temperature_unit: Optional[List[str]] = None
    radiation_north: Optional[np.ndarray] = None
    radiation_south: Optional[np.ndarray] = None
    radiation_east: Optional[np.ndarray] = None
    radiation_west: Optional[np.ndarray] = None
    radiation_unit: Optional[List[str]] = None


@dataclass
class Electricity(Component):
    region: Optional[List[str]] = None
    consumption_price: Optional[np.ndarray] = None
    feed_in_tariff: Optional[np.ndarray] = None
    price_unit: Optional[List[str]] = None


@dataclass
class DistrictHeating(Component):
    region: Optional[List[str]] = None
    price: Optional[np.ndarray] = None
    price_unit: Optional[List[str]] = None


@dataclass
class NaturalGas(Component):
    region: Optional[List[str]] = None
    price: Optional[np.ndarray] = None
    price_unit: Optional[List[str]] = None


@dataclass
class HeatingOil(Component):
    region: Optional[List[str]] = None
    price: Optional[np.ndarray] = None
    price_unit: Optional[List[str]] = None


@dataclass
class Biomass(Component):
    region: Optional[List[str]] = None
    price: Optional[np.ndarray] = None
    price_unit: Optional[List[str]] = None


@dataclass
class Gasoline(Component):
    region: Optional[List[str]] = None
    gasoline: Optional[np.ndarray] = None
    gasoline_unit: Optional[List[str]] = None


@dataclass
class BehaviorTargetTemperature(Component):
    range_max: Optional[np.ndarray] = None
    range_min: Optional[np.ndarray] = None
    temperature_unit: Optional[List[str]] = None


@dataclass
class BehaviorVehicle(Component):
    at_home: Optional[np.ndarray] = None
    distance: Optional[np.ndarray] = None
    distance_unit: Optional[List[str]] = None


@dataclass
class BehaviorHotWater(Component):
    demand: Optional[np.ndarray] = None
    demand_unit: Optional[List[str]] = None


@dataclass
class BehaviorApplianceElectricity(Component):
    demand: Optional[np.ndarray] = None
    demand_unit: Optional[List[str]] = None


class OperationComponentEnum(ComponentEnum):
    Region = "region"
    Year = "year"
    Building = "building"
    Boiler = "boiler"
    SpaceHeatingTank = "space_heating_tank"
    HotWaterTank = "hot_water_tank"
    SpaceCoolingTechnology = "space_cooling_technology"
    PV = "pv"
    Battery = "battery"
    Vehicle = "vehicle"
    SEMS = "sems"
    Weather = "weather"
    Electricity = "electricity"
    DistrictHeating = "district_heating"
    NaturalGas = "natural_gas"
    HeatingOil = "heating_oil"
    Biomass = "biomass"
    Gasoline = "gasoline"
    BehaviorTargetTemperature = "behavior_target_temperature"
    BehaviorVehicle = "behavior_vehicle"
    BehaviorHotWater = "behavior_hot_water"
    BehaviorApplianceElectricity = "behavior_appliance_electricity"

    @property
    def table_name(self):
        return "OperationScenario_" + self.name
