
import sqlalchemy.types


class PVData:
    def __init__(self):
        self.nuts_id = sqlalchemy.types.String
        self.ID_PV = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.power = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String
        self.peak_power = sqlalchemy.types.Float
        self.peak_power_unit = sqlalchemy.types.String


class ElectricityPriceData:
    def __init__(self):
        self.ID_ElectricityPrice = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.electricity_price = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class FeedInTariffData:
    def __init__(self):
        self.ID_FeedInTariff = sqlalchemy.types.Integer
        self.id_hour = sqlalchemy.types.Integer
        self.feed_in_tariff = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class AirConditionerData:
    def __init__(self):
        self.ID_AirConditioner = sqlalchemy.types.Integer
        self.efficiency = sqlalchemy.types.Float
        self.power = sqlalchemy.types.Float
        self.power_unit = sqlalchemy.types.String


class BatteryData:
    def __init__(self):
        self.ID_Battery = sqlalchemy.types.Integer
        self.capacity = sqlalchemy.types.Float
        self.capacity_unit = sqlalchemy.types.String
        self.charge_efficiency = sqlalchemy.types.Float
        self.discharge_efficiency = sqlalchemy.types.Float
        self.charge_power_max = sqlalchemy.types.Float
        self.charge_power_max_unit = sqlalchemy.types.String
        self.discharge_power_max = sqlalchemy.types.Float
        self.discharge_power_max_unit = sqlalchemy.types.String


class BehaviorData:
    def __init__(self):
        self.ID_Behavior = sqlalchemy.types.Integer
        self.indoor_set_temperature_min = sqlalchemy.types.Float
        self.indoor_set_temperature_max = sqlalchemy.types.Float


class BoilerData:
    def __init__(self):
        self.ID_Boiler = sqlalchemy.types.Integer
        self.name = sqlalchemy.types.String
        self.power_max = sqlalchemy.types.Float
        self.power_max_unit = sqlalchemy.types.String
        self.heating_element_power = sqlalchemy.types.Float
        self.heating_element_power_unit = sqlalchemy.types.String
        self.carnot_efficiency_factor = sqlalchemy.types.Float
        self.heating_supply_temperature = sqlalchemy.types.Float
        self.hot_water_supply_temperature = sqlalchemy.types.Float


class BuildingData:

    def __init__(self):
        self.ID_Building = sqlalchemy.types.Integer
        self.invert_index = sqlalchemy.types.Integer
        self.name = sqlalchemy.types.String
        self.construction_period = sqlalchemy.types.Integer
        self.hwb_norm = sqlalchemy.types.Float
        self.effective_window_area_west_east = sqlalchemy.types.Float
        self.effective_window_area_north = sqlalchemy.types.Float
        self.effective_window_area_south = sqlalchemy.types.Float
        self.internal_gains = sqlalchemy.types.Float
        self.Af = sqlalchemy.types.Float
        self.Hop = sqlalchemy.types.Float
        self.Htr_w = sqlalchemy.types.Float
        self.Hve = sqlalchemy.types.Float
        self.CM_factor = sqlalchemy.types.Float
        self.Am_factor = sqlalchemy.types.Float
        self.number_of_buildings = sqlalchemy.types.Float
        self.number_of_buildings_with_HP_ground = sqlalchemy.types.Float
        self.number_of_buildings_with_HP_air = sqlalchemy.types.Float
        self.building_mass_temperature_max = sqlalchemy.types.Float
        self.grid_power_max = sqlalchemy.types.Float
        self.grid_power_max_unit = sqlalchemy.types.String


class HotWaterDemandData:
    def __init__(self):
        self.ID_HotWaterDemand = sqlalchemy.types.Integer
        self.hot_water_demand = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class ElectricityDemandData:
    def __init__(self):
        self.ID_ElectricityDemand = sqlalchemy.types.Integer
        self.electricity_demand = sqlalchemy.types.Float
        self.unit = sqlalchemy.types.String


class HotWaterTankData:
    def __init__(self):
        self.ID_HotWaterTank = sqlalchemy.types.Integer
        self.size = sqlalchemy.types.Float
        self.size_unit = sqlalchemy.types.String
        self.surface_area = sqlalchemy.types.Float
        self.surface_area_unit = sqlalchemy.types.String
        self.loss = sqlalchemy.types.Float
        self.loss_unit = sqlalchemy.types.String
        self.temperature_start = sqlalchemy.types.Float
        self.temperature_max = sqlalchemy.types.Float
        self.temperature_min = sqlalchemy.types.Float
        self.temperature_surrounding = sqlalchemy.types.Float


class PersonData:
    def __init__(self):
        self.var_1 = None
        self.var_2 = None


class RegionData:
    def __init__(self):
        self.ID_Region = sqlalchemy.types.String
        self.id_hour = sqlalchemy.types.Integer
        self.south = sqlalchemy.types.Float
        self.east = sqlalchemy.types.Float
        self.west = sqlalchemy.types.Float
        self.north = sqlalchemy.types.Float
        self.temperature = sqlalchemy.types.Float


class SpaceHeatingTankData:
    def __init__(self):
        self.ID_SpaceHeatingTank = sqlalchemy.types.Integer
        self.size = sqlalchemy.types.Float
        self.size_unit = sqlalchemy.types.String
        self.surface_area = sqlalchemy.types.Float
        self.surface_area_unit = sqlalchemy.types.String
        self.loss = sqlalchemy.types.Float
        self.loss_unit = sqlalchemy.types.String
        self.temperature_start = sqlalchemy.types.Float
        self.temperature_max = sqlalchemy.types.Float
        self.temperature_min = sqlalchemy.types.Float
        self.temperature_surrounding = sqlalchemy.types.Float


class VehicleData:
    def __init__(self):
        self.var_1 = None
        self.var_2 = None


class RadiationData:
    def __init__(self):
        # columns
        self.nuts_id = sqlalchemy.types.String
        self.id_hour = sqlalchemy.types.Integer
        self.south = sqlalchemy.types.Float
        self.east = sqlalchemy.types.Float
        self.west = sqlalchemy.types.Float
        self.north = sqlalchemy.types.Float


class TemperatureData:
    def __init__(self):
        self.nuts_id = sqlalchemy.types.String
        self.id_hour = sqlalchemy.types.Integer
        self.temperature = sqlalchemy.types.Float
