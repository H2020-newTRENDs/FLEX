from enum import Enum, auto
from models.operation import components


class ScenarioEnum(Enum):
    Scenario = "scenario"
    Region = "region"
    Building = "building"
    Boiler = "boiler"
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
    ResultOptHour = 'Result_OptimizationHour'
    ResultOptYear = 'Result_OptimizationYear'
    ResultRefHour = 'Result_ReferenceHour'
    ResultRefYear = 'Result_ReferenceYear'


class ResultEnum(Enum):

    # space heating
    T_outside = "year_not_include"
    Q_Solar = "year_include"
    SpaceHeatingHourlyCOP = "year_not_include"
    SpaceHeatingHourlyCOP_tank = "year_not_include"

    Q_HeatingTank_in = "year_include"
    Q_HeatingElement = "year_include"
    Q_HeatingTank_out = "year_include"
    Q_HeatingTank = "year_not_include"
    Q_HeatingTank_bypass = "year_include"
    E_Heating_HP_out = "year_include"
    Q_RoomHeating = "year_include"
    T_Room = "year_not_include"
    T_BuildingMass = "year_not_include"

    # hot water
    HotWaterProfile = "year_include"
    HotWaterHourlyCOP = "year_not_include"
    HotWaterHourlyCOP_tank = "year_not_include"

    Q_DHWTank = "year_not_include"
    Q_DHWTank_out = "year_include"
    Q_DHWTank_in = "year_include"
    Q_DHWTank_bypass = "year_include"
    E_DHW_HP_out = "year_include"

    # space cooling
    Q_RoomCooling = "year_include"
    E_RoomCooling = "year_include"
    CoolingHourlyCOP = "year_not_include"

    # PV
    PhotovoltaicProfile = "year_include"

    PV2Load = "year_include"
    PV2Bat = "year_include"
    PV2Grid = "year_include"
    PV2EV = "year_include"

    # battery
    BatSoC = "year_not_include"
    BatCharge = "year_include"
    BatDischarge = "year_include"
    Bat2Load = "year_include"
    Bat2EV = "year_include"

    # EV
    EVDemandProfile = "year_include"
    EVAtHomeProfile = "year_not_include"

    EVSoC = "year_not_include"
    EVCharge = "year_include"
    EVDischarge = "year_include"
    EV2Bat = "year_include"
    EV2Load = "year_include"

    # energy price
    ElectricityPrice = "year_not_include"
    FiT = "year_not_include"

    # grid
    BaseLoadProfile = "year_include"
    Grid = "year_include"
    Grid2Load = "year_include"
    Grid2Bat = "year_include"
    Grid2EV = "year_include"
    Load = "year_include"
    Feed2Grid = "year_include"

