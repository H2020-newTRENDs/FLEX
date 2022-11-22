from flex_operation import components
from dataclasses import dataclass


class OperationComponentInfo:

    def __init__(self, name):
        self.name = name
        self.camel_name = self.to_camel(name)
        self.id_name = "".join(["ID_", self.camel_name])
        self.class_var = components.__dict__[self.camel_name]

    @staticmethod
    def to_camel(name: str):
        if name == "pv":
            return "PV"
        else:
            items = [item[0].upper() + item[1:] for item in name.split("_")]
            return "".join(items)

    @property
    def table_name(self) -> str:
        table_name = "OperationScenario_Component_" + self.camel_name
        return table_name


@dataclass
class OperationScenarioComponent:
    Region = OperationComponentInfo("region")
    Building = OperationComponentInfo("building")
    Boiler = OperationComponentInfo("boiler")
    HeatingElement = OperationComponentInfo("heating_element")
    SpaceHeatingTank = OperationComponentInfo("space_heating_tank")
    HotWaterTank = OperationComponentInfo("hot_water_tank")
    SpaceCoolingTechnology = OperationComponentInfo("space_cooling_technology")
    PV = OperationComponentInfo("pv")
    Battery = OperationComponentInfo("battery")
    Vehicle = OperationComponentInfo("vehicle")
    EnergyPrice = OperationComponentInfo("energy_price")
    Behavior = OperationComponentInfo("behavior")


@dataclass
class OperationTable:
    Scenarios = "OperationScenario"
    BehaviorProfile = "OperationScenario_BehaviorProfile"
    EnergyPriceProfile = "OperationScenario_EnergyPrice"
    RegionWeatherProfile = "OperationScenario_RegionWeather"
    ResultOptHour = "OperationResult_OptimizationHour"
    ResultOptYear = "OperationResult_OptimizationYear"
    ResultRefHour = "OperationResult_ReferenceHour"
    ResultRefYear = "OperationResult_ReferenceYear"
    ResultEnergyCost = "OperationResult_OperationEnergyCost"
    ResultEnergyCostChange = "OperationResult_OperationEnergyCostChange"


@dataclass
class OperationResultVar:

    # space heating
    T_outside = "year_not_include"
    Q_Solar = "year_include"
    SpaceHeatingHourlyCOP = "year_not_include"
    SpaceHeatingHourlyCOP_tank = "year_not_include"

    Q_HeatingTank_in = "year_include"
    Q_HeatingTank_out = "year_include"
    Q_HeatingTank = "year_not_include"
    Q_HeatingTank_bypass = "year_include"
    E_Heating_HP_out = "year_include"
    Q_RoomHeating = "year_include"
    T_Room = "year_not_include"
    T_BuildingMass = "year_not_include"

    # Heating Element
    Q_HeatingElement = "year_include"

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
    # EVAtHomeProfile = "year_not_include"

    EVSoC = "year_not_include"
    EVCharge = "year_include"
    EVDischarge = "year_include"
    EV2Bat = "year_include"
    EV2Load = "year_include"

    # energy price
    ElectricityPrice = "year_not_include"
    FiT = "year_not_include"
    GasPrice = "year_not_include"
    Gas = "year_include"

    # grid
    BaseLoadProfile = "year_include"
    Grid = "year_include"
    Grid2Load = "year_include"
    Grid2Bat = "year_include"
    Grid2EV = "year_include"
    Load = "year_include"
    Feed2Grid = "year_include"
