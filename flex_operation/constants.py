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
    ResultOptMonth = "OperationResult_OptimizationMonth"
    ResultOptYear = "OperationResult_OptimizationYear"
    ResultRefHour = "OperationResult_ReferenceHour"
    ResultRefMonth = "OperationResult_ReferenceMonth"
    ResultRefYear = "OperationResult_ReferenceYear"
    ResultEnergyCost = "OperationResult_OperationEnergyCost"
    ResultEnergyCostChange = "OperationResult_OperationEnergyCostChange"


@dataclass
class OperationResultVar:

    # space heating
    T_outside = "hour"
    Q_Solar = "hour&year"
    SpaceHeatingHourlyCOP = "hour"
    SpaceHeatingHourlyCOP_tank = "hour"

    Q_HeatingTank_in = "hour&year"
    Q_HeatingTank_out = "hour&year"
    Q_HeatingTank = "hour"
    Q_HeatingTank_bypass = "hour&year"
    E_Heating_HP_out = "hour&year"
    Q_RoomHeating = "hour&year"
    T_Room = "hour"
    T_BuildingMass = "hour"

    # Heating Element
    Q_HeatingElement = "hour&year"

    # hot water
    HotWaterProfile = "hour&year"
    HotWaterHourlyCOP = "hour"
    HotWaterHourlyCOP_tank = "hour"

    Q_DHWTank = "hour"
    Q_DHWTank_out = "hour&year"
    Q_DHWTank_in = "hour&year"
    Q_DHWTank_bypass = "hour&year"
    E_DHW_HP_out = "hour&year"

    # space cooling
    Q_RoomCooling = "hour&year"
    E_RoomCooling = "hour&year"
    CoolingHourlyCOP = "hour"

    # PV
    PhotovoltaicProfile = "hour&year"

    PV2Load = "hour&year"
    PV2Bat = "hour&year"
    PV2Grid = "hour&year"
    PV2EV = "hour&year"

    # battery
    BatSoC = "hour"
    BatCharge = "hour&year"
    BatDischarge = "hour&year"
    Bat2Load = "hour&year"
    Bat2EV = "hour&year"

    # EV
    EVDemandProfile = "hour&year"
    # EVAtHomeProfile = "hour"

    EVSoC = "hour"
    EVCharge = "hour&year"
    EVDischarge = "hour&year"
    EV2Bat = "hour&year"
    EV2Load = "hour&year"

    # energy price
    ElectricityPrice = "hour"
    FiT = "hour"
    GasPrice = "hour"
    Gas = "hour&year"

    # grid
    BaseLoadProfile = "hour&year"
    Grid = "hour&year"
    Grid2Load = "hour&year"
    Grid2Bat = "hour&year"
    Grid2EV = "hour&year"
    Load = "hour&year"
    Feed2Grid = "hour&year"
