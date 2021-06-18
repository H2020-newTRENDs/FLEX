
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class SpaceHeating:

    def __init__(self, para_series):

        self.ID_SpaceHeatingBoilerType = para_series["ID_SpaceHeatingBoilerType"]
        self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.HeatPumpMaximalThermalPower = para_series["HeatPumpMaximalThermalPower"]

        self.ID_SpaceHeatingTankType = para_series["ID_SpaceHeatingTankType"]
        self.TankSize = para_series["TankSize"]
        self.TankSurfaceArea = para_series["TankSurfaceArea"]
        self.TankLoss = para_series["TankLoss"]

        self.TankStartTemperature = para_series["TankStartTemperature"]
        self.TankMaximalTemperature = para_series["TankMaximalTemperature"]
        self.TankMinimalTemperature = para_series["TankMinimalTemperature"]
        self.TankSurroundingTemperature = para_series["TankSurroundingTemperature"]

        self.ID_SpaceHeatingPumpType = para_series["ID_SpaceHeatingPumpType"]
        self.PumpConsumptionRate = para_series["PumpConsumptionRate"]
        self.MaximalPowerFloorHeating = para_series["MaximalPowerFloorHeating"]






