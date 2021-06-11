
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class SpaceHeating:

    def __init__(self, para_series):

        self.ID_SpaceHeatingBoilerType = para_series["ID_SpaceHeatingBoilerType"]
        self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.BoilerEfficiency = para_series["BoilerEfficiency"]
        self.ID_SpaceHeatingTankType = para_series["ID_SpaceHeatingTankType"]
        self.TankSize = para_series["TankSize"]
        self.TankSurfaceArea = para_series["TankSurfaceArea"]
        self.TankLoss = para_series["TankLoss"]
        self.ID_SpaceHeatingPumpType = para_series["ID_SpaceHeatingPumpType"]
        self.PumpConsumptionRate = para_series["PumpConsumptionRate"]
        self.HeatingPowerUpLimit = para_series["HeatingPowerUpLimit"]



