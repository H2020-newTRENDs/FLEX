
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class HotWater:

    def __init__(self, para_series):

        #idea: HotWaterBoilerType is always same as SpaceHeating
        self.ID_HotWaterBoilerType = para_series["ID_HotWaterBoilerType"]
        #self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        #self.BoilerEfficiency = para_series["BoilerEfficiency"]
        #self.ID_HotWaterTankType = para_series["ID_HotWaterTankType"]
        #self.TankSize = para_series["TankSize"]
        #self.TankSurfaceArea = para_series["TankSurfaceArea"]
        #self.TankLoss = para_series["TankLoss"]
        #self.HotWaterPowerUpLimit = para_series["HotWaterPowerUpLimit"]
        #self.BaseHotWaterProfile = para_series["Sce_Demand_BaseHotWaterProfile"]

