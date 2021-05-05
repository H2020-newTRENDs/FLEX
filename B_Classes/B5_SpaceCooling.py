
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class SpaceCooling:

    def __init__(self, para_series):
        self.ID_SpaceCoolingType = para_series["ID_SpaceCoolingType"]
        self.ID_EnergyCarrier = para_series["ID_EnergyCarrier"]
        self.SpaceCoolingEfficiency = para_series["SpaceCoolingEfficiency"]
