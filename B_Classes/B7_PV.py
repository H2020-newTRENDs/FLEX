
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class PV:

    def __init__(self, para_series):
        self.ID_PVType = para_series["ID_PVType"]
        self.PVPower = para_series["PVPower"]
        self.PVPower_unit = para_series["PVPower_unit"]

