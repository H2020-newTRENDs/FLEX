
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class ElectricVehicle:

    def __init__(self, para_series):
        self.ID_ElectricVehicleType = para_series["ID_ElectricVehicleType"]
        self.BatterySize = para_series["BatterySize"]
        self.BatteryChargeEfficiency = para_series["BatteryChargeEfficiency"]
        self.BatteryMaxChargePower = para_series["BatteryMaxChargePower"]
        self.BatteryDischargeEfficiency = para_series["BatteryDischargeEfficiency"]
        self.BatteryMaxDischargePower = para_series["BatteryMaxDischargePower"]

