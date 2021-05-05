
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class Building:

    def __init__(self, para_series):
        self.ID_DishWasherType = para_series["ID_DishWasherType"]
        self.ID_EnergyCarrier_DishWasher = para_series["ID_EnergyCarrier_DishWasher"]
        self.DishWasherPower = para_series["DishWasherPower"]
        self.DishWasherShifting = para_series["DishWasherShifting"]
        pass

