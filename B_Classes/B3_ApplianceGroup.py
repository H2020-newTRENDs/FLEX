
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class ApplianceGroup:

    def __init__(self, para_series):

        self.ID_DishWasherType = para_series["ID_DishWasherType"]
        self.ID_EnergyCarrier_DishWasher = para_series["ID_EnergyCarrier_DishWasher"]
        self.DishWasherPower = para_series["DishWasherPower"]
        self.DishWasherShifting = para_series["DishWasherShifting"]
        self.ID_DryerType = para_series["ID_DryerType"]
        self.ID_EnergyCarrier_Dryer = para_series["ID_EnergyCarrier_Dryer"]
        self.DryerPower = para_series["DryerPower"]
        self.DryerShifting = para_series["DryerShifting"]
        self.ID_WashingMachineType = para_series["ID_WashingMachineType"]
        self.ID_EnergyCarrier_WashingMachine = para_series["ID_EnergyCarrier_WashingMachine"]
        self.WashingMachinePower = para_series["WashingMachinePower"]
        self.WashingMachineShifting = para_series["WashingMachineShifting"]





