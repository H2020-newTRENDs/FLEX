
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from B_Classes.B2_Building import Building
from B_Classes.B3_ApplianceGroup import ApplianceGroup
from B_Classes.B4_SpaceHeating import SpaceHeating
from B_Classes.B5_SpaceCooling import SpaceCooling
from B_Classes.B6_HotWater import HotWater
from B_Classes.B7_PV import PV
from B_Classes.B8_Battery import Battery
from B_Classes.B9_ElectricVehicle import ElectricVehicle

class Household:

    def __init__(self, para_series, conn):

        self.Conn = conn
        self.ID_Country = para_series["ID_Country"]
        self.ID_HouseholdType = para_series["ID_HouseholdType"]
        self.AveragePersons = para_series["AveragePersons"]
        self.EquivalentPersons = para_series["EquivalentPersons"]
        self.ID_AgeGroup = para_series["ID_AgeGroup"]
        # self.Building = Building(DB().read_DataFrameRow(REG().Gen_OBJ_ID_Building, para_series["ID_Building"] - 1, self.Conn))
        self.ApplianceGroup = ApplianceGroup(DB().read_DataFrameRow(REG().Gen_OBJ_ID_ApplianceGroup, (para_series["ID_ApplianceGroup"] - 1), self.Conn))
        self.SpaceHeating = SpaceHeating(DB().read_DataFrameRow(REG().Gen_OBJ_ID_SpaceHeating, (para_series["ID_SpaceHeating"] - 1), self.Conn))
        self.SpaceCooling = SpaceCooling(DB().read_DataFrameRow(REG().Gen_OBJ_ID_SpaceCooling, (para_series["ID_SpaceCooling"] - 1), self.Conn))
        self.HotWater = HotWater(DB().read_DataFrameRow(REG().Gen_OBJ_ID_HotWater, (para_series["ID_HotWater"] - 1), self.Conn))
        self.PV = PV(DB().read_DataFrameRow(REG().Gen_OBJ_ID_PV, (para_series["ID_PV"] - 1), self.Conn))
        self.Battery = Battery(DB().read_DataFrameRow(REG().Gen_OBJ_ID_Battery, (para_series["ID_Battery"]), self.Conn))
        self.ElectricVehicle = ElectricVehicle(DB().read_DataFrameRow(REG().Gen_OBJ_ID_ElectricVehicle, (para_series["ID_ElectricVehicle"] - 1), self.Conn))




