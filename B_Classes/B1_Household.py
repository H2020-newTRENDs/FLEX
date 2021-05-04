
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
        self.ID_AgeGroup = para_series["ID_AgeGroup"]
        [self.SpaceHeatingTargetTemperature, self.SpaceCoolingTargetTemperature] = self.init_TargetTemperature()
        self.ID_LifeStyleType = para_series["ID_LifeStyleType"]

        # Building Object
        Table_OBJ_Building = DB().read_DataFrame(REG().Gen_ID_OBJ_Building, self.Conn, ID=para_series["ID_OBJ_Building"])
        self.Building = Building(self, Table_OBJ_Building.iloc[0], self.Conn)

        # Appliances Object
        Table_OBJ_Appliances = DB().read_DataFrame(REG().Gen_ID_OBJ_Appliances, self.Conn, ID=para_series["ID_OBJ_Appliances"])
        self.Appliances = ApplianceGroup(self, Table_OBJ_Appliances.iloc[0], self.Conn)

        # SpaceHeating Object
        Table_OBJ_SpaceHeating = DB().read_DataFrame(REG().Gen_ID_OBJ_SpaceHeating, self.Conn, ID=para_series["ID_OBJ_SpaceHeating"])
        self.SpaceHeating = SpaceHeating(self, Table_OBJ_SpaceHeating.iloc[0], self.Conn)

        # SpaceCooling Object
        Table_OBJ_SpaceCooling = DB().read_DataFrame(REG().Gen_ID_OBJ_SpaceCooling, self.Conn, ID=para_series["ID_OBJ_SpaceCooling"])
        self.SpaceCooling = SpaceCooling(self, Table_OBJ_SpaceCooling.iloc[0], self.Conn)

        # HotWater Object
        Table_OBJ_HotWater = DB().read_DataFrame(REG().Gen_ID_OBJ_HotWater, self.Conn, ID=para_series["ID_OBJ_HotWater"])
        self.HotWater = HotWater(self, Table_OBJ_HotWater.iloc[0], self.Conn)

        # PV Object
        Table_OBJ_PV = DB().read_DataFrame(REG().Gen_ID_OBJ_PV, self.Conn, ID=para_series["ID_OBJ_PV"])
        self.PV = PV(self, Table_OBJ_PV.iloc[0], self.Conn)

        # Battery Object
        Table_OBJ_Battery = DB().read_DataFrame(REG().Gen_ID_OBJ_Battery, self.Conn, ID=para_series["ID_OBJ_Battery"])
        self.Battery = Battery(self, Table_OBJ_Battery.iloc[0], self.Conn)

        # ElectricVehicle Object
        Table_OBJ_ElectricVehicle = DB().read_DataFrame(REG().Gen_ID_OBJ_ElectricVehicle, self.Conn, ID=para_series["ID_OBJ_ElectricVehicle"])
        self.ElectricVehicle = ElectricVehicle(self, Table_OBJ_ElectricVehicle.iloc[0], self.Conn)

    # Target Temperature
    def init_TargetTemperature(self):

        HeatingTable = DB().read_DataFrame(REG().Exo_SpaceHeatingTargetTemperature, self.Conn, ID_AgeGroup=self.ID_AgeGroup)
        HeatingTargetTemperature = HeatingTable.iloc[0]["Value"]
        CoolingTable = DB().read_DataFrame(REG().Exo_SpaceCoolingTargetTemperature, self.Conn, ID_AgeGroup=self.ID_AgeGroup)
        CoolingTargetTemperature = CoolingTable.iloc[0]["Value"]

        return [HeatingTargetTemperature, CoolingTargetTemperature]
