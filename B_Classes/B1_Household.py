
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB
from B_Classes.B2_Environment import Environment
from B_Classes.B3_Building import Building
from B_Classes.B4_Appliances import Appliances
from B_Classes.B5_SpaceHeating import SpaceHeating
from B_Classes.B6_SpaceCooling import SpaceCooling
from B_Classes.B7_HotWater import HotWater
from B_Classes.B8_PV import PV
from B_Classes.B9_Battery import Battery
from B_Classes.B10_ElectricVehicle import ElectricVehicle

class Household:

    """
    Data type:
    (1) self.ID_Country: int
    (2) self.ID_HouseholdType: int
    (3) self.ID_AgeGroup: int
    (4) self.ID_LifeStyleType: int
    (5) self.Environment: Object
    (6) self.Building: Object
    (7) self.Appliances: Object
    (8) self.SpaceHeating: Object
    (9) self.SpaceCooling: Object
    (10) self.HotWater: Object
    (11) self.PV: Object
    (12) self.Battery: Object
    (13) self.ElectricVehicle: Object
    """

    def __init__(self, para_series, conn):

        self.Conn = conn
        self.ID_Country = para_series["ID_Country"]
        self.ID_HouseholdType = para_series["ID_HouseholdType"]
        self.ID_AgeGroup = para_series["ID_AgeGroup"]
        self.ID_LifeStyleType = para_series["ID_LifeStyleType"]

        # Environment Object
        Row_OBJ_Environment = int(para_series["ID_OBJ_Environment"] - 1)
        Table_OBJ_Environment = DB().read_DataFrame(REG().Gen_ID_OBJ_Environment, self.Conn)
        self.Environment = Environment(self, Table_OBJ_Environment.iloc[Row_OBJ_Environment]["Value"], self.Conn)

        # Building Object
        Row_OBJ_Building = int(para_series["ID_OBJ_Building"] - 1)
        Table_OBJ_Building = DB().read_DataFrame(REG().Gen_ID_OBJ_Building, self.Conn)
        self.Building = Building(self, Table_OBJ_Building.iloc[Row_OBJ_Building]["Value"], self.Conn)

        # Appliances Object
        Row_OBJ_Appliances = int(para_series["ID_OBJ_Appliances"] - 1)
        Table_OBJ_Appliances = DB().read_DataFrame(REG().Gen_ID_OBJ_Appliances, self.Conn)
        self.Appliances = Appliances(self, Table_OBJ_Appliances.iloc[Row_OBJ_Appliances]["Value"], self.Conn)

        # SpaceHeating Object
        Row_OBJ_SpaceHeating = int(para_series["ID_OBJ_SpaceHeating"] - 1)
        Table_OBJ_SpaceHeating = DB().read_DataFrame(REG().Gen_ID_OBJ_SpaceHeating, self.Conn)
        self.SpaceHeating = SpaceHeating(self, Table_OBJ_SpaceHeating.iloc[Row_OBJ_SpaceHeating]["Value"], self.Conn)

        # SpaceCooling Object
        Row_OBJ_SpaceCooling = int(para_series["ID_OBJ_SpaceCooling"] - 1)
        Table_OBJ_SpaceCooling = DB().read_DataFrame(REG().Gen_ID_OBJ_SpaceCooling, self.Conn)
        self.SpaceCooling = SpaceCooling(self, Table_OBJ_SpaceCooling.iloc[Row_OBJ_SpaceCooling]["Value"], self.Conn)

        # HotWater Object
        Row_OBJ_HotWater = int(para_series["ID_OBJ_HotWater"] - 1)
        Table_OBJ_HotWater = DB().read_DataFrame(REG().Gen_ID_OBJ_HotWater, self.Conn)
        self.HotWater = HotWater(self, Table_OBJ_HotWater.iloc[Row_OBJ_HotWater]["Value"], self.Conn)

        # PV Object
        Row_OBJ_PV = int(para_series["ID_OBJ_PV"] - 1)
        Table_OBJ_PV = DB().read_DataFrame(REG().Gen_ID_OBJ_PV, self.Conn)
        self.PV = PV(self, Table_OBJ_PV.iloc[Row_OBJ_PV]["Value"], self.Conn)

        # Battery Object
        Row_OBJ_Battery = int(para_series["ID_OBJ_Battery"] - 1)
        Table_OBJ_Battery = DB().read_DataFrame(REG().Gen_ID_OBJ_Battery, self.Conn)
        self.Battery = Battery(self, Table_OBJ_Battery.iloc[Row_OBJ_Battery]["Value"], self.Conn)

        # ElectricVehicle Object
        Row_OBJ_ElectricVehicle = int(para_series["ID_OBJ_ElectricVehicle"] - 1)
        Table_OBJ_ElectricVehicle = DB().read_DataFrame(REG().Gen_ID_OBJ_ElectricVehicle, self.Conn)
        self.ElectricVehicle = ElectricVehicle(self, Table_OBJ_ElectricVehicle.iloc[Row_OBJ_ElectricVehicle]["Value"], self.Conn)
