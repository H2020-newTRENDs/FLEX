
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

    """
    In the updated version, we do not initialize so many attribute values for households and the objects.
    We only give array of OBJ_types. Then, all the parameter tables are read into memory at the beginning.

    The next classes are instantiated with only one number, and this is the only attribute of it.
    The main reason to have these classes is to put behavioral functions there, for example, diffusion...

    So, in the yearly model, in each year, agents switch between the rows in table "Gen_ID_OBJ_Households".
    This single number goes to DataCollector and stored.
    Following the yearly investment simulation, the saved table is translated to energy consumption and everything.
    """

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
