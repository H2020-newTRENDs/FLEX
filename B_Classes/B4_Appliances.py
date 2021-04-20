
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class Appliances:

    """
    Data type:
    (1) self.BaseElectricityProfile: numpy real array (len = 8760)
    (2) self.WashingMachineAdoptionStatus: boolean
    (3) self.WashingMachineType: int
    (4) self.WashingMachinePower: real
    (5) self.WashingMachineDuration: real
    (6) self.WashingMachineUseDay: numpy 1/0 array (len = 365)
    (7) self.DryerAdoptionStatus: boolean
    (8) self.DryerType: int
    (9) self.DryerPower: real
    (10) self.DryerDuration: real
    (11) self.DryerUseDay: numpy 1/0 array (len = 365)
    (12) self.DishWasherAdoptionStatus: boolean
    (13) self.DishWasherType: int
    (14) self.DishWasherPower: real
    (15) self.DishWasherDuration: real
    (16) self.DishWasherUseDay: numpy 1/0 array (len = 365)
    (17) self.WashingMachineOperation: numpy 1/0 array (len = 8760)
    (18) self.DryerOperation: numpy 1/0 array (len = 8760)
    (19) self.DishWasherOperation: numpy 1/0 array (len = 8760)
    """

    def __init__(self, household, para_series, conn):

        self.Household = household
        self.Conn = conn

        # non-shiftable electricity demand profile
        self.BaseElectricityProfile = self.init_BaseElectricityProfile(para_series["ID_BaseElectricityProfileType"])

        # washing machine
        self.WashingMachineAdoptionStatus = bool(para_series["ID_WashingMachineAdoptionStatus"])
        self.WashingMachineType = para_series["ID_WashingMachineType"]
        [self.WashingMachinePower,
         self.WashingMachineDuration,
         self.WashingMachineUseDay] = self.init_WashingMachine()

        # dryer
        self.DryerAdoptionStatus = bool(para_series["ID_DryerAdoptionStatus"])
        self.DryerType = para_series["ID_DryerType"]
        [self.DryerPower,
         self.DryerDuration,
         self.DryerUseDay] = self.init_Dryer()

        # dish washer
        self.DishWasherAdoptionStatus = bool(para_series["ID_DishWasherAdoptionStatus"])
        self.DishWasherType = para_series["ID_DishWasherType"]
        [self.DishWasherPower,
         self.DishWasherDuration,
         self.DishWasherUseDay] = self.init_DishWasher()

        # operation
        self.WashingMachineOperation = 0
        self.DryerOperation = 0
        self.DishWasherOperation = 0


    def init_BaseElectricityProfile(self, type):

        Table = DB().read_DataFrame(REG().Exo_BaseElectricityProfile, self.Conn,
                                    ID_HouseholdType=self.Household.ID_HouseholdType,
                                    ID_BaseElectricityProfileType=type)
        Profile = Table["Value"].values

        return Profile


    def init_WashingMachine(self):

        type = self.WashingMachineType
        Power = DB().read_DataFrame(REG().Exo_WashingMachinePower, self.Conn).iloc[int(type - 1)]["Value"]
        Duration = DB().read_DataFrame(REG().Exo_WashingMachineDuration, self.Conn).iloc[int(type - 1)]["Value"]
        UseDay = DB().read_DataFrame(REG().Gen_WashingMachineUseDay, self.Conn,
                                     ID_HouseholdType=self.Household.ID_HouseholdType)["Value"].values

        return [Power, Duration, UseDay]


    def init_Dryer(self):

        type = self.DryerType
        Power = DB().read_DataFrame(REG().Exo_DryerPower, self.Conn).iloc[int(type - 1)]["Value"]
        Duration = DB().read_DataFrame(REG().Exo_DryerDuration, self.Conn).iloc[int(type - 1)]["Value"]
        UseDay = DB().read_DataFrame(REG().Gen_DryerUseDay, self.Conn,
                                     ID_HouseholdType=self.Household.ID_HouseholdType)["Value"].values

        return [Power, Duration, UseDay]


    def init_DishWasher(self):

        type = self.DishWasherType
        Power = DB().read_DataFrame(REG().Exo_DishWasherPower, self.Conn).iloc[int(type - 1)]["Value"]
        Duration = DB().read_DataFrame(REG().Exo_DishWasherDuration, self.Conn).iloc[int(type - 1)]["Value"]
        UseDay = DB().read_DataFrame(REG().Gen_DishWasherUseDay, self.Conn,
                                     ID_HouseholdType=self.Household.ID_HouseholdType)["Value"].values

        return [Power, Duration, UseDay]



