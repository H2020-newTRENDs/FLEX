
class REG:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.Object = "OBJ_"
        self.Scenario = "Sce_"
        self.GeneratedData = "Gen_"

        # ------------------------------
        # Part I: Operation optimization
        # ------------------------------
        # 1 Objects
        # 1.1 Exogenous tables: ID and parameter
        self.ID_Country = self.ID + "Country"
        self.ID_EnergyCarrier = self.ID + "EnergyCarrier"
        self.ID_HouseholdType = self.ID + "HouseholdType"
        self.ID_AgeGroup = self.ID + "AgeGroup"
        self.ID_SpaceHeatingBoilerType = self.ID + "SpaceHeatingBoilerType"
        self.ID_SpaceHeatingTankType = self.ID + "SpaceHeatingTankType"
        self.ID_SpaceHeatingPumpType = self.ID + "SpaceHeatingPumpType"
        self.ID_HotWaterBoilerType = self.ID + "HotWaterBoilerType"
        self.ID_HotWaterTankType = self.ID + "HotWaterTankType"
        self.ID_SpaceCoolingType = self.ID + "SpaceCoolingType"
        self.ID_PVType = self.ID + "PVType"
        self.ID_BatteryType = self.ID + "BatteryType"
        self.ID_ElectricVehicleType = self.ID + "ElectricVehicleType"

        # 1.2 Generated tables: objects
        self.Gen_ID_OBJ_Household = self.GeneratedData + self.ID + self.Object + "Household"
        # self.Gen_ID_OBJ_Building = self.GeneratedData + self.ID + self.OBJ + "Building"
        # self.Gen_ID_OBJ_ApplianceGroup = self.GeneratedData + self.ID + self.OBJ + "ApplianceGroup"
        self.Gen_ID_OBJ_SpaceHeating = self.GeneratedData + self.ID + self.Object + "SpaceHeating"
        self.Gen_ID_OBJ_SpaceCooling = self.GeneratedData + self.ID + self.Object + "SpaceCooling"
        self.Gen_ID_OBJ_HotWater = self.GeneratedData + self.ID + self.Object + "HotWater"
        self.Gen_ID_OBJ_PV = self.GeneratedData + self.ID + self.Object + "PV"
        self.Gen_ID_OBJ_Battery = self.GeneratedData + self.ID + self.Object + "Battery"
        self.Gen_ID_OBJ_ElectricVehicle = self.GeneratedData + self.ID + self.Object + "ElectricVehicle"

        # 2 Scenarios
        # 2.1 Exogenous tables
        self.Sce_TimeStructure = self.Scenario + "TimeStructure"  # base year: 2010
        self.Sce_Irradiation = self.Scenario + "Irradiation" # from helioclim dataset for year 2010
        self.Sce_Temperature = self.Scenario + "Temperature" # from MERRA2 dataset for year 2010

        # Generated scenario tables
        # self.Gen_ID_Sce_Environment

        # ------------------------------
        # Part II: Investment simulation
        # ------------------------------








