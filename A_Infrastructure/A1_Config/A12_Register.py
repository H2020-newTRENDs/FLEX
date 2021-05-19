
class REG:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.Object = "OBJ_"
        self.Scenario = "Sce_"
        self.GeneratedData = "Gen_"
        self.Result = "Res_"

        # ------------------------------
        # Part I: Operation optimization
        # ------------------------------
        # 1 Objects
        # 1.1 Exogenous tables: ID and parameter
        self.ID_Country = self.ID + "Country"
        self.ID_DayType = self.ID + "DayType"
        self.ID_SeasonType = self.ID + "SeasonType"
        self.ID_EnergyCarrier = self.ID + "EnergyCarrier"
        self.ID_HouseholdType = self.ID + "HouseholdType"
        self.ID_AgeGroup = self.ID + "AgeGroup"
        self.ID_BuildingType = self.ID + "BuildingType"
        self.ID_BuildingAgeClass = self.ID + "BuildingAgeClass"
        self.ID_BuildingOption = self.ID + "BuildingOption"
        self.ID_DishWasherType = self.ID + "DishWasherType"
        self.ID_DryerType = self.ID + "DryerType"
        self.ID_WashingMachineType = self.ID + "WashingMachineType"
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
        self.Gen_OBJ_ID_Household = self.GeneratedData + self.Object + self.ID + "Household"
        self.Gen_OBJ_ID_Building = self.GeneratedData + self.Object + self.ID + "Building"
        self.Gen_OBJ_ID_ApplianceGroup = self.GeneratedData + self.Object + self.ID + "ApplianceGroup"
        self.Gen_OBJ_ID_SpaceHeating = self.GeneratedData + self.Object + self.ID + "SpaceHeating"
        self.Gen_OBJ_ID_SpaceCooling = self.GeneratedData + self.Object + self.ID + "SpaceCooling"
        self.Gen_OBJ_ID_HotWater = self.GeneratedData + self.Object + self.ID + "HotWater"
        self.Gen_OBJ_ID_PV = self.GeneratedData + self.Object + self.ID + "PV"
        self.Gen_OBJ_ID_Battery = self.GeneratedData + self.Object + self.ID + "Battery"
        self.Gen_OBJ_ID_ElectricVehicle = self.GeneratedData + self.Object + self.ID + "ElectricVehicle"

        # 2 Scenarios
        # 2.1 Exogenous tables: ID and parameter
        self.Sce_ID_TimeStructure = self.Scenario + self.ID + "TimeStructure" # base year: 2010
        self.Sce_ID_LifestyleType = self.Scenario + self.ID + "LifestyleType"
        self.Sce_ID_TargetTemperature = self.Scenario + self.ID + "TargetTemperature"
        self.Sce_ID_ElectricVehicleBehavior = self.Scenario + self.ID + "ElectricVehicleBehavior"
        self.Sce_ID_BaseElectricityProfileType = self.Scenario + self.ID + "BaseElectricityProfileType"
        self.Sce_ID_ElectricityPriceType = self.Scenario + self.ID + "ElectricityPriceType"
        self.Sce_ID_FeedinTariffType = self.Scenario + self.ID + "FeedinTariffType"
        self.Sce_Weather_Radiation = self.Scenario + "Weather_Radiation" # from helioclim dataset for year 2010
        self.Sce_Weather_Temperature = self.Scenario + "Weather_Temperature" # from MERRA2 dataset for year 2010
        self.Sce_Technology_HeatPumpCOP = self.Scenario + "Technology_HeatPumpCOP"
        self.Sce_Demand_BaseElectricityProfile = self.Scenario + "Demand_BaseElectricityProfile"
        self.Sce_Demand_DishWasher = self.Scenario + "Demand_DishWasher"
        self.Sce_Demand_Dryer = self.Scenario + "Demand_Dryer"
        self.Sce_Demand_WashingMachine = self.Scenario + "Demand_WashingMachine"
        self.Sce_Price_HourlyElectricityPrice = self.Scenario + "Price_HourlyElectricityPrice"
        self.Sce_Price_HourlyFeedinTariff = self.Scenario + "Price_HourlyFeedinTariff"
        self.Sce_Demand_BaseHotWaterProfile = self.Scenario + 'Demand_BaseHotWaterProfile' # based on 1 person; part 1
        self.Sce_HotWaterProfile = self.Scenario + 'HotWaterProfile' # based on 1 person; part 2; hourly profile
        self.Sce_BasePhotovoltaicProfile = self.Scenario + 'BasePhotovoltaicProfile' # based on 1kWp; hourly profile


        self.Sce_Weather_Temperature_test = self.Scenario + 'Weather_Temperature_test' # only a test with 24 values


        # Generated scenario tables
        self.Gen_Sce_ID_Environment = self.GeneratedData + self.Scenario + self.ID + "Environment"
        self.Gen_Sce_DishWasherUseDays = self.GeneratedData + self.Scenario + "DishWasherWorkingDays"
        self.Gen_Sce_DryerUseDays = self.GeneratedData + self.Scenario + "DryerWorkingDays"
        self.Gen_Sce_WashingMachineUseDays = self.GeneratedData + self.Scenario + "WashingMachineWorkingDays"

        # Result tables
        self.Res_MinimizedOperationCost = self.Result + "MinimizedOperationCost" #3: col: IDHH, col IDEnvi, col: cost
        self.Res_SystemOperation = self.Result + "SystemOperation" #4: col: IDHH, col IDEnvi, , col: IDHour, col: technologies


        # ------------------------------
        # Part II: Investment simulation
        # ------------------------------








