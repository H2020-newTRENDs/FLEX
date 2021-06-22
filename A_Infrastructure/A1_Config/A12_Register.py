
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

        self.ID_GridInfrastructure = self.ID + "GridInfrastructure"
        self.ID_BuildingMassTemperature = self.ID + "BuildingMassTemperature"


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
        #self.Sce_ID_LifestyleType = self.Scenario + self.ID + "LifestyleType"
        self.Sce_ID_TargetTemperatureType = self.Scenario + self.ID + "TargetTemperatureType"

        self.Sce_ID_BaseElectricityProfileType = self.Scenario + self.ID + "BaseElectricityProfileType"
        self.Sce_ID_ElectricityPriceType = self.Scenario + self.ID + "ElectricityPriceType"
        self.Sce_ID_FeedinTariffType = self.Scenario + self.ID + "FeedinTariffType"
        self.Sce_ID_HotWaterProfileType = self.Scenario + self.ID + 'HotWaterProfileType'
        self.Sce_ID_PhotovoltaicProfileType = self.Scenario + self.ID + 'PhotovoltaicProfileType'
        self.Sce_ID_EnergyCostType = self.Scenario+ self.ID + 'EnergyCostType'

        self.Sce_Weather_Radiation = self.Scenario + "Weather_Radiation" # from helioclim dataset for year 2010
        self.Sce_Weather_Temperature = self.Scenario + "Weather_Temperature" # from MERRA2 dataset for year 2010

        self.Sce_Demand_DishWasher = self.Scenario + "Demand_DishWasher"
        self.Sce_Demand_Dryer = self.Scenario + "Demand_Dryer"
        self.Sce_Demand_WashingMachine = self.Scenario + "Demand_WashingMachine"
        self.Sce_Demand_ElectricVehicleBehavior = self.Scenario + "Demand_ElectricVehicleBehavior"
        self.Sce_Demand_BaseElectricityProfile = self.Scenario + "Demand_BaseElectricityProfile"

        self.Sce_Price_HourlyElectricityPrice = self.Scenario + "Price_HourlyElectricityPrice"
        self.Sce_Price_HourlyFeedinTariff = self.Scenario + "Price_HourlyFeedinTariff"
        self.Sce_Price_EnergyCost = self.Scenario + "Price_EnergyCost"

        self.Sce_HeatPump_COPCurve = self.Scenario + "HeatPump_COPCurve"


        #self.Sce_Weather_Temperature_test = self.Scenario + 'Weather_Temperature_test'


        # Generated scenario tables
        self.Gen_Sce_ID_Environment = self.GeneratedData + self.Scenario + self.ID + "Environment"

        self.Gen_Sce_DishWasherHours = self.GeneratedData + self.Scenario + "DishWasherHours"
        self.Gen_Sce_WashingMachineHours = self.GeneratedData + self.Scenario + "WashingMachineHours"
        self.Gen_Sce_CarAtHomeHours = self.GeneratedData + self.Scenario + 'CarAtHomeHours'

        self.Gen_Sce_Technology_HeatPumpCOP = self.GeneratedData +self.Scenario + "Technology_HeatPumpCOP"
        self.Gen_Sce_HotWaterProfile = self.GeneratedData +self.Scenario + 'HotWaterProfile'
        self.Gen_Sce_PhotovoltaicProfile = self.GeneratedData +self.Scenario + 'PhotovoltaicProfile'  # based on 1kWp
        self.Gen_Sce_Weather_Radiation_SkyDirections = self.GeneratedData +self.Scenario + 'Weather_Radiation_SkyDirections'  # Philipp
        self.Gen_Sce_HeatPump_HourlyCOP = self.GeneratedData +self.Scenario + "HeatPump_HourlyCOP"



        # Gen_Sce_SolarRadiationDirections...

        # Result tables
        self.Res_MinimizedOperationCost = self.Result + "MinimizedOperationCost" #3: col: IDHH, col IDEnvi, col: cost
        self.Res_SystemOperation = self.Result + "SystemOperation" #4: col: IDHH, col IDEnvi, , col: IDHour, col: technologies
        self.Res_YearlyValues = self.Result + "YearlyValues" # yearly sum of technology demands
        self.Res_HourlyValues = self.Result + 'HourlyValues' # HourlyValue of each variable, technology

        # ------------------------------
        # Part II: Investment simulation
        # ------------------------------








