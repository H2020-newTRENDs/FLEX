
class REG:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.OBJ = "OBJ_"
        self.Parameter = "Par_"
        self.ScenarioData = "Sce_"
        self.GeneratedData = "Gen_"

        # ID Table
        self.ID_Country = self.ID + "Country"
        self.ID_BaseYearTimeStructure = self.ID + "BaseYearTimeStructure" # base year: 2010
        self.ID_EnergyCarrier = self.ID + "EnergyCarrier"
        self.ID_ElectricityPriceScenario = self.ID + "ElectricityPriceScenario"
        self.ID_FeedinTariffScenario = self.ID + "FeedinTariffScenario"
        self.ID_BuildingType = self.ID + "BuildingType"
        self.ID_BuildingAgeClass = self.ID + "BuildingAgeClass"
        self.ID_HouseholdType = self.ID + "HouseholdType"
        self.ID_AgeGroup = self.ID + "AgeGroup"
        self.ID_LifeStyleType = self.ID + "LifeStyleType"
        self.ID_BaseElectricityProfileType = self.ID + "BaseElectricityProfileType"
        self.ID_WashingMachineAdoptionStatus = self.ID + "WashingMachineAdoptionStatus"
        self.ID_WashingMachineType = self.ID + "WashingMachineType"
        self.ID_DryerAdoptionStatus = self.ID + "DryerAdoptionStatus"
        self.ID_DryerType = self.ID + "DryerType"
        self.ID_DishWasherAdoptionStatus = self.ID + "DishWasherAdoptionStatus"
        self.ID_DishWasherType = self.ID + "DishWasherType"
        self.ID_SpaceHeatingAdoptionStatus = self.ID + "SpaceHeatingAdoptionStatus"
        self.ID_SpaceHeatingType = self.ID + "SpaceHeatingType"
        self.ID_SpaceHeatingTankType = self.ID + "SpaceHeatingTankType"
        self.ID_SpaceCoolingAdoptionStatus = self.ID + "SpaceCoolingAdoptionStatus"
        self.ID_SpaceCoolingType = self.ID + "SpaceCoolingType"
        self.ID_HotWaterAdoptionStatus = self.ID + "HotWaterAdoptionStatus"
        self.ID_HotWaterType = self.ID + "HotWaterType"
        self.ID_PVAdoptionStatus = self.ID + "PVAdoptionStatus"
        self.ID_PVType = self.ID + "PVType"
        self.ID_BatteryAdoptionStatus = self.ID + "BatteryAdoptionStatus"
        self.ID_BatteryType = self.ID + "BatteryType"
        self.ID_ElectricVehicleAdoptionStatus = self.ID + "ElectricVehicleAdoptionStatus"
        self.ID_ElectricVehicleType = self.ID + "ElectricVehicleType"

        # Exogenous Table
        self.Par_GlobalParameterValue = self.Parameter + "GlobalParameterValue"
        self.Par_BaseYearIrradiation = self.Parameter + "BaseYearIrradiation" # from helioclim dataset for year 2010
        self.Par_BaseYearTemperature = self.Parameter + "BaseYearTemperature" # from MERRA2 dataset for year 2010
        self.Par_EnergyCarrierPrice = self.Parameter + "EnergyCarrierPrice"
        self.Par_HourlyElectricityPrice = self.Parameter + "HourlyElectricityPrice"
        self.Par_HourlyFeedinTariff = self.Parameter + "HourlyFeedinTariff"
        self.Par_HouseholdPeopleAverage = self.Parameter + "HouseholdPeopleAverage"
        self.Par_HouseholdPeopleEquivalent = self.Parameter + "HouseholdPeopleEquivalent"
        self.Par_BaseElectricityProfile = self.Parameter + "BaseElectricityProfile"
        self.Par_WashingMachinePower = self.Parameter + "WashingMachinePower"
        self.Par_WashingMachineDuration = self.Parameter + "WashingMachineDuration"
        self.Par_WashingMachineYearlyUse = self.Parameter + "WashingMachineYearlyUse"
        self.Par_DryerPower = self.Parameter + "DryerPower"
        self.Par_DryerDuration = self.Parameter + "DryerDuration"
        self.Par_DryerYearlyUse = self.Parameter + "DryerYearlyUse"
        self.Par_DishWasherPower = self.Parameter + "DishWasherPower"
        self.Par_DishWasherDuration = self.Parameter + "DishWasherDuration"
        self.Par_DishWasherYearlyUse = self.Parameter + "DishWasherYearlyUse"
        self.Par_SpaceHeatingTargetTemperature = self.Parameter + "SpaceHeatingTargetTemperature"
        self.Par_SpaceHeatingEfficiency = self.Parameter + "SpaceHeatingEfficiency"
        self.Par_SpaceHeatingTankSize = self.Parameter + "SpaceHeatingTankSize"
        self.Par_SpaceHeatingTankSurfaceArea = self.Parameter + "SpaceHeatingTankSurfaceArea"
        self.Par_SpaceCoolingTargetTemperature = self.Parameter + "SpaceCoolingTargetTemperature"
        self.Par_SpaceCoolingEfficiency = self.Parameter + "SpaceCoolingEfficiency"
        self.Par_HotWaterProfile = self.Parameter + "HotWaterProfile"
        self.Par_HotWaterEfficiency = self.Parameter + "HotWaterEfficiency"
        self.Par_PVSize = self.Parameter + "PVSize"
        self.Par_BatteryCapacity = self.Parameter + "BatteryCapacity"
        self.Par_EVBatteryCapacity = self.Parameter + "EVBatteryCapacity"

        # Generated Table
        self.Gen_ID_OBJ_Household = self.GeneratedData + self.ID + self.OBJ + "Household"
        self.Gen_ID_OBJ_Building = self.GeneratedData + self.ID + self.OBJ + "Building"
        self.Gen_ID_OBJ_ApplianceGroup = self.GeneratedData + self.ID + self.OBJ + "ApplianceGroup"
        self.Gen_ID_OBJ_SpaceHeating = self.GeneratedData + self.ID + self.OBJ + "SpaceHeating"
        self.Gen_ID_OBJ_SpaceCooling = self.GeneratedData + self.ID + self.OBJ + "SpaceCooling"
        self.Gen_ID_OBJ_HotWater = self.GeneratedData + self.ID + self.OBJ + "HotWater"
        self.Gen_ID_OBJ_PV = self.GeneratedData + self.ID + self.OBJ + "PV"
        self.Gen_ID_OBJ_Battery = self.GeneratedData + self.ID + self.OBJ + "Battery"
        self.Gen_ID_OBJ_ElectricVehicle = self.GeneratedData + self.ID + self.OBJ + "ElectricVehicle"











