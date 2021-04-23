
class REG:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.OBJ = "OBJ_"
        self.ExogenousData = "Exo_"
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
        self.Exo_GlobalParameterValue = self.ExogenousData + "GlobalParameterValue"
        self.Exo_BaseYearIrradiation = self.ExogenousData + "BaseYearIrradiation" # from helioclim dataset for year 2010
        self.Exo_BaseYearTemperature = self.ExogenousData + "BaseYearTemperature" # from MERRA2 dataset for year 2010
        self.Exo_EnergyCarrierPrice = self.ExogenousData + "EnergyCarrierPrice"
        self.Exo_HourlyElectricityPrice = self.ExogenousData + "HourlyElectricityPrice"
        self.Exo_HourlyFeedinTariff = self.ExogenousData + "HourlyFeedinTariff"
        self.Exo_HouseholdPeopleAverage = self.ExogenousData + "HouseholdPeopleAverage"
        self.Exo_HouseholdPeopleEquivalent = self.ExogenousData + "HouseholdPeopleEquivalent"
        self.Exo_BaseElectricityProfile = self.ExogenousData + "BaseElectricityProfile"
        self.Exo_WashingMachinePower = self.ExogenousData + "WashingMachinePower"
        self.Exo_WashingMachineDuration = self.ExogenousData + "WashingMachineDuration"
        self.Exo_WashingMachineYearlyUse = self.ExogenousData + "WashingMachineYearlyUse"
        self.Exo_DryerPower = self.ExogenousData + "DryerPower"
        self.Exo_DryerDuration = self.ExogenousData + "DryerDuration"
        self.Exo_DryerYearlyUse = self.ExogenousData + "DryerYearlyUse"
        self.Exo_DishWasherPower = self.ExogenousData + "DishWasherPower"
        self.Exo_DishWasherDuration = self.ExogenousData + "DishWasherDuration"
        self.Exo_DishWasherYearlyUse = self.ExogenousData + "DishWasherYearlyUse"
        self.Exo_SpaceHeatingTargetTemperature = self.ExogenousData + "SpaceHeatingTargetTemperature"
        self.Exo_SpaceHeatingEfficiency = self.ExogenousData + "SpaceHeatingEfficiency"
        self.Exo_SpaceHeatingTankSize = self.ExogenousData + "SpaceHeatingTankSize"
        self.Exo_SpaceHeatingTankSurfaceArea = self.ExogenousData + "SpaceHeatingTankSurfaceArea"
        self.Exo_SpaceCoolingTargetTemperature = self.ExogenousData + "SpaceCoolingTargetTemperature"
        self.Exo_SpaceCoolingEfficiency = self.ExogenousData + "SpaceCoolingEfficiency"
        self.Exo_HotWaterProfile = self.ExogenousData + "HotWaterProfile"
        self.Exo_HotWaterEfficiency = self.ExogenousData + "HotWaterEfficiency"
        self.Exo_PVSize = self.ExogenousData + "PVSize"
        self.Exo_BatteryCapacity = self.ExogenousData + "BatteryCapacity"
        self.Exo_EVBatteryCapacity = self.ExogenousData + "EVBatteryCapacity"

        # Generated Table
        self.Gen_ID_OBJ_Environment = self.GeneratedData + self.ID + self.OBJ + "Environment"
        self.Gen_ID_OBJ_Household = self.GeneratedData + self.ID + self.OBJ + "Household"
        self.Gen_ID_OBJ_Building = self.GeneratedData + self.ID + self.OBJ + "Building"
        self.Gen_ID_OBJ_Appliances = self.GeneratedData + self.ID + self.OBJ + "Appliances"
        self.Gen_ID_OBJ_SpaceHeating = self.GeneratedData + self.ID + self.OBJ + "SpaceHeating"
        self.Gen_ID_OBJ_SpaceCooling = self.GeneratedData + self.ID + self.OBJ + "SpaceCooling"
        self.Gen_ID_OBJ_HotWater = self.GeneratedData + self.ID + self.OBJ + "HotWater"
        self.Gen_ID_OBJ_PV = self.GeneratedData + self.ID + self.OBJ + "PV"
        self.Gen_ID_OBJ_Battery = self.GeneratedData + self.ID + self.OBJ + "Battery"
        self.Gen_ID_OBJ_ElectricVehicle = self.GeneratedData + self.ID + self.OBJ + "ElectricVehicle"
        self.Gen_WashingMachineUseDay = self.GeneratedData + "WashingMachineUseDay"
        self.Gen_DryerUseDay = self.GeneratedData + "DryerUseDay"
        self.Gen_DishWasherUseDay = self.GeneratedData + "DishWasherUseDay"











