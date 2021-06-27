# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import numpy as np
from A_Infrastructure.A2_Register import REG
from A_Infrastructure.A3_DB import DB

class DataCollector:

    def __init__(self, conn):
        self.Conn = conn
        self.AgentVarList = ["Period", "ID", "Account"]
        self.AgentVarDataType = {"Period": "INTEGER",
                                 "ID": "INTEGER",
                                 "Account": "REAL"}
        self.AgentVar = []
        self.EnvironmentVarList = ["Period", "TotalWealth", "Gini"]
        self.EnvironmentVarDataType = {"Period": "INTEGER",
                                       "TotalWealth": "REAL",
                                       "Gini": "REAL"}
        self.EnvironmentVar = []

    def extract_Result2Array(self, result_DictValues):

        result_array = np.nan_to_num(np.array(list(result_DictValues.values()), dtype=np.float), nan=0)
        
        return result_array

    def save_AgentData(self):
        DB().write_DataFrame(self.AgentVar, REG().Res_AgentPara + "_S" + str(self.ID_Scenario),
                             self.AgentVarList, self.Conn, dtype=self.AgentVarDataType)

        return None

        # # extract variable values
        # Q_TankHeatingHeatPump_array = self.extract_Result2Array(instance.Q_TankHeating.extract_values())/1000 #kWh
        # Q_TankHeatingHeatingElement_array = self.extract_Result2Array(instance.Q_HeatingElement.extract_values())/1000 #kWh
        # SpaceHeatingHourlyCOP_array = self.extract_Result2Array(instance.SpaceHeatingHourlyCOP.extract_values())
        # E_TankHeatingHeatPump_array = Q_TankHeatingHeatPump_array / SpaceHeatingHourlyCOP_array
        # Q_RoomHeating_array = self.extract_Result2Array(instance.Q_RoomHeating.extract_values())/1000 #kWh
        # Q_RoomCooling_array = self.extract_Result2Array(instance.Q_RoomCooling.extract_values())/1000 #kWh
        # E_RoomCooling_array = Q_RoomCooling_array / Household.SpaceCooling.SpaceCoolingEfficiency
        # Q_Solar_array = self.extract_Result2Array(instance.Q_Solar.extract_values())/1000 #kWh
        # E_Grid_array = self.extract_Result2Array(instance.Grid.extract_values())
        # E_BaseLoad_array = self.extract_Result2Array(instance.LoadProfile.extract_values())
        # E_PV_array = self.extract_Result2Array(instance.PhotovoltaicProfile.extract_values())
        # Q_HW1_array = self.extract_Result2Array(instance.HWPart1.extract_values())
        # E_HW1_array = Q_HW1_array / SpaceHeatingHourlyCOP_array
        # Q_HW2_array = self.extract_Result2Array(instance.HWPart2.extract_values())
        # HotWaterHourlyCOP_array = self.extract_Result2Array(instance.HotWaterHourlyCOP.extract_values())
        # E_HW2_array = Q_HW2_array / HotWaterHourlyCOP_array
        # E_Grid2Load_array = self.extract_Result2Array(instance.Grid2Load.extract_values())
        # E_Grid2EV_array = self.extract_Result2Array(instance.Grid2EV.extract_values())
        # E_Grid2Bat_array = self.extract_Result2Array(instance.Grid2Bat.extract_values())
        # E_PV2Load_array = self.extract_Result2Array(instance.PV2Load.extract_values())
        # E_PV2Bat_array = self.extract_Result2Array(instance.PV2Bat.extract_values())
        # E_PV2Grid_array = self.extract_Result2Array(instance.PV2Grid.extract_values())
        # E_PV2EV_array = self.extract_Result2Array(instance.PV2EV.extract_values())
        # E_Load_array = self.extract_Result2Array(instance.Load.extract_values())
        # E_Feedin_array = self.extract_Result2Array(instance.Feedin.extract_values())
        # E_BatCharge_array = self.extract_Result2Array(instance.BatCharge.extract_values())
        # E_BatDischarge_array = self.extract_Result2Array(instance.BatDischarge.extract_values())
        # E_Bat2Load_array = self.extract_Result2Array(instance.Bat2Load.extract_values())
        # E_Bat2EV_array = self.extract_Result2Array(instance.Bat2EV.extract_values())
        # E_EVCharge_array = self.extract_Result2Array(instance.EVCharge.extract_values())
        # E_EVDischarge_array = self.extract_Result2Array(instance.EVDischarge.extract_values())
        # E_EV2Load_array = self.extract_Result2Array(instance.EV2Load.extract_values())
        # E_EV2Bat_array = self.extract_Result2Array(instance.EV2Bat.extract_values())
        # E_DishWasher1_array = self.extract_Result2Array(instance.DishWasher1.extract_values())
        # E_DishWasher2_array = self.extract_Result2Array(instance.DishWasher2.extract_values())
        # E_DishWasher3_array = self.extract_Result2Array(instance.DishWasher3.extract_values())
        # E_DishWasher_array = E_DishWasher1_array + E_DishWasher2_array + E_DishWasher3_array
        # E_WashingMachine1_array = self.extract_Result2Array(instance.WashingMachine1.extract_values())
        # E_WashingMachine2_array = self.extract_Result2Array(instance.WashingMachine2.extract_values())
        # E_WashingMachine3_array = self.extract_Result2Array(instance.WashingMachine3.extract_values())
        # E_WashingMachine_array = E_WashingMachine1_array + E_WashingMachine2_array+ E_WashingMachine3_array
        # E_Dryer1_array = self.extract_Result2Array(instance.Dryer1.extract_values())
        # E_Dryer2_array = self.extract_Result2Array(instance.Dryer2.extract_values())
        # E_Dryer_array = E_Dryer1_array + E_Dryer2_array
        # E_SmartAppliances_array = E_DishWasher_array + E_WashingMachine_array + E_Dryer_array




        #
        # Yearly_E_UseOfPV = Yearly_E_PV - Yearly_E_Feedin
        # Yearly_E_ElectricityDemand = Yearly_E_Load + Yearly_E_EVDriving
        #
        # SelfConsumption = round(Yearly_E_UseOfPV / Yearly_E_PV, 2)
        # SelfSufficiency = round(Yearly_E_UseOfPV / (Yearly_E_ElectricityDemand), 2)
        #
        # YearlyDemandValues = [Household.ID,
        #                       Environment.ID,
        #                       Cost,
        #                       SelfConsumption,
        #                       SelfSufficiency,
        #                       APF,
        #                       Yearly_E_ElectricityDemand,
        #                       Yearly_Q_TankHeatingHP,
        #                       Yearly_Q_TankHeatingHE,
        #                       Yearly_E_TankHeatingHP,
        #                       Yearly_Q_RoomHeating,
        #                       Yearly_Q_SolarGains,
        #                       Yearly_Q_RoomCooling,
        #                       Yearly_E_RoomCooling,
        #                       Yearly_Q_HW,
        #                       Yearly_E_HW,
        #                       Yearly_E_Grid,
        #                       Yearly_E_Grid2Load,
        #                       Yearly_E_Grid2EV,
        #                       Yearly_E_Grid2Bat,
        #                       Yearly_E_BaseLoad,
        #                       Yearly_E_PV,
        #                       Yearly_E_PV2Load,
        #                       Yearly_E_PV2Bat,
        #                       Yearly_E_PV2EV,
        #                       Yearly_E_PV2Grid,
        #                       Yearly_E_EVCharge,
        #                       Yearly_E_EVDriving,
        #                       Yearly_E_EVDischarge,
        #                       Yearly_E_EV2Bat,
        #                       Yearly_E_EV2Load,
        #                       Yearly_E_BatCharge,
        #                       Yearly_E_BatDischarge,
        #                       Yearly_E_Bat2Load,
        #                       Yearly_E_Bat2EV,
        #                       Yearly_E_DishWasher,
        #                       Yearly_E_WashingMachine,
        #                       Yearly_E_Dryer,
        #                       Yearly_E_SmartAppliances]







        # # Generates SystemOperation
        # HouseholdTechnologies = [Household.ID,
        #                          Environment.ID,
        #                          Cost,
        #
        #                          Household.Name_AgeGroup,
        #
        #                          Household.Building.name,
        #                          Household.Building.Af,
        #                          Household.Building.hwb_norm1,
        #
        #                          Household.PV.PVPower,
        #                          Household.Battery.Capacity,
        #                          Household.Battery.Grid2Battery,
        #
        #                          Household.SpaceHeating.Name_SpaceHeatingBoilerType,
        #                          Household.SpaceHeating.TankSize,
        #
        #                          Household.SpaceCooling.SpaceCoolingPower,
        #                          Household.SpaceCooling.AdoptionStatus,
        #
        #                          Household.ElectricVehicle.Name_ElectricVehicleType,
        #                          Household.ElectricVehicle.BatterySize,
        #                          Household.ElectricVehicle.ConsumptionPer100km,
        #                          Household.ElectricVehicle.V2B,
        #
        #                          Household.ApplianceGroup.DishWasherPower,
        #                          Household.ApplianceGroup.DishWasherShifting,
        #                          Household.ApplianceGroup.DishWasherAdoption,
        #
        #                          Household.ApplianceGroup.WashingMachinePower,
        #                          Household.ApplianceGroup.WashingMachineShifting,
        #                          Household.ApplianceGroup.WashingMachineAdoption,
        #
        #                          Household.ApplianceGroup.DryerPower,
        #                          Household.ApplianceGroup.DryerAdoption]






        # # MinimizedCost
        # TargetTable_columnsMinimizedCost = ['ID_Household', 'ID_Environment', "TotalCost"]
        # DB().write_DataFrame(TargetTable_MinimizedCost, REG().Res_MinimizedOperationCost,
        #                      TargetTable_columnsMinimizedCost, self.Conn)
        #
        # # SystemOperation
        # TargetTable_columnsSystemOperation = ['ID_Household', 'ID_Environment', 'YearlyCost',
        #                                       'AgeGroup', 'Building_Name', 'Building_Af', 'Building_hwbNorm', 'PVPower',
        #                                       'SBS_Capacity', 'Grid2Battery', 'SpaceHeatingBoilerType', 'TankSize',
        #                                       'SpaceCoolingPower', 'SpaceCooling_AdoptionStatus', 'EV_Type',
        #                                       'EV_BatteryCapacity', 'EV_ConsumptionPer100km', 'EV_V2BStatus',
        #                                       'DishWasherPower', 'DishWasherShifting',
        #                                       'DishWasherAdoption', 'WashingMachinePower', 'WashingMachineShifting',
        #                                       'WashingMachineAdoption', 'DryerPower', 'DryerAdoption']
        # DB().write_DataFrame(TargetTable_SystemOperation, REG().Res_SystemOperation,
        #                      TargetTable_columnsSystemOperation, self.Conn)
        #
        # # YearlyValues
        # TargetTable_columnsYearlyValues = ['ID_Household', 'ID_Environment', 'YearlyCost', 'SelfConsumptionRate',
        #                                    'SelfSufficiencyRate', 'APF', 'Yearly_E_ElectricityDemand',
        #                                    'Yearly_Q_TankHeatingHP','Yearly_Q_TankHeatingHE', 'Yearly_E_TankHeatingHP',
        #                                    'Yearly_Q_RoomHeating', 'Yearly_Q_SolarGains', 'Yearly_Q_RoomCooling',
        #                                    'Yearly_E_RoomCooling', 'Yearly_Q_HW', 'Yearly_E_HW', 'Yearly_E_Grid',
        #                                    'Yearly_E_Grid2Load', 'Yearly_E_Grid2EV', 'Yearly_E_Grid2Bat',
        #                                    'Yearly_E_BaseLoad', 'Yearly_E_PV', 'Yearly_E_PV2Load', 'Yearly_E_PV2Bat',
        #                                    'Yearly_E_PV2EV', 'Yearly_E_PV2Grid', 'Yearly_E_EVCharge',
        #                                    'Yearly_E_EVDriving', 'Yearly_E_EVDischarge',
        #                                    'Yearly_E_EV2Bat', 'Yearly_E_EV2Load', 'Yearly_E_BatCharge',
        #                                    'Yearly_E_BatDischarge', 'Yearly_E_Bat2Load', 'Yearly_E_Bat2EV',
        #                                    'Yearly_E_DishWasher', 'Yearly_E_WashingMachine', 'Yearly_E_Dryer',
        #                                    'Yearly_E_SmartAppliances']
        # DB().write_DataFrame(TargetTable_YearlyValues, REG().Res_YearlyValues,
        #                      TargetTable_columnsYearlyValues, self.Conn)
