# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import numpy as np
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_REG import REG_Table
from A_Infrastructure.A2_REG import REG_Var
from A_Infrastructure.A3_DB import DB

class DataCollector:

    def __init__(self, conn):
        self.Conn = conn
        self.VAR = REG_Var()
        self.SystemOperationHour_Column = {self.VAR.ID_Household: "INTEGER",
                                           self.VAR.ID_Environment: "INTEGER",
                                           self.VAR.ID_Hour: "INTEGER",

                                           self.VAR.E_BaseElectricityLoad: "REAL",
                                           self.VAR.E_DishWasher: "REAL",
                                           self.VAR.E_WashingMachine: "REAL",
                                           self.VAR.E_Dryer: "REAL",
                                           self.VAR.E_SmartAppliance: "REAL",

                                           self.VAR.Q_HeatPump: "REAL",
                                           self.VAR.HeatPumpPerformanceFactor: "REAL",
                                           self.VAR.E_HeatPump: "REAL",
                                           self.VAR.E_AmbientHeat: "REAL",
                                           self.VAR.E_HeatingElement: "REAL",
                                           self.VAR.E_RoomHeating: "REAL",

                                           self.VAR.Q_RoomCooling: "REAL",
                                           self.VAR.E_RoomCooling: "REAL",

                                           self.VAR.Q_HotWater: "REAL",
                                           self.VAR.E_HotWater: "REAL",

                                           self.VAR.E_Grid: "REAL",
                                           self.VAR.E_Grid2Load: "REAL",
                                           self.VAR.E_Grid2Battery: "REAL",
                                           self.VAR.E_Grid2EV: "REAL",

                                           self.VAR.E_PV: "REAL",
                                           self.VAR.E_PV2Load: "REAL",
                                           self.VAR.E_PV2Battery: "REAL",
                                           self.VAR.E_PV2EV: "REAL",
                                           self.VAR.E_PV2Grid: "REAL",

                                           self.VAR.E_BatteryCharge: "REAL",
                                           self.VAR.E_BatteryDischarge: "REAL",
                                           self.VAR.E_Battery2Load: "REAL",
                                           self.VAR.E_Battery2EV: "REAL",
                                           self.VAR.BatteryStateOfCharge: "REAL",

                                           self.VAR.E_EVCharge: "REAL",
                                           self.VAR.E_EV2Discharge: "REAL",
                                           self.VAR.E_EV2Load: "REAL",
                                           self.VAR.E_EV2Battery: "REAL",
                                           self.VAR.EVStateOfCharge: "REAL",

                                           self.VAR.TotalElectricityDemand: "REAL",
                                           }
        self.SystemOperationHour_ValueList = []
        self.SystemOperationYear_Column = {} # also with the information of households?
        self.SystemOperationYear_ValueList = []

    def extract_Result2Array(self, result_DictValues):
        result_array = np.nan_to_num(np.array(list(result_DictValues.values()), dtype=np.float), nan=0)
        return result_array

    def collect_OptimizationResult(self, Household, Environment, PyomoModelInstance):
        # extract variable values
        E_BaseLoad_array = self.extract_Result2Array(PyomoModelInstance.BaseLoadProfile.extract_values())
        E_DishWasher1_array = self.extract_Result2Array(PyomoModelInstance.DishWasher1.extract_values())
        E_DishWasher2_array = self.extract_Result2Array(PyomoModelInstance.DishWasher2.extract_values())
        E_DishWasher3_array = self.extract_Result2Array(PyomoModelInstance.DishWasher3.extract_values())
        E_DishWasher_array = E_DishWasher1_array + E_DishWasher2_array + E_DishWasher3_array
        E_WashingMachine1_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine1.extract_values())
        E_WashingMachine2_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine2.extract_values())
        E_WashingMachine3_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine3.extract_values())
        E_WashingMachine_array = E_WashingMachine1_array + E_WashingMachine2_array + E_WashingMachine3_array
        E_Dryer1_array = self.extract_Result2Array(PyomoModelInstance.Dryer1.extract_values())
        E_Dryer2_array = self.extract_Result2Array(PyomoModelInstance.Dryer2.extract_values())
        E_Dryer_array = E_Dryer1_array + E_Dryer2_array
        E_SmartAppliances_array = E_DishWasher_array + E_WashingMachine_array + E_Dryer_array

        Q_TankHeatingHeatPump_array = self.extract_Result2Array(PyomoModelInstance.Q_TankHeating.extract_values())/1000 #kWh
        SpaceHeatingHourlyCOP_array = self.extract_Result2Array(PyomoModelInstance.SpaceHeatingHourlyCOP.extract_values())
        E_TankHeatingHeatPump_array = Q_TankHeatingHeatPump_array / SpaceHeatingHourlyCOP_array
        Q_TankHeatingHeatingElement_array = self.extract_Result2Array(PyomoModelInstance.Q_HeatingElement.extract_values()) / 1000  # kWh
        Q_RoomHeating_array = self.extract_Result2Array(PyomoModelInstance.Q_RoomHeating.extract_values())/1000 #kWh

        Q_RoomCooling_array = self.extract_Result2Array(PyomoModelInstance.Q_RoomCooling.extract_values())/1000 #kWh
        E_RoomCooling_array = Q_RoomCooling_array / Household.SpaceCooling.SpaceCoolingEfficiency

        Q_HW1_array = self.extract_Result2Array(PyomoModelInstance.HWPart1.extract_values())
        E_HW1_array = Q_HW1_array / SpaceHeatingHourlyCOP_array
        Q_HW2_array = self.extract_Result2Array(PyomoModelInstance.HWPart2.extract_values())
        HotWaterHourlyCOP_array = self.extract_Result2Array(PyomoModelInstance.HotWaterHourlyCOP.extract_values())
        E_HW2_array = Q_HW2_array / HotWaterHourlyCOP_array
        Q_HotWater_array = Q_HW1_array + Q_HW2_array
        E_HotWater_array = E_HW1_array + E_HW2_array

        E_Grid_array = self.extract_Result2Array(PyomoModelInstance.Grid.extract_values())
        E_Grid2Load_array = self.extract_Result2Array(PyomoModelInstance.Grid2Load.extract_values())
        E_Grid2Bat_array = self.extract_Result2Array(PyomoModelInstance.Grid2Bat.extract_values())
        E_Grid2EV_array = self.extract_Result2Array(PyomoModelInstance.Grid2EV.extract_values())

        E_PV_array = self.extract_Result2Array(PyomoModelInstance.PhotovoltaicProfile.extract_values())
        E_PV2Load_array = self.extract_Result2Array(PyomoModelInstance.PV2Load.extract_values())
        E_PV2Bat_array = self.extract_Result2Array(PyomoModelInstance.PV2Bat.extract_values())
        E_PV2EV_array = self.extract_Result2Array(PyomoModelInstance.PV2EV.extract_values())
        E_PV2Grid_array = self.extract_Result2Array(PyomoModelInstance.PV2Grid.extract_values())

        E_BatCharge_array = self.extract_Result2Array(PyomoModelInstance.BatCharge.extract_values())
        E_BatDischarge_array = self.extract_Result2Array(PyomoModelInstance.BatDischarge.extract_values())
        E_Bat2Load_array = self.extract_Result2Array(PyomoModelInstance.Bat2Load.extract_values())
        E_Bat2EV_array = self.extract_Result2Array(PyomoModelInstance.Bat2EV.extract_values())
        E_BatSoC_array = self.extract_Result2Array(PyomoModelInstance.BatSoC.extract_values())

        E_EVCharge_array = self.extract_Result2Array(PyomoModelInstance.EVCharge.extract_values())
        E_EVDischarge_array = self.extract_Result2Array(PyomoModelInstance.EVDischarge.extract_values())
        E_EV2Load_array = self.extract_Result2Array(PyomoModelInstance.EV2Load.extract_values())
        E_EV2Bat_array = self.extract_Result2Array(PyomoModelInstance.EV2Bat.extract_values())
        E_EVSoC_array = self.extract_Result2Array(PyomoModelInstance.EVSoC.extract_values())

        E_Load_array = self.extract_Result2Array(PyomoModelInstance.Load.extract_values())

        for t in range(0, CONS().OptimizationHourHorizon):
            self.SystemOperationHour_ValueList.append([Household.ID,
                                                       Environment["ID"],
                                                       t+1,
                                                       E_BaseLoad_array[t],
                                                       E_DishWasher_array[t],
                                                       E_WashingMachine_array[t],
                                                       E_Dryer_array[t],
                                                       E_SmartAppliances_array[t],

                                                       Q_TankHeatingHeatPump_array[t],
                                                       SpaceHeatingHourlyCOP_array[t],
                                                       E_TankHeatingHeatPump_array[t],
                                                       Q_TankHeatingHeatPump_array[t] - E_TankHeatingHeatPump_array[t],
                                                       Q_TankHeatingHeatingElement_array[t],
                                                       Q_RoomHeating_array[t],

                                                       Q_RoomCooling_array[t],
                                                       E_RoomCooling_array[t],

                                                       Q_HotWater_array[t],
                                                       E_HotWater_array[t],

                                                       E_Grid_array[t],
                                                       E_Grid2Load_array[t],
                                                       E_Grid2Bat_array[t],
                                                       E_Grid2EV_array[t],

                                                       E_PV_array[t],
                                                       E_PV2Load_array[t],
                                                       E_PV2Bat_array[t],
                                                       E_PV2EV_array[t],
                                                       E_PV2Grid_array[t],

                                                       E_BatCharge_array[t],
                                                       E_BatDischarge_array[t],
                                                       E_Bat2Load_array[t],
                                                       E_Bat2EV_array[t],
                                                       E_BatSoC_array[t],

                                                       E_EVCharge_array[t],
                                                       E_EVDischarge_array[t],
                                                       E_EV2Load_array[t],
                                                       E_EV2Bat_array[t],
                                                       E_EVSoC_array[t],

                                                       E_Load_array[t]
                                                       ])

    def save_OptimizationResult(self):
        DB().write_DataFrame(self.SystemOperationHour_ValueList, REG_Table().Res_SystemOperationHour,
                             self.SystemOperationHour_Column.keys(), self.Conn, dtype=self.SystemOperationHour_Column)




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

