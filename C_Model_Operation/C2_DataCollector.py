# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import numpy as np
from A_Infrastructure.A1_CONS import CONS
from C_Model_Operation.C1_REG import REG_Table
from C_Model_Operation.C1_REG import REG_Var
from A_Infrastructure.A2_DB import DB


class DataCollector:

    def __init__(self, conn=DB().create_Connection(CONS().RootDB)):
        self.Conn = conn
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.VAR = REG_Var()
        self.SystemOperationHour_Column = {self.VAR.ID_Household: "INTEGER",
                                           self.VAR.ID_Building: "INTEGER",
                                           self.VAR.ID_Environment: "INTEGER",
                                           self.VAR.ID_Hour: "INTEGER",
                                           self.VAR.ElectricityPrice: "REAL",
                                           self.VAR.FeedinTariff: "REAL",
                                           self.VAR.ID_AgeGroup: "REAL",
                                           self.VAR.ID_SpaceHeatingSystem: "TEXT",
                                           self.VAR.OutsideTemperature: "REAL",

                                           self.VAR.E_BaseElectricityLoad: "REAL",
                                           # self.VAR.E_DishWasher: "REAL",
                                           # self.VAR.E_WashingMachine: "REAL",
                                           # self.VAR.E_Dryer: "REAL",
                                           # self.VAR.E_SmartAppliance: "REAL",

                                           self.VAR.Q_HeatPump: "REAL",
                                           self.VAR.HeatPumpPerformanceFactor: "REAL",
                                           self.VAR.E_HeatPump: "REAL",
                                           self.VAR.E_AmbientHeat: "REAL",
                                           self.VAR.Q_HeatingElement: "REAL",
                                           self.VAR.Q_RoomHeating: "REAL",

                                           self.VAR.RoomTemperature: "REAL",
                                           self.VAR.BuildingMassTemperature: "REAL",
                                           self.VAR.Q_SolarGain: "REAL",

                                           self.VAR.Q_RoomCooling: "REAL",
                                           self.VAR.E_RoomCooling: "REAL",

                                           self.VAR.Q_HotWater: "REAL",
                                           self.VAR.E_HotWater: "REAL",

                                           self.VAR.E_Grid: "REAL",
                                           self.VAR.E_Grid2Load: "REAL",
                                           self.VAR.E_Grid2Battery: "REAL",

                                           self.VAR.E_PV: "REAL",
                                           self.VAR.E_PV2Load: "REAL",
                                           self.VAR.E_PV2Battery: "REAL",
                                           self.VAR.E_PV2Grid: "REAL",

                                           self.VAR.E_BatteryCharge: "REAL",
                                           self.VAR.E_BatteryDischarge: "REAL",
                                           self.VAR.E_Battery2Load: "REAL",
                                           self.VAR.BatteryStateOfCharge: "REAL",

                                           self.VAR.E_Load: "REAL"
                                           }

        self.SystemOperationYear_Column = {self.VAR.ID_Household: "INTEGER",
                                           self.VAR.ID_Building: "INTEGER",
                                           self.VAR.ID_Environment: "INTEGER",
                                           self.VAR.OperationCost: "REAL",

                                           self.VAR.Year_E_BaseElectricityLoad: "REAL",
                                           # self.VAR.Year_E_SmartAppliance: "REAL",
                                           self.VAR.Year_Q_HeatPump: "REAL",
                                           self.VAR.Year_E_HeatPump: "REAL",
                                           self.VAR.Year_HeatPumpPerformanceFactor: "REAL",
                                           self.VAR.Year_E_AmbientHeat: "REAL",
                                           self.VAR.Year_Q_HeatingElement: "REAL",
                                           self.VAR.Year_Q_RoomHeating: "REAL",

                                           self.VAR.Year_Q_RoomCooling: "REAL",
                                           self.VAR.Year_E_RoomCooling: "REAL",

                                           self.VAR.Year_Q_HotWater: "REAL",
                                           self.VAR.Year_E_HotWater: "REAL",

                                           self.VAR.Year_E_Grid: "REAL",
                                           self.VAR.Year_E_Grid2Load: "REAL",
                                           self.VAR.Year_E_Grid2Battery: "REAL",

                                           self.VAR.Year_E_PV: "REAL",
                                           self.VAR.Year_E_PV2Load: "REAL",
                                           self.VAR.Year_E_PV2Battery: "REAL",
                                           self.VAR.Year_E_PV2Grid: "REAL",

                                           self.VAR.Year_E_BatteryCharge: "REAL",
                                           self.VAR.Year_E_BatteryDischarge: "REAL",
                                           self.VAR.Year_E_Battery2Load: "REAL",

                                           self.VAR.Year_E_Load: "REAL",
                                           self.VAR.Year_E_PVSelfUse: "REAL",
                                           self.VAR.Year_PVSelfConsumptionRate: "REAL",
                                           self.VAR.Year_PVSelfSufficiencyRate: "REAL",

                                           self.VAR.Building_hwbnorm: "REAL",
                                           # self.VAR.Household_DishWasherShifting: "REAL",
                                           # self.VAR.Household_WashingMachineShifting: "REAL",
                                           # self.VAR.Household_DryerShifting: "REAL",
                                           self.VAR.Household_TankSize: "REAL",
                                           self.VAR.Household_CoolingAdoption: "REAL",
                                           self.VAR.Household_PVPower: "REAL",
                                           self.VAR.Household_BatteryCapacity: "REAL",
                                           self.VAR.ID_AgeGroup: "REAL",
                                           self.VAR.ID_SpaceHeatingSystem: "TEXT",
                                           self.VAR.Environment_ElectricityPriceType: "REAL"
                                           }
        self.SystemOperationYear_ValueList = []
        self.SystemOperationHour_ValueList = []

    def extract_Result2Array(self, result_DictValues):
        result_array = np.nan_to_num(np.array(list(result_DictValues.values()), dtype=np.float), nan=0)
        return result_array

    def collect_OptimizationResult(self, household, environment, instance) -> (np.array, np.array):
        ElectricityPrice_array = self.extract_Result2Array(
            instance.ElectricityPrice.extract_values()) * 1_000  # Cent/kWh
        FeedinTariff_array = self.extract_Result2Array(instance.FiT.extract_values()) * 1_000 # Cent/kWh
        OutsideTemperature_array = self.extract_Result2Array(instance.T_outside.extract_values())

        E_BaseLoad_array = self.extract_Result2Array(instance.BaseLoadProfile.extract_values()) / 1000  # kW
        # Hour_DishWasher1_array = self.extract_Result2Array(instance.DishWasher1.extract_values())
        # Hour_DishWasher2_array = self.extract_Result2Array(instance.DishWasher2.extract_values())
        # Hour_DishWasher3_array = self.extract_Result2Array(instance.DishWasher3.extract_values())
        # E_DishWasher_array = (Hour_DishWasher1_array + Hour_DishWasher2_array + Hour_DishWasher3_array) * household[
        #     "ApplianceGroup_DishWasherPower"]
        # Hour_WashingMachine1_array = self.extract_Result2Array(instance.WashingMachine1.extract_values())
        # Hour_WashingMachine2_array = self.extract_Result2Array(instance.WashingMachine2.extract_values())
        # Hour_WashingMachine3_array = self.extract_Result2Array(instance.WashingMachine3.extract_values())
        # E_WashingMachine_array = (
        #                                      Hour_WashingMachine1_array + Hour_WashingMachine2_array + Hour_WashingMachine3_array) * \
        #                          household["ApplianceGroup_WashingMachinePower"]
        # Hour_Dryer1_array = self.extract_Result2Array(instance.Dryer1.extract_values())
        # Hour_Dryer2_array = self.extract_Result2Array(instance.Dryer2.extract_values())
        # E_Dryer_array = (Hour_Dryer1_array + Hour_Dryer2_array) * household["ApplianceGroup_DryerPower"]
        # E_SmartAppliances_array = E_DishWasher_array + E_WashingMachine_array + E_Dryer_array

        Q_TankHeatingHeatPump_array = self.extract_Result2Array(instance.Q_TankHeating.extract_values()) / 1000  # kWh
        SpaceHeatingHourlyCOP_array = self.extract_Result2Array(instance.SpaceHeatingHourlyCOP.extract_values())
        E_TankHeatingHeatPump_array = Q_TankHeatingHeatPump_array / SpaceHeatingHourlyCOP_array
        Q_TankHeatingHeatingElement_array = self.extract_Result2Array(
            instance.Q_HeatingElement.extract_values()) / 1000  # kWh
        Q_RoomHeating_array = self.extract_Result2Array(instance.Q_RoomHeating.extract_values()) / 1000  # kWh

        RoomTemperature_array = self.extract_Result2Array(instance.T_room.extract_values())
        BuildingMassTemperature_array = self.extract_Result2Array(instance.Tm_t.extract_values())
        SolarGain_array = self.extract_Result2Array(instance.Q_Solar.extract_values()) / 1000  # kW

        Q_RoomCooling_array = self.extract_Result2Array(instance.Q_RoomCooling.extract_values()) / 1000  # kWh
        CoolingHourlyCOP = self.extract_Result2Array(instance.CoolingCOP.extract_values())
        E_RoomCooling_array = Q_RoomCooling_array / CoolingHourlyCOP

        Q_HotWater_array = self.extract_Result2Array(instance.HotWater.extract_values()) / 1000
        HotWaterHourlyCOP_array = self.extract_Result2Array(instance.HotWaterHourlyCOP.extract_values())
        E_HotWater_array = Q_HotWater_array / HotWaterHourlyCOP_array

        E_Grid_array = self.extract_Result2Array(instance.Grid.extract_values()) / 1000
        E_Grid2Load_array = self.extract_Result2Array(instance.Grid2Load.extract_values()) / 1000
        E_Grid2Bat_array = self.extract_Result2Array(instance.Grid2Bat.extract_values()) / 1000

        E_PV_array = self.extract_Result2Array(instance.PhotovoltaicProfile.extract_values()) / 1000
        E_PV2Load_array = self.extract_Result2Array(instance.PV2Load.extract_values()) / 1000
        E_PV2Bat_array = self.extract_Result2Array(instance.PV2Bat.extract_values()) / 1000
        E_PV2Grid_array = self.extract_Result2Array(instance.PV2Grid.extract_values()) / 1000

        E_BatCharge_array = self.extract_Result2Array(instance.BatCharge.extract_values()) / 1000
        E_BatDischarge_array = self.extract_Result2Array(instance.BatDischarge.extract_values()) / 1000
        E_Bat2Load_array = self.extract_Result2Array(instance.Bat2Load.extract_values()) / 1000
        E_BatSoC_array = self.extract_Result2Array(instance.BatSoC.extract_values()) / 1000

        E_Load_array = self.extract_Result2Array(instance.Load.extract_values()) / 1000
        print(f"{E_Grid_array.sum()*0.2 - E_PV2Grid_array.sum()*0.0767}")

        # (column stack is 6 times faster than for loop)
        hourly_value_list = np.column_stack([
            np.full((len(E_Load_array),), household["ID"]),
            np.full((len(E_Load_array),), household["ID_Building"]),
            np.full((len(E_Load_array),), environment["ID"]),
            np.arange(1, self.OptimizationHourHorizon + 1),
            ElectricityPrice_array,
            FeedinTariff_array,
            np.full((len(E_Load_array),), household["ID_AgeGroup"]),
            np.full((len(E_Load_array),), household["Name_SpaceHeatingPumpType"]),
            OutsideTemperature_array,

            E_BaseLoad_array,
            # E_DishWasher_array,
            # E_WashingMachine_array,
            # E_Dryer_array,
            # E_SmartAppliances_array,

            Q_TankHeatingHeatPump_array,
            SpaceHeatingHourlyCOP_array,
            E_TankHeatingHeatPump_array,
            Q_TankHeatingHeatPump_array - E_TankHeatingHeatPump_array,
            Q_TankHeatingHeatingElement_array,
            Q_RoomHeating_array,

            RoomTemperature_array,
            BuildingMassTemperature_array,
            SolarGain_array,

            Q_RoomCooling_array,
            E_RoomCooling_array,

            Q_HotWater_array,
            E_HotWater_array,

            E_Grid_array,
            E_Grid2Load_array,
            E_Grid2Bat_array,

            E_PV_array,
            E_PV2Load_array,
            E_PV2Bat_array,
            E_PV2Grid_array,

            E_BatCharge_array,
            E_BatDischarge_array,
            E_Bat2Load_array,
            E_BatSoC_array,

            E_Load_array])

        yearly_value_list=[household["ID"],
                           household["ID_Building"],
                           environment["ID"],
                           instance.Objective(),

                           E_BaseLoad_array.sum(),
                           # E_SmartAppliances_array.sum(),

                           Q_TankHeatingHeatPump_array.sum(),
                           E_TankHeatingHeatPump_array.sum(),
                           Q_TankHeatingHeatPump_array.sum() / E_TankHeatingHeatPump_array.sum(),
                           (Q_TankHeatingHeatPump_array - E_TankHeatingHeatPump_array).sum(),
                           Q_TankHeatingHeatingElement_array.sum(),
                           Q_RoomHeating_array.sum(),

                           Q_RoomCooling_array.sum(),
                           E_RoomCooling_array.sum(),

                           Q_HotWater_array.sum(),
                           E_HotWater_array.sum(),

                           E_Grid_array.sum(),
                           E_Grid2Load_array.sum(),
                           E_Grid2Bat_array.sum(),

                           E_PV_array.sum(),
                           E_PV2Load_array.sum(),
                           E_PV2Bat_array.sum(),
                           E_PV2Grid_array.sum(),

                           E_BatCharge_array.sum(),
                           E_BatDischarge_array.sum(),
                           E_Bat2Load_array.sum(),

                           E_Load_array.sum(),
                           E_PV_array.sum() - E_PV2Grid_array.sum(),
                           (E_PV_array.sum() - E_PV2Grid_array.sum()) / E_PV_array.sum(),
                           (E_PV_array.sum() - E_PV2Grid_array.sum()) / E_Load_array.sum(),

                           household["Building_hwb_norm1"],
                           # household["ApplianceGroup_DishWasherShifting"],
                           # household["ApplianceGroup_WashingMachineShifting"],
                           # household["ApplianceGroup_DryerShifting"],
                           household["SpaceHeating_TankSize"],
                           household["SpaceCooling_AdoptionStatus"],
                           household["PV_PVPower"],
                           household["Battery_Capacity"],
                           household["ID_AgeGroup"],
                           household["Name_SpaceHeatingPumpType"],
                           environment["ID_ElectricityPriceType"]]
        return yearly_value_list, hourly_value_list

        # old code (stack it up in RAM)
        # self.SystemOperationHour_ValueList.append(hourly_value_list)
        # self.SystemOperationYear_ValueList.append(yearly_value_list)

    def save_optimization_results_in_the_loop(self, household, environment, instance):
        # new approach (save it directly to SQLite) may be impossible with multiprocessing
        yearly_value_list, hourly_value_list = self.collect_OptimizationResult(household, environment, instance)
        # Hourly:
        DB().add_DataFrame(table=hourly_value_list,
                           table_name="Res_SystemOperationHour_new",
                           column_names=self.SystemOperationHour_Column.keys(),
                           conn=self.Conn,
                           dtype=self.SystemOperationHour_Column)
        # Year:
        DB().add_DataFrame(table=np.array(yearly_value_list).reshape(1, -1),
                           table_name="Res_SystemOperationYear_new",
                           column_names=self.SystemOperationYear_Column.keys(),
                           conn=self.Conn,
                           dtype=self.SystemOperationYear_Column)

    def save_OptimizationResult_outside_loop(self):
        DB().write_DataFrame(np.vstack(self.SystemOperationHour_ValueList), REG_Table().Res_SystemOperationHour,
                             self.SystemOperationHour_Column.keys(), self.Conn, dtype=self.SystemOperationHour_Column)
        DB().write_DataFrame(self.SystemOperationYear_ValueList, REG_Table().Res_SystemOperationYear,
                             self.SystemOperationYear_Column.keys(), self.Conn, dtype=self.SystemOperationYear_Column)



