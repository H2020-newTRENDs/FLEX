# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import numpy as np
from A_Infrastructure.A1_CONS import CONS
from C_Model_Operation.C1_REG import REG_Table
from C_Model_Operation.C1_REG import REG_Var
from A_Infrastructure.A2_DB import DB

class DataCollector:

    def __init__(self, conn):
        self.Conn = conn
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.OptimizationHourHorizon = len(self.TimeStructure)
        self.VAR = REG_Var()
        self.SystemOperationHour_Column = {self.VAR.ID_Household: "INTEGER",
                                           self.VAR.ID_Environment: "INTEGER",
                                           self.VAR.ID_Hour: "INTEGER",
                                           self.VAR.ElectricityPrice: "REAL",
                                           self.VAR.FeedinTariff: "REAL",
                                           self.VAR.OutsideTemperature: "REAL",

                                           self.VAR.E_BaseElectricityLoad: "REAL",
                                           self.VAR.E_DishWasher: "REAL",
                                           self.VAR.E_WashingMachine: "REAL",
                                           self.VAR.E_Dryer: "REAL",
                                           self.VAR.E_SmartAppliance: "REAL",

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
                                           self.VAR.E_EVDischarge: "REAL",
                                           self.VAR.E_EV2Load: "REAL",
                                           self.VAR.E_EV2Battery: "REAL",
                                           self.VAR.EVStateOfCharge: "REAL",

                                           self.VAR.E_Load: "REAL",
                                           }
        self.SystemOperationHour_ValueList = []
        self.SystemOperationYear_Column = {self.VAR.ID_Household: "INTEGER",
                                           self.VAR.ID_Environment: "INTEGER",
                                           self.VAR.OperationCost: "REAL",

                                           self.VAR.Year_E_BaseElectricityLoad: "REAL",
                                           self.VAR.Year_E_SmartAppliance: "REAL",
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
                                           self.VAR.Year_E_Grid2EV: "REAL",

                                           self.VAR.Year_E_PV: "REAL",
                                           self.VAR.Year_E_PV2Load: "REAL",
                                           self.VAR.Year_E_PV2Battery: "REAL",
                                           self.VAR.Year_E_PV2EV: "REAL",
                                           self.VAR.Year_E_PV2Grid: "REAL",

                                           self.VAR.Year_E_BatteryCharge: "REAL",
                                           self.VAR.Year_E_BatteryDischarge: "REAL",
                                           self.VAR.Year_E_Battery2Load: "REAL",
                                           self.VAR.Year_E_Battery2EV: "REAL",

                                           self.VAR.Year_E_EVCharge: "REAL",
                                           self.VAR.Year_E_EVDischarge: "REAL",
                                           self.VAR.Year_E_EV2Load: "REAL",
                                           self.VAR.Year_E_EV2Battery: "REAL",

                                           self.VAR.Year_E_Load: "REAL",
                                           self.VAR.Year_E_ElectricityDemand: "REAL",
                                           self.VAR.Year_E_PVSelfUse: "REAL",
                                           self.VAR.Year_PVSelfConsumptionRate: "REAL",
                                           self.VAR.Year_PVSelfSufficiencyRate: "REAL",

                                           self.VAR.Building_hwbnorm: "REAL",
                                           self.VAR.Household_DishWasherShifting: "REAL",
                                           self.VAR.Household_WashingMachineShifting: "REAL",
                                           self.VAR.Household_DryerShifting: "REAL",
                                           self.VAR.Household_TankSize: "REAL",
                                           self.VAR.Household_CoolingAdoption: "REAL",
                                           self.VAR.Household_PVPower: "REAL",
                                           self.VAR.Household_BatteryCapacity: "REAL",
                                           self.VAR.Household_EVBatterySize: "REAL",
                                           self.VAR.Environment_ElectricityPriceType: "REAL"
                                           }
        self.SystemOperationYear_ValueList = []

    def extract_Result2Array(self, result_DictValues):
        result_array = np.nan_to_num(np.array(list(result_DictValues.values()), dtype=np.float), nan=0)
        return result_array

    def collect_OptimizationResult(self, Household, Environment, PyomoModelInstance):

        ElectricityPrice_array = self.extract_Result2Array(PyomoModelInstance.ElectricityPrice.extract_values())
        FeedinTariff_array = self.extract_Result2Array(PyomoModelInstance.FiT.extract_values())
        OutsideTemperature_array = self.extract_Result2Array(PyomoModelInstance.T_outside.extract_values())

        E_BaseLoad_array = self.extract_Result2Array(PyomoModelInstance.BaseLoadProfile.extract_values())
        Hour_DishWasher1_array = self.extract_Result2Array(PyomoModelInstance.DishWasher1.extract_values())
        Hour_DishWasher2_array = self.extract_Result2Array(PyomoModelInstance.DishWasher2.extract_values())
        Hour_DishWasher3_array = self.extract_Result2Array(PyomoModelInstance.DishWasher3.extract_values())
        E_DishWasher_array = (Hour_DishWasher1_array + Hour_DishWasher2_array + Hour_DishWasher3_array) * Household.ApplianceGroup.DishWasherPower
        Hour_WashingMachine1_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine1.extract_values())
        Hour_WashingMachine2_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine2.extract_values())
        Hour_WashingMachine3_array = self.extract_Result2Array(PyomoModelInstance.WashingMachine3.extract_values())
        E_WashingMachine_array = (Hour_WashingMachine1_array + Hour_WashingMachine2_array + Hour_WashingMachine3_array) * Household.ApplianceGroup.WashingMachinePower
        Hour_Dryer1_array = self.extract_Result2Array(PyomoModelInstance.Dryer1.extract_values())
        Hour_Dryer2_array = self.extract_Result2Array(PyomoModelInstance.Dryer2.extract_values())
        E_Dryer_array = (Hour_Dryer1_array + Hour_Dryer2_array) * Household.ApplianceGroup.DryerPower
        E_SmartAppliances_array = E_DishWasher_array + E_WashingMachine_array + E_Dryer_array

        Q_TankHeatingHeatPump_array = self.extract_Result2Array(PyomoModelInstance.Q_HeatingTank_in.extract_values()) / 1000 #kWh
        SpaceHeatingHourlyCOP_array = self.extract_Result2Array(PyomoModelInstance.SpaceHeatingHourlyCOP.extract_values())
        E_TankHeatingHeatPump_array = Q_TankHeatingHeatPump_array / SpaceHeatingHourlyCOP_array
        Q_TankHeatingHeatingElement_array = self.extract_Result2Array(PyomoModelInstance.Q_HeatingElement.extract_values()) / 1000  # kWh
        Q_RoomHeating_array = self.extract_Result2Array(PyomoModelInstance.Q_HeatingTank_out.extract_values()) / 1000 #kWh

        RoomTemperature_array = self.extract_Result2Array(PyomoModelInstance.T_room.extract_values())
        BuildingMassTemperature_array = self.extract_Result2Array(PyomoModelInstance.Tm_t.extract_values())
        SolarGain_array = self.extract_Result2Array(PyomoModelInstance.Q_Solar.extract_values())/1000

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

        for t in range(0, self.OptimizationHourHorizon):
            self.SystemOperationHour_ValueList.append([Household.ID,
                                                       Environment["ID"],
                                                       t + 1,
                                                       ElectricityPrice_array[t],
                                                       FeedinTariff_array[t],
                                                       OutsideTemperature_array[t],

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

                                                       RoomTemperature_array[t],
                                                       BuildingMassTemperature_array[t],
                                                       SolarGain_array[t],

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

        self.SystemOperationYear_ValueList.append([Household.ID,
                                                   Environment["ID"],
                                                   PyomoModelInstance.Objective(),

                                                   E_BaseLoad_array.sum(),
                                                   E_SmartAppliances_array.sum(),

                                                   Q_TankHeatingHeatPump_array.sum(),
                                                   E_TankHeatingHeatPump_array.sum(),
                                                   Q_TankHeatingHeatPump_array.sum()/E_TankHeatingHeatPump_array.sum(),
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
                                                   E_Grid2EV_array.sum(),

                                                   E_PV_array.sum(),
                                                   E_PV2Load_array.sum(),
                                                   E_PV2Bat_array.sum(),
                                                   E_PV2EV_array.sum(),
                                                   E_PV2Grid_array.sum(),

                                                   E_BatCharge_array.sum(),
                                                   E_BatDischarge_array.sum(),
                                                   E_Bat2Load_array.sum(),
                                                   E_Bat2EV_array.sum(),

                                                   E_EVCharge_array.sum(),
                                                   E_EVDischarge_array.sum(),
                                                   E_EV2Load_array.sum(),
                                                   E_EV2Bat_array.sum(),

                                                   E_Load_array.sum(),
                                                   (E_Load_array + E_EVCharge_array - E_EVDischarge_array).sum(),
                                                   E_PV_array.sum() - E_PV2Grid_array.sum(),
                                                   (E_PV_array.sum() - E_PV2Grid_array.sum())/E_PV_array.sum(),
                                                   (E_PV_array.sum() - E_PV2Grid_array.sum())/(E_Load_array + E_EVCharge_array - E_EVDischarge_array).sum(),

                                                   Household.Building.hwb_norm1,
                                                   Household.ApplianceGroup.DishWasherShifting,
                                                   Household.ApplianceGroup.WashingMachineShifting,
                                                   Household.ApplianceGroup.DryerShifting,
                                                   Household.SpaceHeatingSystem.TankSize,
                                                   Household.SpaceCooling.AdoptionStatus,
                                                   Household.PV.PVPower,
                                                   Household.Battery.Capacity,
                                                   Household.ElectricVehicle.BatterySize,
                                                   Environment["ID_ElectricityPriceType"]])

    def save_OptimizationResult(self):
        DB().write_DataFrame(self.SystemOperationHour_ValueList, REG_Table().Res_SystemOperationHour + "_S1",
                             self.SystemOperationHour_Column.keys(), self.Conn, dtype=self.SystemOperationHour_Column)
        DB().write_DataFrame(self.SystemOperationYear_ValueList, REG_Table().Res_SystemOperationYear + "_S1",
                             self.SystemOperationYear_Column.keys(), self.Conn, dtype=self.SystemOperationYear_Column)


