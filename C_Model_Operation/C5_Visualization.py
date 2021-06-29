# -*- coding: utf-8 -*-
__author__ = 'Songmin'

from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from C_Model_Operation.C1_REG import REG_Var


class Visualization:

    def __init__(self, conn):

        self.Conn = conn
        self.VAR = REG_Var()
        self.VarColors = {self.VAR.E_BaseElectricityLoad: CONS().green,
                          self.VAR.E_DishWasher: CONS().green,
                          self.VAR.E_WashingMachine: CONS().green,
                          self.VAR.E_Dryer: CONS().green,
                          self.VAR.E_SmartAppliance: CONS().green,

                          self.VAR.Q_HeatPump: CONS().green,
                          self.VAR.HeatPumpPerformanceFactor: CONS().green,
                          self.VAR.E_HeatPump: CONS().green,
                          self.VAR.E_AmbientHeat: CONS().green,
                          self.VAR.Q_HeatingElement: CONS().green,
                          self.VAR.E_RoomHeating: CONS().green,

                          self.VAR.Q_RoomCooling: CONS().green,
                          self.VAR.E_RoomCooling: CONS().green,

                          self.VAR.Q_HotWater: CONS().green,
                          self.VAR.E_HotWater: CONS().green,

                          self.VAR.E_Grid: CONS().green,
                          self.VAR.E_Grid2Load: CONS().green,
                          self.VAR.E_Grid2Battery: CONS().green,
                          self.VAR.E_Grid2EV: CONS().green,

                          self.VAR.E_PV: CONS().green,
                          self.VAR.E_PV2Load: CONS().green,
                          self.VAR.E_PV2Battery: CONS().green,
                          self.VAR.E_PV2EV: CONS().green,
                          self.VAR.E_PV2Grid: CONS().green,

                          self.VAR.E_BatteryCharge: CONS().green,
                          self.VAR.E_BatteryDischarge: CONS().green,
                          self.VAR.E_Battery2Load: CONS().green,
                          self.VAR.E_Battery2EV: CONS().green,
                          self.VAR.BatteryStateOfCharge: CONS().green,

                          self.VAR.E_EVCharge: CONS().green,
                          self.VAR.E_EV2Discharge: CONS().green,
                          self.VAR.E_EV2Load: CONS().green,
                          self.VAR.E_EV2Battery: CONS().green,
                          self.VAR.EVStateOfCharge: CONS().green,

                          self.VAR.TotalElectricityDemand: CONS().green,
                          }
        self.PlotHorizon = {}
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.SystemOperationHour = DB().read_DataFrame(REG_Table().Res_SystemOperationHour, self.Conn)

    def plot_Func(self): # a super flexible plotting function
        pass

    def plot_SystemLoad(self, id_household, id_environment, **kargs):

        if "horizon" in kargs:
            HourStart = kargs["horizon"][0]
            HourEnd = kargs["horizon"][1]
            pass
        else:
            HourStart = 1
            HourEnd = 8760

        SystemOperation = self.SystemOperationHour.loc[(self.SystemOperationHour["ID_Household"] == id_household) &
                                                       (self.SystemOperationHour["ID_Environment"] == id_environment) &
                                                       (self.SystemOperationHour["ID_Hour"] >= HourStart) &
                                                       (self.SystemOperationHour["ID_Hour"] <= HourEnd)]
        BaseElectricityLoad = SystemOperation["E_BaseElectricityLoad"]
        SmartAppliance = SystemOperation["E_SmartAppliance"]
































