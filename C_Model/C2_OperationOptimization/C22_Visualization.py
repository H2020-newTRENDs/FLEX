# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import numpy as np
import matplotlib.pyplot as plt

from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_REG import REG_Table
from A_Infrastructure.A2_REG import REG_Var
from A_Infrastructure.A2_REG import REG_Color
from A_Infrastructure.A3_DB import DB



class Visualization:

    def __init__(self):

        self.VAR = REG_Var()
        self.COLOR = REG_Color()
        self.VarColors = {self.VAR.E_BaseElectricityLoad: self.COLOR.green,
                          self.VAR.E_DishWasher: self.COLOR.green,
                          self.VAR.E_WashingMachine: self.COLOR.green,
                          self.VAR.E_Dryer: self.COLOR.green,
                          self.VAR.E_SmartAppliance: self.COLOR.green,

                          self.VAR.Q_HeatPump: self.COLOR.green,
                          self.VAR.HeatPumpPerformanceFactor: self.COLOR.green,
                          self.VAR.E_HeatPump: self.COLOR.green,
                          self.VAR.E_AmbientHeat: self.COLOR.green,
                          self.VAR.E_HeatingElement: self.COLOR.green,
                          self.VAR.E_RoomHeating: self.COLOR.green,

                          self.VAR.Q_RoomCooling: self.COLOR.green,
                          self.VAR.E_RoomCooling: self.COLOR.green,

                          self.VAR.Q_HotWater: self.COLOR.green,
                          self.VAR.E_HotWater: self.COLOR.green,

                          self.VAR.E_Grid: self.COLOR.green,
                          self.VAR.E_Grid2Load: self.COLOR.green,
                          self.VAR.E_Grid2Battery: self.COLOR.green,
                          self.VAR.E_Grid2EV: self.COLOR.green,

                          self.VAR.E_PV: self.COLOR.green,
                          self.VAR.E_PV2Load: self.COLOR.green,
                          self.VAR.E_PV2Battery: self.COLOR.green,
                          self.VAR.E_PV2EV: self.COLOR.green,
                          self.VAR.E_PV2Grid: self.COLOR.green,

                          self.VAR.E_BatteryCharge: self.COLOR.green,
                          self.VAR.E_BatteryDischarge: self.COLOR.green,
                          self.VAR.E_Battery2Load: self.COLOR.green,
                          self.VAR.E_Battery2EV: self.COLOR.green,
                          self.VAR.BatteryStateOfCharge: self.COLOR.green,

                          self.VAR.E_EVCharge: self.COLOR.green,
                          self.VAR.E_EV2Discharge: self.COLOR.green,
                          self.VAR.E_EV2Load: self.COLOR.green,
                          self.VAR.E_EV2Battery: self.COLOR.green,
                          self.VAR.EVStateOfCharge: self.COLOR.green,

                          self.VAR.TotalElectricityDemand: self.COLOR.green,
                          }
        self.PlotHorizon = {}

    def plot_Func(self): # a super flexible plotting function
        pass

    def plot_SystemLoad(self, **kargs):

        if "horizon" in kargs:
            pass

        pass
