# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import matplotlib.pyplot as plt

from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from C_Model_Operation.C1_REG import REG_Var


class Visualization:

    def __init__(self, conn):

        self.Conn = conn
        self.VAR = REG_Var()
        self.VarColors = {self.VAR.E_BaseElectricityLoad: CONS().light_brown,
                          self.VAR.E_DishWasher: CONS().green,
                          self.VAR.E_WashingMachine: CONS().green,
                          self.VAR.E_Dryer: CONS().green,
                          self.VAR.E_SmartAppliance: CONS().green,

                          self.VAR.Q_HeatPump: CONS().green,
                          self.VAR.HeatPumpPerformanceFactor: CONS().green,
                          self.VAR.E_HeatPump: CONS().red,
                          self.VAR.E_AmbientHeat: CONS().green,
                          self.VAR.Q_HeatingElement: CONS().red,
                          self.VAR.E_RoomHeating: CONS().green,

                          self.VAR.Q_RoomCooling: CONS().green,
                          self.VAR.E_RoomCooling: CONS().blue,

                          self.VAR.Q_HotWater: CONS().green,
                          self.VAR.E_HotWater: CONS().purple,

                          self.VAR.E_Grid: CONS().black,
                          self.VAR.E_Grid2Load: CONS().black,
                          self.VAR.E_Grid2Battery: CONS().green,
                          self.VAR.E_Grid2EV: CONS().green,

                          self.VAR.E_PV: CONS().green,
                          self.VAR.E_PV2Load: CONS().yellow,
                          self.VAR.E_PV2Battery: CONS().green,
                          self.VAR.E_PV2EV: CONS().green,
                          self.VAR.E_PV2Grid: CONS().green,

                          self.VAR.E_BatteryCharge: CONS().green,
                          self.VAR.E_BatteryDischarge: CONS().green,
                          self.VAR.E_Battery2Load: CONS().dark_red,
                          self.VAR.E_Battery2EV: CONS().green,
                          self.VAR.BatteryStateOfCharge: CONS().green,

                          self.VAR.E_EVCharge: CONS().green,
                          self.VAR.E_EVDischarge: CONS().green,
                          self.VAR.E_EV2Load: CONS().turquoise,
                          self.VAR.E_EV2Battery: CONS().green,
                          self.VAR.EVStateOfCharge: CONS().green,

                          self.VAR.TotalElectricityDemand: CONS().green,
                          }
        self.PlotHorizon = {}
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.SystemOperationHour = DB().read_DataFrame(REG_Table().Res_SystemOperationHour, self.Conn)

    def plot_SystemLoad(self, id_household, id_environment, horizon,
                        base_load,
                        smart_appliance,
                        heat_pump,
                        heat_element,
                        room_cooling,
                        hot_water,
                        grid_2_load,
                        pv_2_load,
                        battery_2_load,
                        ev_2_load,
                        **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.15, 0.68, 0.8])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.bar(x_values,
                 base_load["values"],
                 label=base_load["label"],
                 alpha=alpha_value,
                 color=base_load["color"])

        ax_1.bar(x_values,
                 smart_appliance["values"],
                 label=smart_appliance["label"],
                 bottom=base_load["values"],
                 alpha=alpha_value,
                 color=smart_appliance["color"])

        ax_1.bar(x_values,
                 heat_pump["values"],
                 label=heat_pump["label"],
                 bottom=base_load["values"] + smart_appliance["values"],
                 alpha=alpha_value,
                 color=heat_pump["color"])

        ax_1.bar(x_values,
                 heat_element["values"],
                 label=heat_element["label"],
                 bottom=base_load["values"] + smart_appliance["values"] + heat_pump["values"],
                 alpha=alpha_value,
                 color=heat_element["color"])

        ax_1.bar(x_values,
                 room_cooling["values"],
                 label=room_cooling["label"],
                 bottom=base_load["values"] + smart_appliance["values"] + heat_pump["values"] + heat_element["values"],
                 alpha=alpha_value,
                 color=room_cooling["color"])

        ax_1.bar(x_values,
                 hot_water["values"],
                 label=hot_water["label"],
                 bottom=base_load["values"] + smart_appliance["values"] + heat_pump["values"] + heat_element["values"] + room_cooling["values"],
                 alpha=alpha_value,
                 color=hot_water["color"])

        ax_1.plot(x_values,
                  grid_2_load["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=grid_2_load["label"],
                  color=grid_2_load["color"])

        ax_1.plot(x_values,
                  pv_2_load["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=pv_2_load["label"],
                  color=pv_2_load["color"])

        ax_1.plot(x_values,
                  battery_2_load["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=battery_2_load["label"],
                  color=battery_2_load["color"])

        ax_1.plot(x_values,
                  ev_2_load["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=ev_2_load["label"],
                  color=ev_2_load["color"])

        ax_1.set_xlabel("Hour of the Year", fontsize=25, labelpad=10)
        ax_1.set_ylabel("Electricity load (kW)", fontsize=25, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(kwargs["y_lim"])
        figure.legend(fontsize=15, bbox_to_anchor=(1.01, 0, 0.2, 1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=1, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)

        fig_name = "SystemLoad_H" + str(id_household) + "_E" + str(id_environment) + "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def visualization_SystemLoad(self, id_household, id_environment, **kargs):

        if "horizon" in kargs:
            HourStart = kargs["horizon"][0]
            HourEnd = kargs["horizon"][1]
        else:
            HourStart = 1
            HourEnd = 8760

        SystemOperation = self.SystemOperationHour.loc[(self.SystemOperationHour[self.VAR.ID_Household] == id_household) &
                                                       (self.SystemOperationHour[self.VAR.ID_Environment] == id_environment) &
                                                       (self.SystemOperationHour[self.VAR.ID_Hour] >= HourStart) &
                                                       (self.SystemOperationHour[self.VAR.ID_Hour] <= HourEnd)]

        ElectricityPrice = SystemOperation[self.VAR.ElectricityPrice]
        FeedinTariff = SystemOperation[self.VAR.FeedinTariff]

        # ----------------
        # Data preparation
        # ----------------

        # 1. Electricity balance
        # Grid
        Grid = SystemOperation[self.VAR.E_Grid]
        Grid2Load = SystemOperation[self.VAR.E_Grid2Load]
        Grid2Battery = SystemOperation[self.VAR.E_Grid2Battery]
        Grid2EV = SystemOperation[self.VAR.E_Grid2EV]

        # Load
        BaseElectricityLoad = SystemOperation[self.VAR.E_BaseElectricityLoad]
        SmartAppliance = SystemOperation[self.VAR.E_SmartAppliance]
        HeatPump = SystemOperation[self.VAR.E_HeatPump]
        HeatElement = SystemOperation[self.VAR.Q_HeatingElement]
        Cooling = SystemOperation[self.VAR.E_RoomCooling]
        HotWater = SystemOperation[self.VAR.E_HotWater]

        # PV
        PV = SystemOperation[self.VAR.E_PV]
        PV2Load = SystemOperation[self.VAR.E_PV2Load]
        PV2Battery = SystemOperation[self.VAR.E_PV2Battery]
        PV2EV = SystemOperation[self.VAR.E_PV2EV]
        PV2Grid = SystemOperation[self.VAR.E_PV2Grid]

        # Battery
        BatteryCharge = SystemOperation[self.VAR.E_BatteryCharge]
        BatteryDischarge = SystemOperation[self.VAR.E_BatteryDischarge]
        Battery2Load = SystemOperation[self.VAR.E_Battery2Load]
        Battery2EV = SystemOperation[self.VAR.E_Battery2EV]
        BatteryStateOfCharge = SystemOperation[self.VAR.BatteryStateOfCharge]

        # EV
        EVCharge = SystemOperation[self.VAR.E_EVCharge]
        EVDischarge = SystemOperation[self.VAR.E_EVDischarge]
        EV2Load = SystemOperation[self.VAR.E_EV2Load]
        EV2Battery = SystemOperation[self.VAR.E_EV2Battery]
        EVStateOfCharge = SystemOperation[self.VAR.EVStateOfCharge]


        # 2. Space heating and cooling

        # ----
        # Plot
        # ----

        # Load
        Horizon = [HourStart, HourEnd]
        BaseElectricityLoad_element = {"values": BaseElectricityLoad,
                                       "label": "BaseLoad",
                                       "color": self.VarColors[self.VAR.E_BaseElectricityLoad]}
        SmartAppliance_element = {"values": SmartAppliance,
                                  "label": "SmartAppliance",
                                  "color": self.VarColors[self.VAR.E_SmartAppliance]}
        HeatPump_element = {"values": HeatPump,
                            "label": "HeatPump",
                            "color": self.VarColors[self.VAR.E_HeatPump]}
        HeatElement_element = {"values": HeatElement,
                               "label": "HeatElement",
                               "color": self.VarColors[self.VAR.Q_HeatingElement]}
        RoomCooling_element = {"values": Cooling,
                               "label": "RoomCooling",
                               "color": self.VarColors[self.VAR.E_RoomCooling]}
        HotWater_element = {"values": HotWater,
                            "label": "HotWater",
                            "color": self.VarColors[self.VAR.E_HotWater]}

        Grid2Load_element = {"values": Grid2Load,
                             "label": "Grid2Load",
                             "color": self.VarColors[self.VAR.E_Grid2Load]}
        PV2Load_element = {"values": PV2Load,
                           "label": "PV2Load",
                           "color": self.VarColors[self.VAR.E_PV2Load]}
        Battery2Load_element = {"values": Battery2Load,
                                "label": "Battery2Load",
                                "color": self.VarColors[self.VAR.E_Battery2Load]}
        EV2Load_element = {"values": EV2Load,
                           "label": "EV2Load",
                           "color": self.VarColors[self.VAR.E_EV2Load]}

        self.plot_SystemLoad(id_household, id_environment, Horizon,
                             BaseElectricityLoad_element,
                             SmartAppliance_element,
                             HeatPump_element,
                             HeatElement_element,
                             RoomCooling_element,
                             HotWater_element,
                             Grid2Load_element,
                             PV2Load_element,
                             Battery2Load_element,
                             EV2Load_element)

    def plot_SpaceHeatingAndCooling(self):
        pass

    def run(self):
        self.visualization_SystemLoad(1, 1, horizon=[5500, 5600])
        self.visualization_SystemLoad(1, 1, horizon=[100, 200])



























