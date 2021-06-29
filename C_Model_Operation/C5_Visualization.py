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
        self.COLOR = CONS()
        self.VarColors = {self.VAR.ElectricityPrice: self.COLOR.red_pink,
                          self.VAR.FeedinTariff: self.COLOR.green,

                          self.VAR.E_BaseElectricityLoad: self.COLOR.light_brown,
                          self.VAR.E_DishWasher: self.COLOR.green,
                          self.VAR.E_WashingMachine: self.COLOR.green,
                          self.VAR.E_Dryer: self.COLOR.green,
                          self.VAR.E_SmartAppliance: self.COLOR.green,

                          self.VAR.Q_HeatPump: self.COLOR.green,
                          self.VAR.HeatPumpPerformanceFactor: self.COLOR.green,
                          self.VAR.E_HeatPump: self.COLOR.red,
                          self.VAR.E_AmbientHeat: self.COLOR.green,
                          self.VAR.Q_HeatingElement: self.COLOR.red,
                          self.VAR.E_RoomHeating: self.COLOR.green,

                          self.VAR.Q_RoomCooling: self.COLOR.green,
                          self.VAR.E_RoomCooling: self.COLOR.blue,

                          self.VAR.Q_HotWater: self.COLOR.green,
                          self.VAR.E_HotWater: self.COLOR.purple,

                          self.VAR.E_Grid: self.COLOR.black,
                          self.VAR.E_Grid2Load: self.COLOR.black,
                          self.VAR.E_Grid2Battery: self.COLOR.red,
                          self.VAR.E_Grid2EV: self.COLOR.red,

                          self.VAR.E_PV: self.COLOR.green,
                          self.VAR.E_PV2Load: self.COLOR.brown,
                          self.VAR.E_PV2Battery: self.COLOR.blue,
                          self.VAR.E_PV2EV: self.COLOR.dark_grey,
                          self.VAR.E_PV2Grid: self.COLOR.yellow,

                          self.VAR.E_BatteryCharge: self.COLOR.green,
                          self.VAR.E_BatteryDischarge: self.COLOR.green,
                          self.VAR.E_Battery2Load: self.COLOR.dark_red,
                          self.VAR.E_Battery2EV: self.COLOR.green,
                          self.VAR.BatteryStateOfCharge: self.COLOR.turquoise,

                          self.VAR.E_EVCharge: self.COLOR.green,
                          self.VAR.E_EVDischarge: self.COLOR.green,
                          self.VAR.E_EV2Load: self.COLOR.dark_red,
                          self.VAR.E_EV2Battery: self.COLOR.green,
                          self.VAR.EVStateOfCharge: self.COLOR.turquoise,

                          self.VAR.TotalElectricityDemand: self.COLOR.green,
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

        ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)
        ax_1.set_ylabel("Electricity load (kW)", fontsize=20, labelpad=10)
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

    def plot_PV(self, id_household, id_environment, horizon,
                pv_2_load,
                pv_2_battery,
                pv_2_ev,
                pv_2_grid,
                electricity_price,
                feedin_tariff,
                **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.bar(x_values,
                 pv_2_load["values"],
                 label=pv_2_load["label"],
                 alpha=alpha_value,
                 color=pv_2_load["color"])

        ax_1.bar(x_values,
                 pv_2_battery["values"],
                 label=pv_2_battery["label"],
                 bottom=pv_2_load["values"],
                 alpha=alpha_value,
                 color=pv_2_battery["color"])

        ax_1.bar(x_values,
                 pv_2_ev["values"],
                 label=pv_2_ev["label"],
                 bottom=pv_2_load["values"] + pv_2_battery["values"],
                 alpha=alpha_value,
                 color=pv_2_ev["color"])

        ax_1.bar(x_values,
                 pv_2_grid["values"],
                 label=pv_2_grid["label"],
                 bottom=pv_2_load["values"] + pv_2_battery["values"] + pv_2_ev["values"],
                 alpha=alpha_value,
                 color=pv_2_grid["color"])

        ax_2 = ax_1.twinx()

        ax_2.plot(x_values,
                  electricity_price["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=electricity_price["label"],
                  color=electricity_price["color"])

        ax_2.plot(x_values,
                  feedin_tariff["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=feedin_tariff["label"],
                  color=feedin_tariff["color"])

        ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)
        ax_1.set_ylabel("PV Generation and Consumption (kW)", fontsize=20, labelpad=10)
        ax_2.set_ylabel("Electricity Price and Feed-in Tariff (€)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(kwargs["y_lim"])
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=3, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(20)

        fig_name = "PV_H" + str(id_household) + "_E" + str(id_environment) + "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def plot_Battery_EV(self, id_household, id_environment, horizon,
                        grid_2_battery_ev,
                        pv_2_battery_ev,
                        battery_ev_each_other,
                        battery_ev_2_load,
                        battery_ev_soc,
                        electricity_price,
                        feedin_tariff,
                        **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.bar(x_values,
                 grid_2_battery_ev["values"],
                 label=grid_2_battery_ev["label"],
                 alpha=alpha_value,
                 color=grid_2_battery_ev["color"])

        ax_1.bar(x_values,
                 pv_2_battery_ev["values"],
                 label=pv_2_battery_ev["label"],
                 bottom=grid_2_battery_ev["values"],
                 alpha=alpha_value,
                 color=pv_2_battery_ev["color"])

        ax_1.bar(x_values,
                 battery_ev_each_other["values"],
                 label=battery_ev_each_other["label"],
                 bottom=grid_2_battery_ev["values"] + pv_2_battery_ev["values"],
                 alpha=alpha_value,
                 color=battery_ev_each_other["color"])

        ax_1.bar(x_values,
                 battery_ev_2_load["values"],
                 label=battery_ev_2_load["label"],
                 alpha=alpha_value,
                 color=battery_ev_2_load["color"])

        # ax_1.plot(x_values,
        #           battery_ev_soc["values"],
        #           linewidth=linewidth_value,
        #           linestyle='-',
        #           label=battery_ev_soc["label"],
        #           color=battery_ev_soc["color"])

        ax_2 = ax_1.twinx()

        ax_2.plot(x_values,
                  electricity_price["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=electricity_price["label"],
                  color=electricity_price["color"])

        ax_2.plot(x_values,
                  feedin_tariff["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=feedin_tariff["label"],
                  color=feedin_tariff["color"])

        ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)
        ax_1.set_ylabel(kwargs["tech"] + " Charge and Discharge (kW)", fontsize=20, labelpad=10)
        ax_2.set_ylabel("Electricity Price and Feed-in Tariff (€)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(kwargs["y_lim"])
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=3, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(20)

        fig_name = kwargs["tech"] + "_H" + str(id_household) + "_E" + str(id_environment) + \
                   "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def visualization_SystemOperation(self, id_household, id_environment, **kargs):

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

        # ----------------
        # Data preparation
        # ----------------

        # 1. Environment
        ElectricityPrice = SystemOperation[self.VAR.ElectricityPrice]
        FeedinTariff = SystemOperation[self.VAR.FeedinTariff]

        # 2. Electricity balance
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

        # 3. Space heating and cooling



        # #####
        # Plots
        # #####

        ElectricityPrice_element = {"values": ElectricityPrice,
                                    "label": "ElectricityPrice",
                                    "color": self.VarColors[self.VAR.ElectricityPrice]}
        FeedinTariff_element = {"values": FeedinTariff,
                                "label": "FeedinTariff",
                                "color": self.VarColors[self.VAR.FeedinTariff]}

        # ------------------
        # Plot 1: SystemLoad
        # ------------------

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

        # ----------
        # Plot 2: PV
        # ----------
        PV2Load_element = {"values": PV2Load,
                           "label": "PV2Load",
                           "color": self.VarColors[self.VAR.E_PV2Load]}
        PV2Battery_element = {"values": PV2Battery,
                              "label": "PV2Battery",
                              "color": self.VarColors[self.VAR.E_PV2Battery]}
        PV2EV_element = {"values": PV2EV,
                         "label": "PV2EV",
                         "color": self.VarColors[self.VAR.E_PV2EV]}
        PV2Grid_element = {"values": PV2Grid,
                           "label": "PV2Grid",
                           "color": self.VarColors[self.VAR.E_PV2Grid]}

        self.plot_PV(id_household, id_environment, Horizon,
                     PV2Load_element,
                     PV2Battery_element,
                     PV2EV_element,
                     PV2Grid_element,
                     ElectricityPrice_element,
                     FeedinTariff_element)

        # ---------------
        # Plot 3: Battery
        # ---------------
        Grid2Battery_element = {"values": Grid2Battery,
                                "label": "Grid2Battery",
                                "color": self.VarColors[self.VAR.E_Grid2Battery]}
        PV2Battery_element = {"values": PV2Battery,
                              "label": "PV2Battery",
                              "color": self.VarColors[self.VAR.E_PV2Battery]}
        EV2Battery_element = {"values": EV2Battery,
                              "label": "EV2Battery",
                              "color": self.VarColors[self.VAR.E_EV2Battery]}
        Battery2Load_element = {"values": Battery2Load * (-1),
                                "label": "Battery2Load",
                                "color": self.VarColors[self.VAR.E_Battery2Load]}
        BatteryStateOfCharge_element = {"values": BatteryStateOfCharge,
                                        "label": "BatterySoC",
                                        "color": self.VarColors[self.VAR.BatteryStateOfCharge]}

        self.plot_Battery_EV(id_household, id_environment, Horizon,
                             Grid2Battery_element,
                             PV2Battery_element,
                             EV2Battery_element,
                             Battery2Load_element,
                             BatteryStateOfCharge_element,
                             ElectricityPrice_element,
                             FeedinTariff_element,
                             tech="Battery")

        # ----------
        # Plot 4: EV
        # ----------
        Grid2EV_element = {"values": Grid2EV,
                           "label": "Grid2EV",
                           "color": self.VarColors[self.VAR.E_Grid2EV]}
        PV2EV_element = {"values": PV2EV,
                         "label": "PV2EV",
                         "color": self.VarColors[self.VAR.E_PV2EV]}
        Battery2EV_element = {"values": Battery2EV,
                              "label": "Battery2EV",
                              "color": self.VarColors[self.VAR.E_Battery2EV]}
        EV2Load_element = {"values": EV2Load * (-1),
                           "label": "EV2Load",
                           "color": self.VarColors[self.VAR.E_EV2Load]}
        EVStateOfCharge_element = {"values": EVStateOfCharge,
                                   "label": "EVSoC",
                                   "color": self.VarColors[self.VAR.EVStateOfCharge]}

        self.plot_Battery_EV(id_household, id_environment, Horizon,
                             Grid2EV_element,
                             PV2EV_element,
                             Battery2EV_element,
                             EV2Load_element,
                             EVStateOfCharge_element,
                             ElectricityPrice_element,
                             FeedinTariff_element,
                             tech="EV")

    def run(self):
        self.visualization_SystemOperation(1, 2, horizon=[100, 200])
        self.visualization_SystemOperation(1, 2, horizon=[5500, 5600])




























