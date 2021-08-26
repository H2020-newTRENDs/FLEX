# -*- coding: utf-8 -*-
__author__ = 'Songmin'

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table
from C_Model_Operation.C1_REG import REG_Var


class Visualization:

    def __init__(self, conn):

        self.Conn = conn
        self.VAR = REG_Var()
        self.COLOR = CONS()
        self.VarColors = {self.VAR.ElectricityPrice: self.COLOR.blue,
                          self.VAR.FeedinTariff: self.COLOR.green,

                          self.VAR.E_BaseElectricityLoad: self.COLOR.light_brown,
                          self.VAR.E_DishWasher: self.COLOR.green,
                          self.VAR.E_WashingMachine: self.COLOR.green,
                          self.VAR.E_Dryer: self.COLOR.green,
                          self.VAR.E_SmartAppliance: self.COLOR.green,

                          self.VAR.Q_HeatPump: self.COLOR.dark_red,
                          self.VAR.Q_HeatPump_Ref: self.COLOR.dark_blue,
                          self.VAR.HeatPumpPerformanceFactor: self.COLOR.green,
                          self.VAR.E_HeatPump: self.COLOR.brown,
                          self.VAR.E_AmbientHeat: self.COLOR.green,
                          self.VAR.Q_HeatingElement: self.COLOR.yellow,
                          self.VAR.Q_HeatingElement_Ref: self.COLOR.turquoise,
                          self.VAR.Q_RoomHeating: self.COLOR.light_brown,

                          self.VAR.Q_RoomCooling: self.COLOR.green,
                          self.VAR.E_RoomCooling: self.COLOR.blue,

                          self.VAR.Q_HotWater: self.COLOR.green,
                          self.VAR.E_HotWater: self.COLOR.dark_blue,

                          self.VAR.E_Grid: self.COLOR.black,
                          self.VAR.E_Grid2Load: self.COLOR.black,
                          self.VAR.E_Grid2Battery: self.COLOR.dark_red,

                          self.VAR.E_PV: self.COLOR.green,
                          self.VAR.E_PV2Load: self.COLOR.orange,
                          self.VAR.E_PV2Battery: self.COLOR.blue,
                          self.VAR.E_PV2Grid: self.COLOR.dark_green,

                          self.VAR.E_BatteryCharge: self.COLOR.green,
                          self.VAR.E_BatteryDischarge: self.COLOR.green,
                          self.VAR.E_Battery2Load: self.COLOR.red,
                          self.VAR.BatteryStateOfCharge: self.COLOR.turquoise,

                          self.VAR.E_Load: self.COLOR.orange,
                          self.VAR.E_Load_Ref: self.COLOR.blue,
                          self.VAR.OutsideTemperature: self.COLOR.blue,
                          self.VAR.RoomTemperature: self.COLOR.red,
                          self.VAR.BuildingMassTemperature: self.COLOR.orange,
                          }
        self.PlotHorizon = {}
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.SystemOperationHour = DB().read_DataFrame(REG_Table().Res_SystemOperationHour, self.Conn)
        self.SystemOperationYear = DB().read_DataFrame(REG_Table().Res_SystemOperationYear, self.Conn)
        self.ReferenceOperationHour = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling, self.Conn)
        self.ReferenceOperationYear = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling_Year, self.Conn)

    def adjust_ylim(self, lim_array):

        tiny = (lim_array[1] - lim_array[0]) * 0.02
        left = lim_array[0] - tiny
        right = lim_array[1] + tiny

        return np.array((left, right))

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
                        **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.15, 0.68, 0.8])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        # demand
        ax_1.bar(x_values,
                 base_load["values"],
                 label=base_load["label"],
                 alpha=alpha_value,
                 color=base_load["color"])

        ax_1.bar(x_values,
                 hot_water["values"],
                 label=hot_water["label"],
                 bottom=base_load["values"],
                 alpha=alpha_value,
                 color=hot_water["color"])

        ax_1.bar(x_values,
                 smart_appliance["values"],
                 label=smart_appliance["label"],
                 bottom=base_load["values"] + hot_water["values"],
                 alpha=alpha_value,
                 color=smart_appliance["color"])

        ax_1.bar(x_values,
                 heat_pump["values"],
                 label=heat_pump["label"],
                 bottom=base_load["values"] + hot_water["values"] + smart_appliance["values"],
                 alpha=alpha_value,
                 color=heat_pump["color"])

        ax_1.bar(x_values,
                 heat_element["values"],
                 label=heat_element["label"],
                 bottom=base_load["values"] + hot_water["values"] + smart_appliance["values"] + heat_pump["values"],
                 alpha=alpha_value,
                 color=heat_element["color"])

        ax_1.bar(x_values,
                 room_cooling["values"],
                 label=room_cooling["label"],
                 bottom=base_load["values"] + hot_water["values"] + smart_appliance["values"] + heat_pump["values"] +
                        heat_element["values"],
                 alpha=alpha_value,
                 color=room_cooling["color"])

        # supply
        ax_1.bar(x_values,
                 - pv_2_load["values"],
                 label=pv_2_load["label"],

                 alpha=alpha_value,
                 color=pv_2_load["color"])

        ax_1.bar(x_values,
                 - grid_2_load["values"],
                 label=grid_2_load["label"],
                 bottom=- pv_2_load["values"],
                 alpha=alpha_value,
                 color=grid_2_load["color"])

        ax_1.bar(x_values,
                 - battery_2_load["values"],
                 label=battery_2_load["label"],
                 bottom=- pv_2_load["values"] - grid_2_load["values"],
                 alpha=alpha_value,
                 color=battery_2_load["color"])

        ax_1.grid(color='grey', linestyle='--', linewidth=1)

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_1.set_xticks(tick_position)
            ax_1.set_xticklabels(x_ticks_label, fontsize=20)
        else:
            ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)

        ax_1.set_ylabel("Electricity load(+) and supply(-) (kW)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"]))

        figure.legend(fontsize=15, bbox_to_anchor=(1.01, 0, 0.2, 1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=1, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)

        fig_name = "SystemLoad_H" + str(id_household) + "_E" + str(id_environment) + "_H" + str(horizon[0]) + "_" + str(
            horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def plot_PV(self, id_household, id_environment, horizon,
                pv_2_load,
                pv_2_battery,
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
                 pv_2_grid["values"],
                 label=pv_2_grid["label"],
                 bottom=pv_2_load["values"] + pv_2_battery["values"],
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

        ax_1.grid(color='grey', linestyle='--', linewidth=1)

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_1.set_xticks(tick_position)
            ax_1.set_xticklabels(x_ticks_label, fontsize=20)
        else:
            ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)

        ax_1.set_ylabel("PV Generation and Consumption (kW)", fontsize=20, labelpad=10)
        ax_2.set_ylabel("Electricity Price and Feed-in Tariff (€)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"][0]))
            ax_2.set_ylim(self.adjust_ylim(kwargs["y_lim"][1]))
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=3, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(20)

        fig_name = "PV_H" + str(id_household) + "_E" + str(id_environment) + "_H" + str(horizon[0]) + "_" + str(
            horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def plot_Battery(self, id_household, id_environment, horizon,
                     grid_2_battery,
                     pv_2_battery,
                     battery_2_load,
                     battery_soc,
                     **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.bar(x_values,
                 grid_2_battery["values"],
                 label=grid_2_battery["label"],
                 alpha=alpha_value,
                 color=grid_2_battery["color"])

        ax_1.bar(x_values,
                 pv_2_battery["values"],
                 label=pv_2_battery["label"],
                 bottom=grid_2_battery["values"],
                 alpha=alpha_value,
                 color=pv_2_battery["color"])

        ax_1.bar(x_values,
                 battery_2_load["values"] * (-1),
                 label=battery_2_load["label"],
                 alpha=alpha_value,
                 color=battery_2_load["color"])

        ax_2 = ax_1.twinx()

        ax_2.plot(x_values,
                  battery_soc["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=battery_soc["label"],
                  color=battery_soc["color"])

        ax_1.grid(color='grey', linestyle='--', linewidth=1)

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_1.set_xticks(tick_position)
            ax_1.set_xticklabels(x_ticks_label, fontsize=20)
        else:
            ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)

        ax_1.set_ylabel(kwargs["tech"] + " Charge and Discharge (kW)", fontsize=20, labelpad=10)
        ax_2.set_ylabel(kwargs["tech"] + " State of Charge (kWh)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"][0]))
            ax_2.set_ylim(self.adjust_ylim(kwargs["y_lim"][1]))
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

    def plot_HeatingTemperature(self, id_household, id_environment, horizon,
                                room_temperature,
                                building_mass_temperature,
                                outside_temperature,
                                **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.plot(x_values,
                  room_temperature["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=room_temperature["label"],
                  color=room_temperature["color"])

        ax_1.plot(x_values,
                  building_mass_temperature["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=building_mass_temperature["label"],
                  color=building_mass_temperature["color"])

        ax_2 = ax_1.twinx()
        ax_2.plot(x_values,
                  outside_temperature["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=outside_temperature["label"],
                  color=outside_temperature["color"])

        ax_1.grid(color='grey', linestyle='--', linewidth=1)

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_1.set_xticks(tick_position)
            ax_1.set_xticklabels(x_ticks_label, fontsize=20)
        else:
            ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)

        ax_1.set_ylabel("Room and Building Mass Temperature (℃)", fontsize=20, labelpad=10)
        ax_2.set_ylabel("Outside Temperature (℃)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"][0]))
            ax_2.set_ylim(self.adjust_ylim(kwargs["y_lim"][1]))
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=2, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(20)

        fig_name = "HeatingTemperature_H" + str(id_household) + "_E" + str(id_environment) + \
                   "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def plot_RoomHeating(self, id_household, id_environment, horizon,
                         heat_pump,
                         heat_element,
                         tank_output,
                         electricity_price,
                         **kwargs):

        figure = plt.figure(figsize=(20, 8), dpi=200, frameon=False)
        ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2

        ax_1.bar(x_values,
                 heat_pump["values"],
                 label=heat_pump["label"],
                 alpha=alpha_value,
                 color=heat_pump["color"])

        ax_1.bar(x_values,
                 heat_element["values"],
                 label=heat_element["label"],
                 bottom=heat_pump["values"],
                 alpha=alpha_value,
                 color=heat_element["color"])

        # ax_1.bar(x_values,
        #          - tank_output["values"],
        #          label=tank_output["label"],
        #          alpha=alpha_value,
        #          color=tank_output["color"])

        ax_2 = ax_1.twinx()

        ax_2.plot(x_values,
                  electricity_price["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=electricity_price["label"],
                  color=electricity_price["color"])

        ax_1.grid(color='grey', linestyle='--', linewidth=1)

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_1.set_xticks(tick_position)
            ax_1.set_xticklabels(x_ticks_label, fontsize=20)
        else:
            ax_1.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)

        # ax_1.set_ylabel("Energy output of boiler(+) and tank(-) (kW)", fontsize=20, labelpad=10)
        ax_1.set_ylabel("Energy output of boiler (kW)", fontsize=20, labelpad=10)

        ax_2.set_ylabel("Electricity Price (€)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"][0]))
            ax_2.set_ylim(self.adjust_ylim(kwargs["y_lim"][1]))
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.1), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=2, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_1.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label2.set_fontsize(20)

        fig_name = "RoomHeating_H" + str(id_household) + "_E" + str(id_environment) + \
                   "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.close(figure)

    def visualization_SystemOperation(self, id_household, id_environment, **kargs):

        HourStart = 1
        HourEnd = 8760
        if "horizon" in kargs:
            HourStart = kargs["horizon"][0]
            HourEnd = kargs["horizon"][1]
        else:
            pass
        if "week" in kargs:
            TimeStructure_select = self.TimeStructure.loc[self.TimeStructure["ID_Week"] == kargs["week"]]
            HourStart = TimeStructure_select.iloc[0]["ID_Hour"]
            HourEnd = TimeStructure_select.iloc[-1]["ID_Hour"]
        else:
            pass

        SystemOperation = self.SystemOperationHour.loc[
            (self.SystemOperationHour[self.VAR.ID_Household] == id_household) &
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
        Grid = SystemOperation[self.VAR.E_Grid] / 1_000  # kW
        Grid2Load = SystemOperation[self.VAR.E_Grid2Load] / 1_000  # kW
        Grid2Battery = SystemOperation[self.VAR.E_Grid2Battery] / 1_000  # kW

        # Load
        BaseElectricityLoad = SystemOperation[self.VAR.E_BaseElectricityLoad]  # kW
        SmartAppliance = SystemOperation[self.VAR.E_SmartAppliance]  # kW
        HeatPump = SystemOperation[self.VAR.E_HeatPump]  # kW
        HeatElement = SystemOperation[self.VAR.Q_HeatingElement]  # kW
        Cooling = SystemOperation[self.VAR.E_RoomCooling] / 1_000  # kW
        HotWater = SystemOperation[self.VAR.E_HotWater] / 1_000  # kW # kW

        # PV
        PV = SystemOperation[self.VAR.E_PV] / 1_000  # kW
        PV2Load = SystemOperation[self.VAR.E_PV2Load] / 1_000  # kW
        PV2Battery = SystemOperation[self.VAR.E_PV2Battery] / 1_000  # kW
        PV2Grid = SystemOperation[self.VAR.E_PV2Grid] / 1_000  # kW

        # Battery
        BatteryCharge = SystemOperation[self.VAR.E_BatteryCharge] / 1_000  # kW
        BatteryDischarge = SystemOperation[self.VAR.E_BatteryDischarge] / 1_000  # kW
        Battery2Load = SystemOperation[self.VAR.E_Battery2Load] / 1_000  # kW
        BatteryStateOfCharge = SystemOperation[self.VAR.BatteryStateOfCharge] / 1_000  # kWh

        # 3. Space heating and cooling
        Q_HeatPump = SystemOperation[self.VAR.Q_HeatPump]  # kW
        Q_RoomHeating = SystemOperation[self.VAR.Q_RoomHeating]  # kW
        OutsideTemperature = SystemOperation[self.VAR.OutsideTemperature]
        RoomTemperature = SystemOperation[self.VAR.RoomTemperature]
        BuildingMassTemperature = SystemOperation[self.VAR.BuildingMassTemperature]

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

        if Horizon[0] < 100:
            y_lim_range = np.array((-15, 15))
        else:
            y_lim_range = np.array((-7, 7))
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
                             x_label_weekday=True,
                             y_lim=y_lim_range)

        # ----------
        # Plot 2: PV
        # ----------
        PV2Load_element = {"values": PV2Load,
                           "label": "PV2Load",
                           "color": self.VarColors[self.VAR.E_PV2Load]}
        PV2Battery_element = {"values": PV2Battery,
                              "label": "PV2Battery",
                              "color": self.VarColors[self.VAR.E_PV2Battery]}
        PV2Grid_element = {"values": PV2Grid,
                           "label": "PV2Grid",
                           "color": self.VarColors[self.VAR.E_PV2Grid]}

        if Horizon[0] < 100:
            y1_lim_range = np.array((0, 0.8))
            y2_lim_range = np.array((0, 0.4))
        else:
            y1_lim_range = np.array((0, 8))
            y2_lim_range = np.array((0, 0.4))
        self.plot_PV(id_household, id_environment, Horizon,
                     PV2Load_element,
                     PV2Battery_element,
                     PV2Grid_element,
                     ElectricityPrice_element,
                     FeedinTariff_element,
                     x_label_weekday=True,
                     y_lim=(y1_lim_range, y2_lim_range))

        # ---------------
        # Plot 3: Battery
        # ---------------
        Grid2Battery_element = {"values": Grid2Battery,
                                "label": "Grid2Battery",
                                "color": self.VarColors[self.VAR.E_Grid2Battery]}
        PV2Battery_element = {"values": PV2Battery,
                              "label": "PV2Battery",
                              "color": self.VarColors[self.VAR.E_PV2Battery]}
        Battery2Load_element = {"values": Battery2Load,
                                "label": "Battery2Load",
                                "color": self.VarColors[self.VAR.E_Battery2Load]}
        BatteryStateOfCharge_element = {"values": BatteryStateOfCharge,
                                        "label": "BatterySoC",
                                        "color": self.VarColors[self.VAR.BatteryStateOfCharge]}

        if Horizon[0] < 100:
            y1_lim_range = np.array((-6, 6))
            y2_lim_range = np.array((0, 12))
        else:
            y1_lim_range = np.array((-6, 6))
            y2_lim_range = np.array((0, 12))
        self.plot_Battery(id_household, id_environment, Horizon,
                          Grid2Battery_element,
                          PV2Battery_element,
                          Battery2Load_element,
                          BatteryStateOfCharge_element,
                          tech="Battery",
                          x_label_weekday=True,
                          y_lim=(y1_lim_range, y2_lim_range))

        # --------------------
        # Plot 4: Room heating
        # --------------------
        Q_HeatPump_element = {"values": Q_HeatPump,
                              "label": "HeatPump",
                              "color": self.VarColors[self.VAR.Q_HeatPump]}
        Q_RoomHeating_element = {"values": Q_RoomHeating,
                                 "label": "TankOutput",
                                 "color": self.VarColors[self.VAR.Q_RoomHeating]}

        OutsideTemperature_element = {"values": OutsideTemperature,
                                      "label": "OutsideTemperature",
                                      "color": self.VarColors[self.VAR.OutsideTemperature]}
        RoomTemperature_element = {"values": RoomTemperature,
                                   "label": "RoomTemperature",
                                   "color": self.VarColors[self.VAR.RoomTemperature]}
        BuildingMassTemperature_element = {"values": BuildingMassTemperature,
                                           "label": "BuildingMassTemperature",
                                           "color": self.VarColors[self.VAR.BuildingMassTemperature]}

        if Horizon[0] < 100:
            y1_lim_range = np.array((-20, 20))
            y2_lim_range = np.array((0.2, 0.4))
        else:
            y1_lim_range = np.array((0, 12))
            # y1_lim_range = np.array((-5, 15))
            y2_lim_range = np.array((0.2, 0.4))
        self.plot_RoomHeating(id_household, id_environment, Horizon,
                              Q_HeatPump_element,
                              HeatElement_element,
                              Q_RoomHeating_element,
                              ElectricityPrice_element,
                              x_label_weekday=True,
                              y_lim=(y1_lim_range, y2_lim_range))

        if Horizon[0] < 100:
            y1_lim_range = np.array((12, 22))
            y2_lim_range = np.array((-10, 0))
        else:
            y1_lim_range = np.array((10, 30))
            y2_lim_range = np.array((-10, 30))
        self.plot_HeatingTemperature(id_household, id_environment, Horizon,
                                     RoomTemperature_element,
                                     BuildingMassTemperature_element,
                                     OutsideTemperature_element,
                                     x_label_weekday=True,
                                     y_lim=(y1_lim_range, y2_lim_range))

    def visualize_comparison2Reference(self, id_household, id_environment, **kargs):

        HourStart = 1
        HourEnd = 8760
        if "horizon" in kargs:
            HourStart = kargs["horizon"][0]
            HourEnd = kargs["horizon"][1]
        else:
            pass
        if "week" in kargs:
            TimeStructure_select = self.TimeStructure.loc[self.TimeStructure["ID_Week"] == kargs["week"]]
            HourStart = TimeStructure_select.iloc[0]["ID_Hour"]
            HourEnd = TimeStructure_select.iloc[-1]["ID_Hour"]
        else:
            pass
        Horizon = [HourStart, HourEnd]
        Optimization_Results = self.SystemOperationHour.loc[
            (self.SystemOperationHour[self.VAR.ID_Household] == id_household) &
            (self.SystemOperationHour[self.VAR.ID_Environment] == id_environment) &
            (self.SystemOperationHour[self.VAR.ID_Hour] >= HourStart) &
            (self.SystemOperationHour[self.VAR.ID_Hour] <= HourEnd)]

        Reference_Results = self.ReferenceOperationHour.loc[
            (self.ReferenceOperationHour[self.VAR.ID_Household] == id_household) &
            (self.ReferenceOperationHour[self.VAR.ID_Environment] == id_environment) &
            (self.ReferenceOperationHour[self.VAR.ID_Hour] >= HourStart) &
            (self.ReferenceOperationHour[self.VAR.ID_Hour] <= HourEnd)]

        Q_HeatPump_Reference = {"values": Reference_Results.Q_HeatPump.to_numpy(),
                                "label": "HeatPump_Ref",
                                "color": self.VarColors[self.VAR.Q_HeatPump_Ref]}
        Q_HeatPump_Optim = {"values": Optimization_Results.Q_HeatPump.to_numpy(),
                            "label": "HeatPump_Optim",
                            "color": self.VarColors[self.VAR.Q_HeatPump]}

        Q_HeatingElement_Reference = {"values": Reference_Results.Q_HeatingElement.to_numpy(),
                                      "label": "Heating Element Ref",
                                      "color": self.VarColors[self.VAR.Q_HeatingElement_Ref]}
        Q_HeatingElement_Optim = {"values": Optimization_Results.Q_HeatingElement.to_numpy(),
                                  "label": "Heating Element Optim",
                                  "color": self.VarColors[self.VAR.Q_HeatingElement]}

        E_Load = {"values": Optimization_Results.E_Load.to_numpy(),
                  "label": "E Load Optim",
                  "color": self.VarColors[self.VAR.E_Load]}
        E_Load_Ref = {"values": Reference_Results.E_Load.to_numpy(),
                      "label": "E Load Ref",
                      "color": self.VarColors[self.VAR.E_Load_Ref]}

        if Horizon[0] < 100:
            y1_lim_range = np.array((0, 20))
            y2_lim_range = np.array((0, 8))
        else:
            y1_lim_range = np.array((0, 12))
            # y1_lim_range = np.array((-5, 15))
            y2_lim_range = np.array((0, 6))

        self.plot_load_comparison(id_household, id_environment, Horizon,
                                  Q_HeatPump_Reference,
                                  Q_HeatPump_Optim,
                                  Q_HeatingElement_Reference,
                                  Q_HeatingElement_Optim,
                                  E_Load,
                                  E_Load_Ref,
                                  x_label_weekday=True,
                                  y_lim=(y1_lim_range, y2_lim_range))

    def plot_total_comparison(self, id_household, id_environment):
        # reference_results = self.ReferenceOperationYear.loc[:, ["ID_Household",
        #                                                         "OperationCost",
        #                                                         "Year_Q_HeatPump",
        #                                                         "Year_E_HeatPump",
        #                                                         "Year_Q_HeatingElement",
        #                                                         "Year_Q_RoomCooling",
        #                                                         "Year_E_PV2Load",
        #                                                         "Year_E_PV2Grid",
        #                                                         "Year_E_Load"
        #                                                         ]]
        # reference_results.loc[:, "Optimization"] = "Reference"
        #
        # optimization_results = self.SystemOperationYear.loc[:, ["ID_Household",
        #                                                         "OperationCost",
        #                                                         "Year_Q_HeatPump",
        #                                                         "Year_E_HeatPump",
        #                                                         "Year_Q_HeatingElement",
        #                                                         "Year_Q_RoomCooling",
        #                                                         "Year_E_PV2Load",
        #                                                         "Year_E_PV2Grid",
        #                                                         "Year_E_Load"
        #                                                         ]]

        reference_results = self.ReferenceOperationYear
        optimization_results = self.SystemOperationYear
        reference_results.loc[:, "Option"] = "Referenz"
        optimization_results.loc[:, "Option"] = "Optimization"

        frame = pd.concat([self.ReferenceOperationYear, self.SystemOperationYear], axis=0)

        fig, axes = plt.subplots(2, 2, sharey=True, figsize=[15, 10])
        axes = axes.flatten()
        sns.violinplot(x="ID_Building", y="OperationCost", hue="Option", data=frame, ax=axes[0], split=True,
                       inner="stick", palette="muted")
        sns.violinplot(x="Household_PVPower", y="OperationCost", hue="Option", data=frame, ax=axes[1], split=True,
                       inner="stick", palette="muted")

        sns.violinplot(x="Household_TankSize", y="OperationCost", hue="Option", data=frame, ax=axes[2], split=True,
                       inner="stick", palette="muted")
        sns.violinplot(x="Household_BatteryCapacity", y="OperationCost", hue="Option", data=frame, ax=axes[3],
                       split=True, inner="stick", palette="muted")
        plt.show()





        frame = pd.melt(frame=frame, id_vars=["Optimization", "ID_Household"])

        sns.boxplot(x=frame["variable"], y=frame["value"], hue="Optimization", data=frame)
        ax = plt.gca()
        plt.xticks(rotation=45)

        plt.show()


        pass

    def plot_load_comparison(self, id_household, id_environment, horizon,
                             Q_HeatPump_Reference,
                             Q_HeatPump_Optim,
                             Q_HeatingElement_Reference,
                             Q_HeatingElement_Optim,
                             E_Load,
                             E_Load_Ref,
                             **kwargs):
        # plot comparison:
        figure, (ax_1, ax_2) = plt.subplots(2, 1, figsize=(20, 8), dpi=200, frameon=False)
        # ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = range(horizon[0], horizon[1] + 1)
        alpha_value = 0.8
        linewidth_value = 2
        # heat load
        ax_1.plot(x_values,
                  Q_HeatPump_Optim["values"],
                  linewidth=linewidth_value,
                  label=Q_HeatPump_Optim["label"],
                  alpha=alpha_value,
                  color=Q_HeatPump_Optim["color"])
        # heat load reference
        ax_1.plot(x_values,
                  Q_HeatPump_Reference["values"],
                  linewidth=linewidth_value,
                  label=Q_HeatPump_Reference["label"],
                  alpha=alpha_value,
                  color=Q_HeatPump_Reference["color"])

        ax_1.bar(x_values,
                 Q_HeatingElement_Optim["values"],
                 label=Q_HeatingElement_Optim["label"],
                 alpha=alpha_value,
                 color=Q_HeatingElement_Optim["color"])

        ax_1.bar(x_values,
                 Q_HeatingElement_Reference["values"],
                 label=Q_HeatingElement_Reference["label"],
                 alpha=alpha_value,
                 color=Q_HeatingElement_Reference["color"])

        # ax_2 = ax_1.twinx()

        ax_2.plot(x_values,
                  E_Load["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=E_Load["label"],
                  color=E_Load["color"])

        ax_2.plot(x_values,
                  E_Load_Ref["values"],
                  linewidth=linewidth_value,
                  linestyle='-',
                  label=E_Load_Ref["label"],
                  color=E_Load_Ref["color"])

        ax_1.grid(color='grey', linestyle='--', linewidth=1, which="both", axis="y")
        ax_2.grid(color='grey', linestyle='--', linewidth=1, which="both", axis="y")

        if "x_label_weekday" in kwargs:
            tick_position = [horizon[0] + 11.5] + [horizon[0] + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            ax_2.set_xticks(tick_position)
            ax_2.set_xticklabels(x_ticks_label, fontsize=20)
            ax_1.xaxis.set_visible(False)
        else:
            ax_2.set_xlabel("Hour of the Year", fontsize=20, labelpad=10)
            ax_1.xaxis.set_visible(False)
        # ax_1.set_ylabel("Energy output of boiler(+) and tank(-) (kW)", fontsize=20, labelpad=10)
        ax_1.set_ylabel("Heating Energy (kW)", fontsize=20, labelpad=10)

        ax_1.set_ylabel("Thermal Heating \n Power (kW)", fontsize=20, labelpad=10)
        ax_2.set_ylabel("Total Electric \n Load (kWh)", fontsize=20, labelpad=10)
        if "y_lim" in kwargs:
            ax_1.set_ylim(self.adjust_ylim(kwargs["y_lim"][0]))
            ax_2.set_ylim(self.adjust_ylim(kwargs["y_lim"][1]))
        figure.legend(fontsize=15, bbox_to_anchor=(0, 1.04, 1, 0.2), bbox_transform=ax_1.transAxes,
                      loc=1, ncol=3, borderaxespad=0, mode='expand', frameon=True)

        for tick in ax_2.xaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_1.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)
        for tick in ax_2.yaxis.get_major_ticks():
            tick.label1.set_fontsize(20)

        fig_name = "Heating Comparison" + "_H" + str(id_household) + "_E" + str(id_environment) + \
                   "_H" + str(horizon[0]) + "_" + str(horizon[1])
        figure.savefig(CONS().FiguresPath + fig_name + ".png", dpi=200, format='PNG')
        plt.show()
        plt.close(figure)

    def run(self):
        for household_id in range(3):
            for environment_id in range(1, 2):
                self.visualization_SystemOperation(household_id + 1, environment_id, week=8)
                self.visualization_SystemOperation(household_id + 1, environment_id, week=34)


if __name__ == "__main__":
    CONN = DB().create_Connection(CONS().RootDB)
    Visualization(CONN).plot_total_comparison(1, 1)
    Visualization(CONN).visualize_comparison2Reference(1, 1, week=8)

    Visualization(CONN).run()
