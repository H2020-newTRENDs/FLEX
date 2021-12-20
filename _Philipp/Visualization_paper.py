# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re
import h5py

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
                          self.VAR.E_BatteryCharge_Ref: self.COLOR.dark_green,
                          self.VAR.E_BatteryDischarge: self.COLOR.green,
                          self.VAR.E_Battery2Load: self.COLOR.red,
                          self.VAR.E_Battery2Load_Ref: self.COLOR.dark_red,
                          self.VAR.BatteryStateOfCharge: self.COLOR.turquoise,
                          self.VAR.BatteryStateOfCharge_Ref: self.COLOR.purple,

                          self.VAR.E_Load: self.COLOR.orange,
                          self.VAR.E_Load_Ref: self.COLOR.blue,
                          self.VAR.OutsideTemperature: self.COLOR.blue,
                          self.VAR.RoomTemperature: self.COLOR.red,
                          self.VAR.BuildingMassTemperature: self.COLOR.orange,
                          }
        self.PlotHorizon = {}
        self.TimeStructure = DB().read_DataFrame(REG_Table().Sce_ID_TimeStructure, self.Conn)
        self.SystemOperationYear = DB().read_DataFrame(REG_Table().Res_SystemOperationYear, self.Conn)
        self.ReferenceOperationYear = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling_Year, self.Conn)
        self.figure_path = CONS().FiguresPath
        self.figure_path_aggregated = self.figure_path / Path("Aggregated_Results")
        self.scenario_variables = {"Household_TankSize": {0: "l", 1500: "l"},
                                   "Household_CoolingAdoption": {0: "", 1: ""},
                                   "Household_PVPower": {0: "kWp", 5: "kWp", 10: "kWp"},
                                   "Household_BatteryCapacity": {0: "kWh", 7000: "kWh"},
                                   "Environment_ElectricityPriceType": {1: "variable", 2: "flat"},
                                   "ID_AgeGroup": {1: "fixed \n temperature", 2: "smart"},
                                   "ID_SpaceHeating": {"Air_HP": "Air HP", "Water_HP": "Ground HP"}}

        path2buildingsnumber = CONS().ProjectPath / Path("_Philipp/inputdata/AUT") / Path(
            "040_aut__2__BASE__1_zzz_short_bc_seg__b_building_segment_sh.csv")
        number_of_buildings_frame = pd.read_csv(path2buildingsnumber, sep=";", encoding="ISO-8859-1", header=0)
        self.number_of_buildings_frame = number_of_buildings_frame.iloc[4:, :]

    def determine_ylabel(self, label) -> str():
        """returns the string of the ylabel"""
        if label == "Year_E_Grid":
            ylabel = "Electricity demand from the grid (kWh)"
        elif label == "OperationCost":
            ylabel = "Operation cost"
        elif label == "Year_PVSelfSufficiencyRate":
            ylabel = "self sufficiency (%)"
        elif label == "Year_E_Load":
            ylabel = "total electricity demand (kWh)"
        elif label == "Year_E_PV2Grid":
            ylabel = "PV generation sold to the grid (kWh)"
        elif label == "Year_PVSelfConsumptionRate":
            ylabel = "Self-consumption rate (%)"

        else:
            print("cant determine label")
            ylabel = label

        return ylabel

    def adjust_ylim(self, lim_array):

        tiny = (lim_array[1] - lim_array[0]) * 0.02
        left = lim_array[0] - tiny
        right = lim_array[1] + tiny

        return np.array((left, right))

    def get_total_number_of_buildings(self, building_index: int) -> (float, float):
        """takes the invert building index and returns the number of buildings
        with heat pumps and total number of buildings."""
        total_number_of_buildings = self.number_of_buildings_frame.loc[
                                    pd.to_numeric(
                                        self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index,
                                    :
                                    ].number_of_buildings.sum()

        number_of_buildings_with_HP = self.number_of_buildings_frame.loc[
            (pd.to_numeric(self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index) &
            ((self.number_of_buildings_frame.loc[:,
              "heatingsystem"] == "geothermal_central_heatpump(air/water)_Heat pump air/water") |
             (self.number_of_buildings_frame.loc[:,
              "heatingsystem"] == "geothermal_central_heatpump(water/water)_Heat pump brine/water shallow"))
            ].number_of_buildings.sum()

        return total_number_of_buildings, number_of_buildings_with_HP

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

        Battery_SOC_Ref = {"values": Reference_Results.BatteryStateOfCharge.to_numpy(),
                           "label": "Battery_SOC_Ref",
                           "color": self.VarColors[self.VAR.BatteryStateOfCharge_Ref]}

        Battery2Load_Ref = {"values": Reference_Results.E_Battery2Load.to_numpy(),
                            "label": "Battery2Load_Ref",
                            "color": self.VarColors[self.VAR.E_Battery2Load_Ref]}

        BatteryCharge_Ref = {"values": Reference_Results.E_BatteryCharge.to_numpy(),
                             "label": "BatteryCharge_Ref",
                             "color": self.VarColors[self.VAR.E_BatteryCharge_Ref]}

        Battery_SOC = {"values": Optimization_Results.BatteryStateOfCharge.to_numpy(),
                       "label": "Battery_SOC",
                       "color": self.VarColors[self.VAR.BatteryStateOfCharge]}

        Battery2Load = {"values": Optimization_Results.E_Battery2Load.to_numpy(),
                        "label": "Battery2Load",
                        "color": self.VarColors[self.VAR.E_Battery2Load]}

        BatteryCharge = {"values": Optimization_Results.E_BatteryCharge.to_numpy(),
                         "label": "BatteryCharge",
                         "color": self.VarColors[self.VAR.E_BatteryCharge]}

        self.plot_battery_comparison(id_household, id_environment, Horizon,
                                     Battery_SOC_Ref,
                                     Battery2Load_Ref,
                                     BatteryCharge_Ref,
                                     Battery_SOC,
                                     Battery2Load,
                                     BatteryCharge,
                                     x_label_weekday=True,
                                     y_lim=(y1_lim_range, y2_lim_range)
                                     )

    def plot_average_results(self):
        reference_results_year = self.ReferenceOperationYear
        optimization_results_year = self.SystemOperationYear
        reference_results_year.loc[:, "Option"] = "Reference"
        optimization_results_year.loc[:, "Option"] = "Optimization"
        # exclude Row houses:
        reference_results_year = reference_results_year[reference_results_year.loc[:, "ID_Building"] < 12]
        optimization_results_year = optimization_results_year[optimization_results_year.loc[:, "ID_Building"] < 12]
        # find buildings with the same appliances etc.:
        unique_PVPower = np.unique(optimization_results_year.loc[:, "Household_PVPower"])
        unique_TankSize = np.unique(optimization_results_year.loc[:, "Household_TankSize"])
        unique_BatteryCapacity = np.unique(optimization_results_year.loc[:, "Household_BatteryCapacity"])
        unique_Environment = np.unique(optimization_results_year.loc[:, "ID_Environment"])

        number_of_configurations = len(unique_PVPower) * len(unique_TankSize) * len(unique_BatteryCapacity)
        # number of profiles aggregated:
        number_of_profiles_aggregated = len(optimization_results_year) / number_of_configurations
        # create number of necessary variables:
        config_ids_opt = {}
        config_ids_ref = {}

        config_frame_opt_year = {}
        config_frame_ref_year = {}
        i = 0
        for environment in unique_Environment:
            for PVPower in unique_PVPower:
                for TankSize in unique_TankSize:
                    for BatteryCapacity in unique_BatteryCapacity:
                        optim_year = optimization_results_year.loc[
                            (optimization_results_year.loc[:, "Household_PVPower"] == PVPower) &
                            (optimization_results_year.loc[:, "Household_TankSize"] == TankSize) &
                            (optimization_results_year.loc[:, "Household_BatteryCapacity"] == BatteryCapacity) &
                            (optimization_results_year.loc[:, "ID_Environment"] == environment)]

                        optim_ids = optim_year.ID_Household.to_numpy()

                        ref_year = reference_results_year.loc[
                            (reference_results_year.loc[:, "Household_PVPower"] == PVPower) &
                            (reference_results_year.loc[:, "Household_TankSize"] == TankSize) &
                            (reference_results_year.loc[:, "Household_BatteryCapacity"] == BatteryCapacity) &
                            (optimization_results_year.loc[:, "ID_Environment"] == environment)]

                        ref_ids = ref_year.ID_Household.to_numpy()

                        # save IDs to dict
                        config_ids_opt[("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity / 1_000), TankSize),
                                        "E{:n}".format(environment))] = optim_ids
                        config_ids_ref[
                            ("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity / 1_000), TankSize, environment),
                             "E{:n}".format(environment))] = ref_ids

                        # save same household configuration total results to dict
                        config_frame_opt_year[
                            ("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity / 1_000), TankSize),
                             "E{:n}".format(environment))] = optim_year
                        config_frame_ref_year[
                            ("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity / 1_000), TankSize),
                             "E{:n}".format(environment))] = ref_year
                        i += 1

        def plot_barplot(frame_opt, frame_ref, variable2plot):
            # create barplot with total amounts:
            figure = plt.figure()
            # figure.suptitle(
            #     variable2plot.replace("Year_",
            #                           "") + " average difference between optimization and reference \n for different electricity prices")
            y_axis_title = variable2plot.replace("Year_", "").replace("E_Grid", "Electricity from the grid") + \
                           "\n optimization - reference (kWh)"
            barplot_frame = pd.DataFrame(
                columns=["Configuration", y_axis_title, "Scenario"])  # dataframe for seaborn:
            for key in frame_opt.keys():
                Sum_opt = frame_opt[key][variable2plot].sum() / number_of_profiles_aggregated
                Sum_ref = frame_ref[key][variable2plot].sum() / number_of_profiles_aggregated
                Differenz = Sum_opt - Sum_ref

                # check what the environment means:
                if key[1] == "E1":
                    scenarioname = "variable electricity price"
                elif key[1] == "E2":
                    scenarioname = "flat electricity price"
                barplot_frame = barplot_frame.append({"Configuration": key[0],
                                                      y_axis_title: Differenz,
                                                      "Scenario": scenarioname},
                                                     ignore_index=True)

            sns.barplot(data=barplot_frame,
                        x="Configuration",
                        y=y_axis_title,
                        hue="Scenario",
                        palette="muted")

            ax = plt.gca()
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
            ax.tick_params(axis='x', labelrotation=45)
            plt.tight_layout()
            fig_name = "Aggregated_Barplot " + variable2plot.replace("Year_", "")
            figure.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(fig_name + ".png"), dpi=200,
                           format='PNG')
            plt.show()
            plt.close(figure)

        plot_barplot(config_frame_opt_year, config_frame_ref_year, "Year_E_Grid")
        plot_barplot(config_frame_opt_year, config_frame_ref_year, "OperationCost")
        plot_barplot(config_frame_opt_year, config_frame_ref_year, "Year_E_RoomCooling")
        plot_barplot(config_frame_opt_year, config_frame_ref_year, "Year_E_HeatPump")
        plot_barplot(config_frame_opt_year, config_frame_ref_year, "Year_Q_HeatPump")
        plot_barplot(config_frame_opt_year, config_frame_ref_year, "Year_E_PVSelfUse")

        def get_hourly_aggregated_profile(configuration_IDs, variable):
            # select the aggregation profiles from the hourly results frame
            summary_profile_ref = {}
            summary_profile_opt = {}
            for i, configuration in enumerate(configuration_IDs.keys()):
                print("getting results for configuration: {}".format(configuration))
                environment_id = int(re.sub("[^0-9]", "", configuration[1]))  # remove non numbers to get ID as number
                # Sum_ref = np.zeros(shape=(8760,))
                # Sum_opt = np.zeros(shape=(8760,))
                condition_list = ["ID_Household == '" + str(house_id) for house_id in configuration_IDs[configuration]]
                long_string = ""
                for word in condition_list:
                    long_string += word + "' or "
                long_string = long_string[0:-4]
                condition = " where " + long_string + " and ID_Environment == '{}'".format(environment_id)
                DataFrame_opt = pd.read_sql(
                    'select ' + variable + ' from ' + REG_Table().Res_SystemOperationHour + condition, con=self.Conn)
                Sum_opt = DataFrame_opt.sum()
                DataFrame_ref = pd.read_sql(
                    'select ' + variable + ' from ' + REG_Table().Res_Reference_HeatingCooling + condition,
                    con=self.Conn)
                Sum_ref = DataFrame_ref.sum()
                summary_profile_ref[configuration] = Sum_ref
                summary_profile_opt[configuration] = Sum_opt
            return summary_profile_ref, summary_profile_opt

        # electricity from grid
        profile_ref_E_Grid, profile_opt_E_Grid = get_hourly_aggregated_profile(config_ids_opt, "E_Grid")

        # heating
        # profile_ref_Q_HP, profile_opt_Q_HP = get_hourly_aggregated_profile(config_ids_opt, "Q_HeatPump")
        # profile_ref_E_HP, profile_opt_E_HP = get_hourly_aggregated_profile(config_ids_opt, "E_HeatPump")
        #
        # # cooling
        # profile_ref_Q_Cooling, profile_opt_Q_Cooling = get_hourly_aggregated_profile(config_ids_opt, "Q_RoomCooling")
        # profile_ref_E_Cooling, profile_opt_E_Cooling = get_hourly_aggregated_profile(config_ids_opt, "E_RoomCooling")

        for week in [8, 32]:
            # electricity from grid big subplot
            self.plot_aggregation_comparison_week(profile_opt_E_Grid, profile_ref_E_Grid, "E_Grid",
                                                  number_of_profiles_aggregated, week=week)

            # heating big subplot
            # self.plot_aggregation_comparison_week(profile_opt_Q_HP, profile_ref_Q_HP, "Q_HeatPump",
            #                                       number_of_profiles_aggregated, week=week)
            # self.plot_aggregation_comparison_week(profile_opt_E_HP, profile_ref_E_HP, "E_HeatPump",
            #                                       number_of_profiles_aggregated, week=week)
            #
            # # cooling big subplot
            # self.plot_aggregation_comparison_week(profile_opt_Q_Cooling, profile_ref_Q_Cooling, "Q_Cooling",
            #                                       number_of_profiles_aggregated, week=week)
            # self.plot_aggregation_comparison_week(profile_opt_E_Cooling, profile_ref_E_Cooling, "E_Cooling",
            #                                       number_of_profiles_aggregated, week=week)
            #

    def find_negativ_cost_households(self):
        # households that have negative operation costs:
        # in the reference model
        reference_results_OP_below_zero_env1 = self.ReferenceOperationYear.loc[
            (self.ReferenceOperationYear["OperationCost"] < 0) &
            (self.ReferenceOperationYear["ID_Building"] < 12) &
            (self.ReferenceOperationYear["ID_Environment"] == 1)]
        # in the optimization model
        optimization_results_OP_below_zero_env1 = self.SystemOperationYear.loc[
            (self.SystemOperationYear["OperationCost"] < 0) &
            (self.SystemOperationYear["ID_Building"] < 12) &
            (self.ReferenceOperationYear["ID_Environment"] == 1)]

        reference_results_OP_below_zero_env2 = self.ReferenceOperationYear.loc[
            (self.ReferenceOperationYear["OperationCost"] < 0) &
            (self.ReferenceOperationYear["ID_Building"] < 12) &
            (self.ReferenceOperationYear["ID_Environment"] == 2)]
        # in the optimization model
        optimization_results_OP_below_zero_env2 = self.SystemOperationYear.loc[
            (self.SystemOperationYear["OperationCost"] < 0) &
            (self.SystemOperationYear["ID_Building"] < 12) &
            (self.ReferenceOperationYear["ID_Environment"] == 2)]

        def create_percentage_barplot(title, dataframe):
            analyzed_parameters = {"Household_TankSize": {0: "l", 1500: "l"},
                                   "Household_CoolingAdoption": {0: "", 1: ""},
                                   "Household_PVPower": {0: "kWp", 5: "kWp", 10: "kWp"},
                                   "Household_BatteryCapacity": {0: "Wh", 7000: "Wh"},
                                   "Environment_ElectricityPriceType": {1: "variable", 2: "flat"},
                                   "ID_AgeGroup": {1: "fixed \n temperature", 2: "smart"},
                                   "ID_SpaceHeating": {"Air_HP": "Air HP", "Water_HP": "Ground_HP"}}
            fig = plt.figure()
            fig.suptitle(title)
            ax = plt.gca()

            def create_inplot_text(list_uniques: list, units: dict) -> list:
                """return list of texts for text in plot"""
                text = []
                for element in list_uniques:
                    try:
                        text.append(str(str(int(element)) + str(units[element])))
                    except:
                        text.append(str(str(element) + str(units[element])))
                text_return = [element.replace("1variable", "variable").replace("2flat", "flat").replace(
                    "2smart", "smart").replace("1fixed \n temperature", "fixed \n temperature").replace(
                    "Air_HP", "").replace("Water_HP", "").replace("_HP", " HP") for element in text]
                if text_return == ["0", "1"]:
                    text_return = ["no", "yes"]
                return text_return

            for parameter in analyzed_parameters.keys():
                total_number_ref = len(dataframe)
                unit = analyzed_parameters[parameter]
                x_label = parameter.replace("Household_", "")
                x_label = x_label.replace("Environment_", "")
                x_label = x_label.replace("ID_AgeGroup", "indoor set \n temperature")
                x_label = x_label.replace("ID_SpaceHeating", "HP type")
                # create frames with the
                uniques = np.unique(dataframe[parameter])
                # TODO implement the number of the buildings with HP!
                # count the number of households in each column
                if len(uniques) == 1:
                    text = create_inplot_text(uniques, unit)
                    plt.bar(x_label, 1, color=self.COLOR.red, label=uniques[0])
                    plt.text(x_label, 0.5, text[0], ha="center")
                elif len(uniques) == 2:
                    text = create_inplot_text(uniques, unit)
                    first_bar_ref = len(
                        dataframe.loc[dataframe[parameter] == uniques[0]])
                    percentage_ref = first_bar_ref / total_number_ref

                    plt.bar(x_label, percentage_ref, color=self.COLOR.red, label=uniques[0])
                    plt.bar(x_label, 1 - percentage_ref, bottom=percentage_ref, color=self.COLOR.blue, label=uniques[1])
                    plt.text(x_label, percentage_ref / 2, text[0].replace("Water_HP", ""),
                             ha="center")
                    plt.text(x_label, percentage_ref + (1 - percentage_ref) / 2,
                             text[1], ha="center")
                elif len(uniques) == 3:
                    text = create_inplot_text(uniques, unit)
                    first_bar_ref = len(dataframe.loc[dataframe[parameter] == uniques[0]])
                    second_bar_ref = len(dataframe.loc[dataframe[parameter] == uniques[1]])
                    percentage_ref_1 = first_bar_ref / total_number_ref
                    percentage_ref_2 = second_bar_ref / total_number_ref
                    percentage_ref_3 = 1 - percentage_ref_1 - percentage_ref_2

                    plt.bar(x_label, percentage_ref_1, color=self.COLOR.red, label=uniques[0])
                    plt.bar(x_label, percentage_ref_2, bottom=percentage_ref_1, color=self.COLOR.blue, label=uniques[1])
                    plt.bar(x_label, percentage_ref_3, bottom=percentage_ref_2, color=self.COLOR.green,
                            label=uniques[2])
                    plt.text(x_label, percentage_ref_1 / 2, text[0],
                             ha="center")
                    plt.text(x_label, percentage_ref_1 + percentage_ref_2 / 2,
                             text[1], ha="center")
                    plt.text(x_label, percentage_ref_1 + percentage_ref_2 + percentage_ref_3 / 2,
                             text[2], ha="center")

            plt.xticks(rotation=45)
            plt.grid(axis="y")
            ax.set_axisbelow(True)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            plt.tight_layout()
            plt.savefig(self.figure_path_aggregated / (title.replace("\n", "") + ".png"))
            plt.savefig(self.figure_path_aggregated / (title.replace("\n", "") + ".svg"))
            plt.show()

        create_percentage_barplot("Percentage of household configurations with negative energy costs \n "
                                  "in the reference model", reference_results_OP_below_zero_env1)
        create_percentage_barplot("Percentage of household configurations with negative energy costs \n "
                                  "in the optimization model", optimization_results_OP_below_zero_env1)
        create_percentage_barplot("Percentage of household configurations with negative energy costs \n "
                                  "in both core", pd.concat([optimization_results_OP_below_zero_env1,
                                                               reference_results_OP_below_zero_env1]))

    def violin_plots(self):

        reference_results = self.ReferenceOperationYear
        optimization_results = self.SystemOperationYear
        reference_results.loc[:, "Option"] = "Reference"
        optimization_results.loc[:, "Option"] = "Optimization"
        frame = pd.concat([reference_results, optimization_results], axis=0)
        save_frame = pd.DataFrame(columns=list(frame.columns) + ["percentage difference", "percentage min", "Goal"])

        def create_single_violin_plot(Frame, y_achse, title, *args, **kwargs):
            assert "x" in kwargs and "hue" in kwargs
            if "SFH" in args:
                Frame = Frame.loc[Frame["ID_Building"].isin(np.arange(1, 12))]
                title = title + " SFH"
                figure_name = "ViolinPlot " + title
                figure_path_png = CONS().FiguresPath / Path("Aggregated_Results/SFH") / Path(figure_name + ".png")
                figure_path_svg = CONS().FiguresPath / Path("Aggregated_Results/SFH") / Path(figure_name + ".svg")
            elif "RH" in args:
                Frame = Frame.loc[Frame["ID_Building"].isin(np.arange(12, 23))]
                title = title + " RH"
                figure_name = "ViolinPlot " + title
                figure_path_png = CONS().FiguresPath / Path("Aggregated_Results/RH") / Path(figure_name + ".png")
                figure_path_svg = CONS().FiguresPath / Path("Aggregated_Results/RH") / Path(figure_name + ".svg")
            else:
                figure_name = "ViolinPlot " + title
                figure_path_png = CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".png")
                figure_path_svg = CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".svg")

            fig = plt.figure()
            ax = plt.gca()
            fig.suptitle(title)
            sns.violinplot(x=kwargs["x"], y=y_achse, hue=kwargs["hue"], data=Frame,
                           split=True, inner="stick", palette="muted")
            ax.grid(axis="y")
            if ax.get_ylim()[1] > 999:
                ax.get_yaxis().set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
            ax.yaxis.set_tick_params(labelbottom=True)

            plt.savefig(figure_path_png, dpi=200,
                        format='PNG')
            plt.savefig(figure_path_svg)
            plt.show()

        # for house_type in ["SFH", "RH"]:
        #     create_single_violin_plot(frame, "OperationCost", "Operation Cost for buildings",
        #                               house_type, x="ID_Building", hue="Option")
        #     create_single_violin_plot(frame, "Year_E_Grid", "Grid demand for buildings",
        #                               house_type, x="ID_Building", hue="Option")

        def create_violin_plot_overview(Frame, y_achse, title):
            matplotlib.rc("font", size=14)
            fig, axes = plt.subplots(2, 2, sharey=True, figsize=[12, 10])
            fig.suptitle(title)
            axes = axes.flatten()
            sns.violinplot(x="ID_Building", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}),
                           ax=axes[0], split=True,
                           inner="stick", palette="muted")
            axes[0].set_xlabel("Building ID")
            axes[0].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_PVPower", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}),
                           ax=axes[1], split=True,
                           inner="stick", palette="muted")
            axes[1].set_xlabel("PV Power in kWp")
            axes[1].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_TankSize", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}),
                           ax=axes[2], split=True,
                           inner="stick", palette="muted")
            axes[2].set_xlabel("TankSize in l")
            axes[2].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_BatteryCapacity", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}),
                           ax=axes[3],
                           split=True, inner="stick", palette="muted")
            axes[3].set_xlabel("Battery Capacity in Wh")
            axes[3].set_ylabel(self.determine_ylabel(y_achse))

            # sns.violinplot(x="ID_AgeGroup", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}), ax=axes[4],
            #                split=True, inner="stick", palette="muted")
            # axes[4].set_xlabel("indoor set \n temperature")
            # axes[4].set_ylabel(self.determine_ylabel(y_achse))
            #
            # sns.violinplot(x="ID_SpaceHeating", y=y_achse, hue="", data=Frame.rename(columns={"Option": ""}), ax=axes[5],
            #                split=True, inner="stick", palette="muted")
            # axes[5].set_xlabel("HP type")
            # axes[5].set_ylabel(self.determine_ylabel(y_achse))
            # axes[5].set_xticklabels([str(tick.get_text()).replace("Air_HP", "Air HP").replace("Water_HP", "Ground HP")
            #                          for tick in axes[5].get_xticklabels()])

            for i in range(axes.shape[0]):
                axes[i].grid(axis="y")
                if axes[i].get_ylim()[1] > 999:
                    axes[i].get_yaxis().set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))

                axes[i].yaxis.set_tick_params(labelbottom=True)

            figure_name = f"ViolinPlot Overview {y_achse} with " + title
            plt.tight_layout()
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".png"), dpi=200,
                        format='PNG')
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".svg"))
            plt.show()

        def calculate_most_optim_impact(df: pd.DataFrame, variable: str) -> None:
            """calculated difference between reference and optimization with percentage impact and puts
            it in excel writer"""
            # sort frame:
            df = df.sort_values(["ID_Household", "ID_Environment"]).reset_index(drop=True)

            ref = df.loc[(df.loc[:, "Option"] == "Reference")].reset_index(drop=True)
            opt = df.loc[(df.loc[:, "Option"] == "Optimization")].reset_index(drop=True)
            difference = opt[variable] - ref[variable]
            difference_percentage = difference / ref[variable]
            max_percentage = difference_percentage.max()
            index_max_percentage = difference_percentage.idxmax()
            min_percentage = difference_percentage.min()
            index_min_percentage = difference_percentage.idxmin()

            house_max = ref.iloc[index_max_percentage]
            house_min = ref.iloc[index_min_percentage]
            house_max.loc["percentage max"] = max_percentage
            house_min.loc["percentage min"] = min_percentage
            house_max.loc["Goal"] = variable
            house_min.loc["Goal"] = variable
            return pd.concat([pd.DataFrame(house_max).T, pd.DataFrame(house_min).T])

        frame = frame.loc[frame["ID_Building"] < 12]

        for key, value in {1: "variable", 2: "flat"}.items():
            price_type_frame = frame.loc[frame["Environment_ElectricityPriceType"] == key]
            # create excel writer:
            highest_impact_writer = pd.ExcelWriter(self.figure_path_aggregated / Path("highest_impact.xlsx"),
                                                   engine="xlsxwriter")
            # calculate the maximum cost savings, reference - optimization
            save_frame = save_frame.append(calculate_most_optim_impact(price_type_frame, "OperationCost"))
            save_frame = save_frame.append(calculate_most_optim_impact(price_type_frame, "Year_E_Grid"))

            create_violin_plot_overview(price_type_frame, "OperationCost",
                                        f"{value} price")
            create_violin_plot_overview(price_type_frame, "Year_E_Load",
                                        f"{value} price")
            create_violin_plot_overview(price_type_frame, "Year_PVSelfSufficiencyRate",
                                        f"{value} price")
            create_violin_plot_overview(price_type_frame, "Year_E_Grid",
                                        f"{value} price")
            create_violin_plot_overview(price_type_frame, "Year_E_PV2Grid",
                                        f"{value} price")
            price_type_frame.loc[:, "Year_PVSelfConsumptionRate"] = pd.to_numeric(
                price_type_frame.loc[:, "Year_PVSelfConsumptionRate"].replace("nan", 0))


        # create excel writer:
        highest_impact_writer = pd.ExcelWriter(self.figure_path_aggregated / Path("highest_impact.xlsx"),
                                               engine="xlsxwriter")
        # # calculate the maximum cost savings, reference - optimization
        # save_frame = save_frame.append(calculate_most_optim_impact(frame, "OperationCost"))
        # save_frame = save_frame.append(calculate_most_optim_impact(frame, "Year_E_Grid"))
        #
        # create_violin_plot_overview(frame, "OperationCost",
        #                             f"")
        # create_violin_plot_overview(frame, "Year_E_Load",
        #                             f"")
        # # create_violin_plot_overview(price_type_frame, "Year_PVSelfSufficiencyRate",
        # #                             f"{value} price")
        # create_violin_plot_overview(frame, "Year_E_Grid",
        #                             f"")
        # create_violin_plot_overview(frame, "Year_E_PV2Grid",
        #                             f"")

        save_frame.to_excel(self.figure_path_aggregated / Path("highest_impact.xlsx"), engine="xlsxwriter")
        # ref_list = pd.DataFrame(columns=optimization_results.columns)
        # opt_list = pd.DataFrame(columns=optimization_results.columns)
        # for index in range(optimization_results.shape[0]):
        #     if reference_results.loc[index, "OperationCost"] < optimization_results.loc[index, "OperationCost"]:
        #         print("Household ID: " + str(self.SystemOperationYear.loc[index, "ID_Household"]))
        #         print("Battery: " + str(reference_results.loc[index, "Household_BatteryCapacity"]))
        #         print("PV Peak: " + str(reference_results.loc[index, "Household_PVPower"]))
        #         print("Tank: " + str(reference_results.loc[index, "Household_TankSize"]))
        #
        #         ref_list = ref_list.append(reference_results.iloc[index, :], ignore_index=True)
        #         opt_list = opt_list.append(optimization_results.iloc[index, :], ignore_index=True)
        #
        # frame_for_violin = pd.concat([ref_list, opt_list], axis=0)

        # create_violin_plot(frame_for_violin, "OperationCost", "Costs of special buildings")
        # create_violin_plot(frame_for_violin, "Year_E_Load", "electricity consumption special buildings")

        # house12_ref = self.ReferenceOperationHour.loc[self.ReferenceOperationHour["ID_Household"] == 12, :]
        # house12_opt = self.SystemOperationHour.loc[self.SystemOperationHour["ID_Household"] == 12, :]
        #
        # solar_diff = house12_ref.loc[:, "Q_SolarGain"] / 1000 - house12_opt.loc[:, "Q_SolarGain"]
        #
        # heating_diff = house12_ref.loc[:, "Q_HeatPump"] - house12_opt.loc[:, "Q_HeatPump"]
        #
        # cooling_diff = house12_ref.loc[:, "Q_RoomCooling"] - house12_opt.loc[:, "Q_RoomCooling"]
        #
        # number_of_buildings = self.get_total_number_of_buildings()

        pass

    def plot_aggregation_comparison_week(self, optim_variable, ref_variable, variable_name, number_of_profiles,
                                         **kwargs):
        HourStart = 1
        HourEnd = 8760
        if "horizon" in kwargs:
            HourStart = kwargs["horizon"]
            HourEnd = kwargs["horizon"]
        else:
            pass
        if "week" in kwargs:
            TimeStructure_select = self.TimeStructure.loc[self.TimeStructure["ID_Week"] == kwargs["week"]]
            HourStart = TimeStructure_select.iloc[0]["ID_Hour"]
            HourEnd = TimeStructure_select.iloc[-1]["ID_Hour"] + 1
        else:
            pass

        number_of_subplots = len(optim_variable)
        figure, axes = plt.subplots(number_of_subplots, 1, figsize=(20, 4 * number_of_subplots), dpi=200, frameon=False)
        # ax_1 = figure.add_axes([0.1, 0.1, 0.8, 0.75])
        x_values = np.arange(HourStart, HourEnd)
        alpha_value = 0.8
        linewidth_value = 2

        for index, (key, values) in enumerate(optim_variable.items()):
            axes[index].plot(x_values, values[HourStart:HourEnd] / number_of_profiles,
                             alpha=alpha_value,
                             linewidth=linewidth_value,
                             label="Optimization")

        for index, (key, values) in enumerate(ref_variable.items()):
            axes[index].plot(x_values, values[HourStart:HourEnd] / number_of_profiles,
                             alpha=alpha_value,
                             linewidth=linewidth_value,
                             label="Reference")
            tick_position = [HourStart + 11.5] + [HourStart + 11.5 + 24 * day for day in range(1, 7)]
            x_ticks_label = ["Mon", "Tues", "Wed", "Thur", "Fri", "Sat", "Sun"]
            axes[index].set_xticks(tick_position)
            axes[index].set_xticklabels(x_ticks_label)
            axes[index].tick_params(axis="both", labelsize=20)

            for tick in axes[index].yaxis.get_major_ticks():
                tick.label2.set_fontsize(20)

            if key[1] == "E1":
                price_type = "flat"
            elif key[1] == "E2":
                price_type = "variable"

            split_title = re.split("(\d+)", key[0])
            title = split_title[0] + ": " + split_title[1] + " kWp, " + split_title[2] + "attery: " + split_title[3] + \
                    " kWh, " + split_title[4] + "ank size: " + split_title[5] + " l " + "price: " + price_type
            axes[index].set_title(title, fontsize=20)
            axes[index].grid(which="both", axis="y")
            axes[index].legend()
        figure.suptitle(variable_name + " all variations", fontsize=30)
        plt.grid()
        plt.tight_layout()
        if "week" in kwargs:
            figure_name = "BigSubplots_" + variable_name + " week " + str(kwargs["week"])
        else:
            figure_name = "BigSubplots_" + variable_name
        plt.savefig(CONS().FiguresPath + "\\Aggregated_Results\\" + figure_name + ".png")
        plt.savefig(CONS().FiguresPath + "\\Aggregated_Results\\" + figure_name + ".svg")
        # plt.show()
        plt.close(figure)

    def plot_energy_consumption(self) -> None:
        """visualization of the enrgy consumption of the households in dependence of the different parameters
        that influence the optimization"""

        def remaining_scenario_variables(fixed_parameters: list) -> dict:
            """returns the remaining variables that are not fixed as a dict"""
            return_dict = self.scenario_variables.copy()
            for key in fixed_parameters:
                del return_dict[key]
            return return_dict

        # create boxplot of specific household categories
        def create_subresults_frame():
            # 1) difference between electricity prices (split up dataframes)
            res_optimization_year_elec1 = self.SystemOperationYear.loc[
                (self.SystemOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 1)]

            res_optimization_year_elec2 = self.SystemOperationYear.loc[
                (self.SystemOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 2)]

            res_reference_year_elec1 = self.ReferenceOperationYear.loc[
                (self.ReferenceOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 1)]

            res_reference_year_elec2 = self.ReferenceOperationYear.loc[
                (self.ReferenceOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 2)]

            # take out RH:
            res_optimization_year_elec1 = res_optimization_year_elec1[res_optimization_year_elec1["ID_Building"] < 12]
            res_optimization_year_elec2 = res_optimization_year_elec2[res_optimization_year_elec2["ID_Building"] < 12]
            res_reference_year_elec1 = res_reference_year_elec1[res_reference_year_elec1["ID_Building"] < 12]
            res_reference_year_elec2 = res_reference_year_elec2[res_reference_year_elec2["ID_Building"] < 12]

            return res_optimization_year_elec1, res_optimization_year_elec2, res_reference_year_elec1, res_reference_year_elec2

        def create_boxplot_with_scatter_plot(data, title):
            parameters_to_analyze = {"Household_TankSize": {0: "l", 1500: "l"},
                                     "Household_CoolingAdoption": {0: "", 1: ""},
                                     "Household_PVPower": {0: "kWp", 5: "kWp", 10: "kWp"},
                                     "Household_BatteryCapacity": {0: "Wh", 7000: "Wh"},
                                     "ID_SpaceHeating": {"Air_HP": "", "Water_HP": ""},
                                     "ID_AgeGroup": {1.: "1", 2.: "2"}}
            df_mean_resutls = pd.DataFrame(parameters_to_analyze)

            fig = plt.figure()
            plt.suptitle(title)
            ax = plt.gca()

            xticks = np.arange(6)
            plt.xticks(xticks)
            colors = [self.COLOR.dark_blue, self.COLOR.dark_red, self.COLOR.blue, self.COLOR.brown, self.COLOR.grey,
                      self.COLOR.green, self.COLOR.dark_green, self.COLOR.dark_grey, self.COLOR.yellow, self.COLOR.red,
                      self.COLOR.purple, self.COLOR.dark_blue, self.COLOR.dark_red]
            patches = []
            i = 0  # to iterate through color list
            tick = 0
            for parameter, options in parameters_to_analyze.items():
                space_tick = 0
                for option_name, option_unit in options.items():
                    if len(options) == 2:
                        spacing = [-0.2, 0.2]
                    elif len(options) == 3:
                        spacing = [-0.2, 0, 0.2]
                    try:
                        e_grid = data.loc[data.loc[:, parameter] == option_name.astype(float)].Year_E_Grid.to_numpy()
                    except:
                        e_grid = data.loc[data.loc[:, parameter] == option_name].Year_E_Grid.to_numpy()

                    plt.scatter(np.random.uniform(low=xticks[tick] - 0.1 + spacing[space_tick],
                                                  high=xticks[tick] + 0.1 + spacing[space_tick],
                                                  size=(len(e_grid),)), e_grid,
                                color=colors[i], alpha=0.1, s=10)

                    plt.boxplot(e_grid, meanline=True, positions=[xticks[tick] + spacing[space_tick]])
                    plt.text(xticks[tick] + spacing[space_tick], e_grid.mean(), f"{round(e_grid.mean())}", ha="center")
                    df_mean_resutls.loc[option_name, parameter] = e_grid.mean()
                    patches.append(matplotlib.patches.Patch(color=colors[i],
                                                            label=parameter.replace("Household_", "").
                                                            replace("ID_", "") +
                                                                  " " + str(option_name).replace("Air_HP", "Air HP").
                                                            replace("Water_HP", "Ground HP") + str(option_unit)))
                    i += 1
                    space_tick += 1
                tick += 1
            xticks = [param.replace("Household_", "").replace("ID_", "") for param in parameters_to_analyze.keys()]
            plt.xticks(np.arange(6), xticks, rotation=45)
            plt.legend(handles=patches, bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.ylabel("Grid demand (kWh)")
            ax.set_ylim(1_000, 10_000)
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))

            fig_name = title.replace("\n", "")
            fig.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(fig_name + ".png"), dpi=200,
                        format='PNG', bbox_inches="tight")
            fig.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(fig_name + ".svg"))

            plt.show()
            df_mean_resutls.to_excel(CONS().FiguresPath / Path(f"Aggregated_Results//{fig_name}.xlsx"))

        fixed_variables = ["Environment_ElectricityPriceType"]
        remaining_variables = remaining_scenario_variables(fixed_variables)
        fixed_variables_dict = remaining_scenario_variables(list(remaining_variables.keys()))

        optimization_var, optimization_flat, reference_elec1, reference_elec2 = create_subresults_frame()

        # create_boxplot_with_scatter_plot_total(reference_elec1, optimization_flat, optimization_var)

        create_boxplot_with_scatter_plot(optimization_var,
                                         "Impact of scenario parameters on grid demand \n with variable price, optimization model")
        create_boxplot_with_scatter_plot(optimization_flat,
                                         "Impact of scenario parameters on grid demand \n with flat price, optimization model")

        create_boxplot_with_scatter_plot(reference_elec1,
                                         "Impact of scenario parameters on grid demand \n with variable price, reference model")
        create_boxplot_with_scatter_plot(reference_elec2,
                                         "Impact of scenario parameters on grid demand \n with flat price, reference model")

    def plot_households_without_storages(self):
        """plot households that have no storage"""

        def check_why_elec_increases_flat_price():
            # get profiles from database
            environment = 2
            ref_profiles = DB().read_DataFrame("Res_Reference_HeatingCooling_Year", conn=self.Conn,
                                               ID_Environment=environment,
                                               ID_Spaceheating="Water_HP",
                                               Household_BatteryCapacity=0,
                                               Household_PVPower=0,
                                               Household_TankSize=0,
                                               Household_CoolingAdoption=1,
                                               ID_AgeGroup=1)
            ref_profiles = ref_profiles.loc[ref_profiles.loc[:, "ID_Building"] < 12]
            household_IDs_ref = ref_profiles.ID_Household

            # show price difference:
            for id in household_IDs_ref:
                costs_ref = DB().read_DataFrame("Res_Reference_HeatingCooling_Year", conn=self.Conn,
                                                ID_Household=id, ID_Environment=environment).OperationCost.to_numpy()
                costs_opt = DB().read_DataFrame("Res_SystemOperationYear", conn=self.Conn,
                                                ID_Household=id, ID_Environment=environment).OperationCost.to_numpy()
                print(f"cost reduction: {costs_opt - costs_ref} for id {id}")

            # get hourly data:
            # for id in household_IDs_ref:
            id = household_IDs_ref[2]
            axis = np.arange(0, 8760)
            number_of_subplots = 7
            fig, axes = plt.subplots(number_of_subplots, 1, figsize=(20, 4 * number_of_subplots))
            # ax = plt.gca()
            ref_hour = DB().read_DataFrame("Res_Reference_HeatingCooling", conn=self.Conn,
                                           ID_Household=id, ID_Environment=2)
            opt_hour = DB().read_DataFrame("Res_SystemOperationHour", conn=self.Conn,
                                           ID_Household=id, ID_Environment=2)
            axes[0].plot(axis, ref_hour.E_Load.to_numpy()[axis], color=CONS().green, linewidth=1)
            axes[0].plot(axis, opt_hour.E_Load.to_numpy()[axis], color=CONS().red, linewidth=1)
            axes[0].set_ylabel("E_Load")
            # ax.plot(np.arange(8760), opt_hour.E_Grid2Load.to_numpy()[:8760]-ref_hour.E_Grid2Load.to_numpy()[1800:2000], color=CONS().red, linewidth=1)

            axes[1].plot(axis, opt_hour.Q_HeatPump.to_numpy()[axis], color=CONS().red)
            axes[1].plot(axis, ref_hour.Q_HeatPump.to_numpy()[axis], color=CONS().green)
            axes[1].set_ylabel("Q Heat Pump")

            axes[2].plot(axis, opt_hour.BuildingMassTemperature.to_numpy()[axis], color=CONS().red)
            axes[2].plot(axis, ref_hour.BuildingMassTemperature.to_numpy()[axis], color=CONS().green)
            axes[2].set_ylabel("T mt")

            axes[3].plot(axis, ref_hour.E_Grid.to_numpy()[axis], color=CONS().green)
            axes[3].plot(axis, opt_hour.E_Grid.to_numpy()[axis], color=CONS().red)
            axes[3].set_ylabel("E_Grid")

            axes[4].plot(axis, ref_hour.E_Battery2Load.to_numpy()[axis], color=CONS().green)
            axes[4].plot(axis, opt_hour.E_Battery2Load.to_numpy()[axis], color=CONS().red)
            axes[4].set_ylabel("E_Battery2Load")

            axes[5].plot(axis, ref_hour.HeatPumpPerformanceFactor.to_numpy()[axis], color=CONS().green)
            axes[5].plot(axis, opt_hour.HeatPumpPerformanceFactor.to_numpy()[axis], color=CONS().red)
            axes[5].set_ylabel("HeatPumpPerformanceFactor")

            axes[6].plot(axis, ref_hour.OutsideTemperature.to_numpy()[axis], color=CONS().green)
            axes[6].plot(axis, opt_hour.OutsideTemperature.to_numpy()[axis], color=CONS().red)
            axes[6].set_ylabel("OutsideTemperature")

            plt.savefig(self.figure_path_aggregated / Path(f"checking_for_mistakes_{id}.png"))
            plt.show()

        check_why_elec_increases_flat_price()

    def calculate_key_numbers(self) -> None:
        """this function calculates important values for the projects and is here to play around"""
        # sum of electricity from grid
        elec_grid_opt = DB().read_DataFrame(REG_Table().Res_SystemOperationYear, self.Conn,
                                            "Year_E_Grid")

        elec_grid_ref = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling_Year, self.Conn,
                                            "Year_E_Grid")
        percentage_increase_opt2ref = (elec_grid_opt.sum() - elec_grid_ref.sum()) / elec_grid_ref.sum()
        print(
            f"percentage increase in grid consumption thorugh optimization on all results: {percentage_increase_opt2ref * 100}")

        # maximum reached self sufficiency in the optimization
        self_sufficiency_opt = DB().read_DataFrame(REG_Table().Res_SystemOperationYear, self.Conn,
                                                   "Year_PVSelfSufficiencyRate")
        max_self_sufficiency_opt = self_sufficiency_opt.max()
        mean_self_sufficiency_opt = self_sufficiency_opt.mean()
        median_self_sufficiency_opt = self_sufficiency_opt.median()
        # maximum reached self sufficiency in the reference
        self_sufficiency_ref = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling_Year, self.Conn,
                                                   "Year_PVSelfSufficiencyRate")
        max_self_sufficiency_ref = self_sufficiency_ref.max()
        mean_self_sufficiency_ref = self_sufficiency_ref.mean()
        median_self_sufficiency_ref = self_sufficiency_ref.median()

        outside_temperature = DB().read_DataFrame(REG_Table().Sce_Weather_Temperature, self.Conn).Temperature.to_numpy()
        mean_temp = outside_temperature.mean()
        set_temperatures = DB().read_DataFrame(REG_Table().Gen_Sce_TargetTemperature, self.Conn)
        heat_temp_normal = set_temperatures.HeatingTargetTemperatureYoungNightReduction.to_numpy()
        cool_temp = set_temperatures.CoolingTargetTemperatureYoung
        heat_temp_smart = set_temperatures.HeatingTargetTemperatureSmartHome
        cool_temp_smart = set_temperatures.CoolingTargetTemperatureSmartHome
        matplotlib.rc("font", size=12)
        fig = plt.figure()
        ax = plt.gca()
        x_axis = np.arange(8760)
        ax.plot(x_axis, outside_temperature, label="outside temperature", color=CONS().black, linewidth=0.2)
        ax.plot(x_axis, heat_temp_normal, label="set temperature for heating 1", color=CONS().dark_red, linewidth=0.3)
        ax.plot(x_axis, cool_temp, label="set temperature for cooling 1", color=CONS().dark_blue, linewidth=1.5)
        ax.plot(x_axis, heat_temp_smart, label="set temperature for heating 2", color=CONS().orange)
        ax.plot(x_axis, cool_temp_smart, label="set temperature for cooling 2", color=CONS().blue)
        plt.xlabel("hours")
        plt.ylabel("temperature in °C")
        plt.grid()
        plt.legend()
        plt.savefig(self.figure_path_aggregated / Path("temperature.svg"))
        plt.show()

        elec_price = DB().read_DataFrame(REG_Table().Gen_Sce_ElectricityProfile, self.Conn)
        var_price = elec_price.loc[elec_price.loc[:, "ID_PriceType"] == 1].ElectricityPrice.to_numpy()
        mean_price = elec_price.loc[elec_price.loc[:, "ID_PriceType"] == 2].ElectricityPrice.to_numpy()
        fit = np.full((8760,), 7.67)
        matplotlib.rc("font", size=12)
        fig = plt.figure()
        ax = plt.gca()
        x_axis = np.arange(8760)
        ax.plot(x_axis, var_price, label="variable electricity price", color=CONS().dark_blue, linewidth=0.2)
        ax.plot(x_axis, mean_price, label="flat electricity price", color=CONS().dark_red)
        ax.plot(x_axis, fit, label="feed in tarif", color=CONS().green)
        plt.xlabel("hours")
        plt.ylabel("electricity price in c/kWh")
        plt.grid()
        plt.legend()
        plt.savefig(self.figure_path_aggregated / Path("electricity_price.svg"))
        plt.show()

        pass


class UpScalingToBuildingStock:

    def __init__(self):
        self.Conn = DB().create_Connection(CONS().RootDB)
        self.SystemOperationYear = DB().read_DataFrame(REG_Table().Res_SystemOperationYear, self.Conn)
        self.ReferenceOperationYear = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling_Year, self.Conn)
        self.figure_path = CONS().FiguresPath
        path2buildingsnumber = CONS().ProjectPath / Path("_Philipp/inputdata/AUT") / Path(
            "040_aut__2__BASE__1_zzz_short_bc_seg__b_building_segment_sh.csv")
        number_of_buildings_frame = pd.read_csv(path2buildingsnumber, sep=";", encoding="ISO-8859-1", header=0)
        self.number_of_buildings_frame = number_of_buildings_frame.iloc[4:, :]
        self.path_2_output = CONS().ProjectPath / Path("_Philipp/outputdata")
        # percentage of HP buildings that are going to be optimized

        # 30% of Households with PV have battery storage!
        self.percentage_battery_storage = 0.02
        # 50% of Households with PV have a thermal storage:
        self.percentage_tank_storage = 0.6
        # 10% of Households with PV have Battery and thermal storage:
        self.percentage_tank_battery = self.percentage_battery_storage * self.percentage_tank_storage
        # percentage of households that have only PV and no other appliance:
        self.percentage_PV_nothing_else = 1 - self.percentage_tank_battery - self.percentage_tank_storage - self.percentage_battery_storage
        self.environment = [1, 2]
        self.optimized_percentage = 1
        # 50% of households with HP and without PV adopt a thermal storage:
        self.percentage_tank_no_PV = 0.6
        self.configurations = {"PV0B0T0": {"PVPower": 0, "TankSize": 0, "BatteryCapacity": 0},
                               "PV0B0T1500": {"PVPower": 0, "TankSize": 1500, "BatteryCapacity": 0},
                               "PV5B0T0": {"PVPower": 5, "TankSize": 0, "BatteryCapacity": 0},
                               "PV5B7T0": {"PVPower": 5, "TankSize": 0, "BatteryCapacity": 7000},
                               "PV5B0T1500": {"PVPower": 5, "TankSize": 1500, "BatteryCapacity": 0},
                               "PV5B7T1500": {"PVPower": 5, "TankSize": 1500, "BatteryCapacity": 7000},
                               "PV10B0T0": {"PVPower": 10, "TankSize": 0, "BatteryCapacity": 0},
                               "PV10B7T0": {"PVPower": 10, "TankSize": 0, "BatteryCapacity": 7000},
                               "PV10B0T1500": {"PVPower": 10, "TankSize": 1500, "BatteryCapacity": 0},
                               "PV10B7T1500": {"PVPower": 10, "TankSize": 1500, "BatteryCapacity": 7000}
                               }

    def read_h5(self):
        """reads hdf5 file and returns the dataframe of building segments"""
        hdf_filename = "001_buildings.hdf5"
        hdf_path = CONS().ProjectPath / Path("_Philipp/inputdata/AUT/" + hdf_filename)

        hdf5_f = h5py.File(hdf_path, 'r')

        # get data elements stored in hdf5 container
        item_names_hdf5 = hdf5_f.items()
        # print("items stored in data container:")
        # print(item_names_hdf5)

        # Extract simulation periods stored in data container
        # year_list_hdf5 = []
        # for curr_item_name in item_names_hdf5:
        #     if curr_item_name[0][:3] == "BC_":
        #         year_list_hdf5.append(int(curr_item_name[0][-4:]))

        # print("Years stored in data container: %s" % str(year_list_hdf5))
        # for yr in year_list_hdf5:
        yr = 2017  # TODO achtung bisschen fake
        # print(f"Current yr: {yr}")
        key_bc = "BC_%i" % yr  # Building household file
        key_bssh = "BSSH_%i" % yr  # building segment (space heating) file
        try:
            bc = hdf5_f[key_bc][()]  # building household file as Numpy recarray
            bssh = hdf5_f[key_bssh][()]  # building segment file as Numpy recarray
        except:
            pass
            # continue

        bc_index = bc["index"]  # building household index of elements in bc dataset
        bc_index_bssh = bssh["building_classes_index"]  # building household index of elements in bssh dataset

        # if working with pandas dataframe is preferred:
        # convert recarray to pandas dataframe
        df_bc = pd.DataFrame(bc)  # This is pandas dataframe containing the building household data
        del bc  # remove bc to save RAM
        df_bssh = pd.DataFrame(bssh)  # This is pandas dataframe containing the building segment data
        del bssh

        hdf5_f.close()
        return df_bssh

    def get_total_number_of_building(self, building_index: int) -> (float, float):
        """takes the invert building index and returns the number of buildings
        with heat pumps and total number of buildings."""
        number_of_buildings_frame = self.read_h5()
        total_number_of_buildings = number_of_buildings_frame.loc[
                                    pd.to_numeric(
                                        number_of_buildings_frame.loc[:, "building_classes_index"]) == building_index,
                                    :
                                    ].number_of_buildings.sum()

        number_of_buildings_with_HP = number_of_buildings_frame.loc[
            (pd.to_numeric(number_of_buildings_frame.loc[:, "building_classes_index"]) == building_index) &
            ((number_of_buildings_frame.loc[:, "heat_supply_system_index"] == 42) |
             (number_of_buildings_frame.loc[:, "heat_supply_system_index"] == 43))
            ].number_of_buildings.sum()

        # if from csv:
        # total_number_of_buildings = self.number_of_buildings_frame.loc[
        #                             pd.to_numeric(
        #                                 self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index,
        #                             :
        #                             ].number_of_buildings.sum()
        #
        # number_of_buildings_with_HP = self.number_of_buildings_frame.loc[
        #     (pd.to_numeric(self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index) &
        #     ((self.number_of_buildings_frame.loc[:,
        #       "heatingsystem"] == "geothermal_central_heatpump(air/water)_Heat pump air/water") |
        #      (self.number_of_buildings_frame.loc[:,
        #       "heatingsystem"] == "geothermal_central_heatpump(water/water)_Heat pump brine/water shallow"))
        #     ].number_of_buildings.sum()
        return total_number_of_buildings, number_of_buildings_with_HP

    def visualize_total_number_of_buildings(self) -> dict:
        """creates plot that shows how many buildings exist and how many buildings have heat pumps
        in each building category"""
        buildings_frame = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.Conn)
        invert_index = pd.to_numeric(buildings_frame.index_invert)[:11]  # only SFH
        normal_index = pd.to_numeric(buildings_frame["index"]).to_numpy()
        matplotlib.rc("font", size=14)
        fig = plt.figure(figsize=(9, 6))
        ax = plt.gca()
        all_buildings = 0
        all_HP_buildings = {}
        for i, index in enumerate(invert_index):
            total_number, HP_number = self.get_total_number_of_building(index)
            all_buildings += total_number
            all_HP_buildings[i] = HP_number
            ax.bar(i + 1, total_number, color=CONS().dark_blue)
            ax.bar(i + 1, HP_number, color=CONS().green)
        print(f"total number of buildings considered in this study: {all_buildings}")
        print(f"total number of HP buildings considered in this study: {sum(all_HP_buildings.values())}")
        plt.xlabel("Building ID")
        plt.ylabel("number of buildings")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        # create legend
        blue_patch = matplotlib.patches.Patch(color=CONS().dark_blue, label='number of buildings \n without heat pump')
        gree_path = matplotlib.patches.Patch(color=CONS().green, label='number of buildings \n with heat pump')
        plt.legend(handles=[blue_patch, gree_path])
        plt.grid(axis="y")
        ax.set_axisbelow(True)
        plt.tight_layout()
        # save figure
        figure_name = "Number of buildings"
        figure_path_png = CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".png")
        figure_path_svg = CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".svg")
        plt.savefig(figure_path_png, dpi=200,
                    format='PNG')
        plt.savefig(figure_path_svg)
        plt.show()
        return all_HP_buildings

    def visualize_number_of_buildings_scenario(self, scenario) -> None:
        """shows the number of buildings with different technological apliances for each scenario"""
        fig = plt.figure(figsize=(9, 6))
        # fig.suptitle("number of different buildings in the stock")
        ax = plt.gca()
        cmap = plt.get_cmap("tab10")
        xticks = np.arange(10)
        plt.xticks(xticks, labels=list(scenario.keys()))
        width = 0.9
        plt.grid(axis="y", which="minor")
        # calculate the grid electricity consumption for the different buildings
        for i, (key, number) in enumerate(scenario.items()):
            ax.bar(xticks[i], sum(number.values()), width=width,
                   edgecolor="black", color=CONS().dark_green)
            ax.text(xticks[i], sum(number.values()), f"{round(sum(number.values()))}",
                    ha="center")

        ax.set_xticklabels(list(scenario.keys()), minor=True)
        plt.xticks(rotation=45)
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        ax.set_yscale('log')
        plt.ylabel("Number of buildings ")
        ax.set_axisbelow(True)
        plt.xlabel("configurations")
        plt.tight_layout()
        plt.savefig(self.figure_path / Path("Aggregated_Results//number_of_buildings_scenarios.png"))
        plt.savefig(self.figure_path / Path("Aggregated_Results//number_of_buildings_scenarios.svg"))
        plt.show()

    def calculate_percentage_of_hp_buildings(self):
        """calculates the total number of all buildings IDs with reference to ID"""
        buildings_frame = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.Conn)
        invert_index = pd.to_numeric(buildings_frame.index_invert)[:11]  # only SFH
        total_sum_HP_sfh = 0
        HP_buildings_dict = {}
        for i, index in enumerate(invert_index):
            total_number, HP_number = self.get_total_number_of_building(index)
            total_sum_HP_sfh += HP_number
            HP_buildings_dict[i] = HP_number
        HP_buildings_dict_percentage = {index: number / total_sum_HP_sfh for index, number in HP_buildings_dict.items()}
        return HP_buildings_dict_percentage, HP_buildings_dict


    def select_grid_electricity_consumption(self, buildings_dict: dict, PVPower, TankSize, BatteryCapacity,
                                            environment, percentag_optimized_buildings: float, variable: str) -> (
            dict, dict):
        """returns the annual grid electricity consumption for the households in the dict for reference and
        optimization. NO COOLING, fixed indoor temp, 20% ground source HP"""
        cooling_adoption = 0
        id_agegroup = 1
        results_ref = self.ReferenceOperationYear
        results_opt = self.SystemOperationYear
        results_ref.loc[:, "Year_PVSelfConsumptionRate"] = pd.to_numeric(
            results_ref.loc[:, "Year_PVSelfConsumptionRate"].replace("nan", 0))
        results_opt.loc[:, "Year_PVSelfConsumptionRate"] = pd.to_numeric(
            results_opt.loc[:, "Year_PVSelfConsumptionRate"].replace("nan", 0))

        def select_results(frame):
            return frame.loc[
                (frame.loc[:, "Household_PVPower"] == PVPower) &
                (frame.loc[:, "Household_TankSize"] == TankSize) &
                (frame.loc[:, "Household_BatteryCapacity"] == BatteryCapacity) &
                (frame.loc[:, "ID_Environment"] == environment) &
                (frame.loc[:, "Household_CoolingAdoption"] == cooling_adoption) &
                (frame.loc[:, "ID_AgeGroup"] == id_agegroup)]

        opt = select_results(results_opt)
        ref = select_results(results_ref)

        # multiply grid electricity with number of buildings
        def calculate_total_elec(frame, variable) -> dict:
            if variable == "Year_PVSelfConsumptionRate":
                return {i: (
                        frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                  (frame.loc[:, "ID_SpaceHeating"] == "Air_HP")][
                            variable].to_numpy() * 0.75 +  # 75% air HP
                        frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                  (frame.loc[:, "ID_SpaceHeating"] == "Water_HP")][variable].to_numpy() * 0.25
                # 25% ground HP
                ) for i, nr in buildings_dict.items()}
            else:
                return {i: (
                                   frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                             (frame.loc[:, "ID_SpaceHeating"] == "Air_HP")][
                                       variable].to_numpy() * 0.75 +  # 75% air HP
                                   frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                             (frame.loc[:, "ID_SpaceHeating"] == "Water_HP")][
                                       variable].to_numpy() * 0.25  # 25% ground HP
                           ) * round(nr * percentag_optimized_buildings) for i, nr in buildings_dict.items()}

        opt_elec_grid = calculate_total_elec(opt, variable)
        ref_elec_grid = calculate_total_elec(ref, variable)
        return opt_elec_grid, ref_elec_grid

    def calculate_scenario_numbers(self, percentage_5kW, percentage_10kW, HP_buildings):
        """calculate all nessecary numbers for each scenario"""
        # calculate the number of hp buildings that have PV for certain scenario
        total_number_buildings_5kWp = {key: percentage_5kW * number for key, number in HP_buildings.items()}
        total_number_buildings_10kWp = {key: percentage_10kW * number for key, number in HP_buildings.items()}
        # calculate number of buildings that have no PV
        total_number_buildings_NO_PV = {
            key: number - total_number_buildings_5kWp[key] - total_number_buildings_10kWp[key]
            for key, number in HP_buildings.items()}
        # number of buildings wihtout PV but with thermal storage:
        number_buildings_tank_NO_PV = {key: number * self.percentage_tank_storage
                                       for key, number in total_number_buildings_NO_PV.items()}
        number_buildings_NO_tank_NO_PV = {key: number * (1 - self.percentage_tank_storage)
                                          for key, number in total_number_buildings_NO_PV.items()}

        # BATTERY
        # calculate the number of buildings with a battery storage
        total_number_buildings_battery_5kWp = {key: self.percentage_battery_storage * number
                                               for key, number in total_number_buildings_5kWp.items()}
        total_number_buildings_battery_10kWp = {key: self.percentage_battery_storage * number
                                                for key, number in total_number_buildings_10kWp.items()}

        # number of buildings with battery AND tank:
        number_buildings_battery_tank_5kWp = {key: number * self.percentage_tank_storage
                                              for key, number in total_number_buildings_battery_5kWp.items()}
        number_buildings_battery_tank_10kWp = {key: number * self.percentage_tank_storage
                                               for key, number in total_number_buildings_battery_10kWp.items()}
        # number of buildings ONLY with battery:
        number_buildings_battery_NO_tank_5kWp = {key: number * (1 - self.percentage_tank_storage)
                                                 for key, number in total_number_buildings_battery_5kWp.items()}
        number_buildings_battery_NO_tank_10kWp = {key: number * (1 - self.percentage_tank_storage)
                                                  for key, number in total_number_buildings_battery_10kWp.items()}
        # TANK
        # total number of PV buildings without battery (substract all battery buildings from total):
        total_number_buildings_NO_battery_5kWp = {key: total_number_buildings_5kWp[key] - number
                                                  for key, number in total_number_buildings_battery_5kWp.items()}
        total_number_buildings_NO_battery_10kWp = {key: total_number_buildings_10kWp[key] - number
                                                   for key, number in total_number_buildings_battery_10kWp.items()}
        # number of buildings without battery and with tank
        number_buildings_NO_battery_tank_5kWp = {key: number * self.percentage_tank_storage
                                                 for key, number in total_number_buildings_NO_battery_5kWp.items()}
        number_buildings_NO_battery_tank_10kWp = {key: number * self.percentage_tank_storage
                                                  for key, number in total_number_buildings_NO_battery_10kWp.items()}
        # number of buildings without battery and without tank
        number_buildings_NO_battery_NO_tank_5kWp = {key: number * (1 - self.percentage_tank_storage)
                                                    for key, number in total_number_buildings_NO_battery_5kWp.items()}
        number_buildings_NO_battery_NO_tank_10kWp = {key: number * (1 - self.percentage_tank_storage)
                                                     for key, number in total_number_buildings_NO_battery_10kWp.items()}

        configuration_numbers = {"PV0B0T0": number_buildings_NO_tank_NO_PV,
                                 "PV0B0T1500": number_buildings_tank_NO_PV,
                                 "PV5B0T0": number_buildings_NO_battery_NO_tank_5kWp,
                                 "PV5B7T0": number_buildings_battery_NO_tank_5kWp,
                                 "PV5B0T1500": number_buildings_NO_battery_tank_5kWp,
                                 "PV5B7T1500": number_buildings_battery_tank_5kWp,
                                 "PV10B0T0": number_buildings_NO_battery_NO_tank_10kWp,
                                 "PV10B7T0": number_buildings_battery_NO_tank_10kWp,
                                 "PV10B0T1500": number_buildings_NO_battery_tank_10kWp,
                                 "PV10B7T1500": number_buildings_battery_tank_10kWp}
        return configuration_numbers

    def plot_difference_on_national_level(self, scenario: list, variable2plot: str) -> None:
        """plots change in elec consumption for a scenario as bar chart divided into the different
         household equipments"""
        matplotlib.rc("font", size=14)
        cmap = plt.get_cmap("tab10")
        if variable2plot == "Year_E_Grid":
            ylabel = "Electricity demand from the grid \n in MWh/year (logarithmic)"
            kW2MW = 1_000  # change from kW to MW
            twinx_ylabel = "percentage decrease of electricity \n consumption per configuration"
        elif variable2plot == "OperationCost":
            ylabel = "Operation cost"
            kW2MW = 1
        elif variable2plot == "Year_PVSelfConsumptionRate":
            ylabel = "Self-consumption rate in %"
            twinx_ylabel = "percentage increase in self-consumption rate"
            kW2MW = 1
            fig1 = plt.figure(figsize=(9, 6))
            ax = fig1.gca()

        else:
            kW2MW = 1_000  # change from kW to MW
            ylabel = "PV generation sold to the grid \n in MWh/year (logarithmic)"
            twinx_ylabel = "percentage decrease of PV generation \n sold to the grid"
        line_plot_values_1 = []
        line_plot_values_2 = []
        # third figure for the total impact:

        for env_nr, env in enumerate(self.environment):

            # second figure to zoom in:
            # fig2 = plt.figure(figsize=(9, 6)
            if variable2plot == "Year_PVSelfConsumptionRate":
                pass
            else:
                fig1 = plt.figure(figsize=(9, 6))
                ax = fig1.gca()
                ax_twin = ax.twinx()
                ax = fig1.gca()

            xticks = np.arange(10)
            ax.set_xticks(xticks)
            width = 0.9

            # for total plot:
            sum_opt = []
            sum_ref = []
            sum_ref_all = []

            total_saving_values = []
            ref_saving_values = []
            line_plot_color_1 = CONS().dark_red
            line_plot_color_2 = CONS().dark_blue
            df_for_analysis = pd.DataFrame(columns=self.configurations.keys())
            df_for_analysis2 = pd.DataFrame(columns=self.configurations.keys())
            # calculate the consumption for the different buildings
            for tick_index, (config, config_dict) in enumerate(self.configurations.items()):
                opt, ref = self.select_grid_electricity_consumption(
                    scenario[config],
                    PVPower=config_dict["PVPower"],
                    TankSize=config_dict["TankSize"],
                    BatteryCapacity=config_dict["BatteryCapacity"],
                    environment=env,
                    percentag_optimized_buildings=self.optimized_percentage,
                    variable=variable2plot
                )
                sum_opt.append(sum(opt.values()))
                sum_ref.append(sum(ref.values()))
                no_use_, ref_all = self.select_grid_electricity_consumption(
                    scenario[config],
                    PVPower=config_dict["PVPower"],
                    TankSize=config_dict["TankSize"],
                    BatteryCapacity=config_dict["BatteryCapacity"],
                    environment=env,
                    percentag_optimized_buildings=1,
                    variable=variable2plot
                )
                sum_ref_all.append(sum(ref_all.values()))

                if variable2plot == "Year_PVSelfConsumptionRate":
                    ax.bar(xticks[tick_index], sum(ref_all.values()) / len(ref_all),
                           width=width, edgecolor="black", color=CONS().grey)
                    if env == 2:
                        ax.bar(xticks[tick_index] + width/4, sum(opt.values()) / len(opt) - sum(ref.values()) / len(ref),
                               bottom=sum(ref_all.values()) / len(ref_all), width=0.45,
                               color=CONS().dark_blue, edgecolor="black")
                        total_saving_values.append((sum(opt.values()) / len(opt) - sum(ref.values()) / len(ref)).item())
                        ref_saving_values.append((sum(ref.values()) / len(ref)).item())
                    else:
                        ax.bar(xticks[tick_index] - width/4, sum(opt.values()) / len(opt) - sum(ref.values()) / len(ref),
                               bottom=sum(ref_all.values()) / len(ref_all), width=0.45,
                               color=CONS().dark_red, edgecolor="black")
                        total_saving_values.append((sum(opt.values()) / len(opt) - sum(ref.values()) / len(ref)).item())
                        ref_saving_values.append((sum(ref.values()) / len(ref)).item())

                else:
                    ax.bar(xticks[tick_index], sum(ref_all.values()) / kW2MW,
                           width=width, edgecolor="black", color=CONS().grey)
                    ax.text(xticks[tick_index], sum(ref_all.values()) / kW2MW,
                            f"{round(sum(ref_all.values()).item() / kW2MW):,}".replace(",", " "), ha="center", size=10)

                    # ausnahme weil ich einen fehler noch nicht gefunden habe:
                    if tick_index == 0 and env == 2:
                        line_plot_values_2.append(0)
                        total_saving_values.append(0)
                    else:
                        # check if energy demand is reduced or raised:
                        if env == 2:
                            line_plot_values_2.append(
                                ((sum(opt.values()) - sum(ref.values())) / sum(ref.values())).item())
                            total_saving_values.append(((sum(opt.values()) - sum(ref.values())) / kW2MW).item())
                        else:
                            line_plot_values_1.append(
                                ((sum(opt.values()) - sum(ref.values())) / sum(ref.values())).item())
                            total_saving_values.append(((sum(opt.values()) - sum(ref.values())) / kW2MW).item())
            df_for_analysis = df_for_analysis.append( pd.Series(total_saving_values, index=list(self.configurations.keys())), ignore_index=True)
            df_for_analysis.to_excel(self.figure_path / Path(f"Aggregated_Results//change_{variable2plot}_env{env}_scenarios.xlsx"))
            df_for_analysis2 = df_for_analysis2.append( pd.Series(ref_saving_values, index=list(self.configurations.keys())), ignore_index=True)
            df_for_analysis2.to_excel(self.figure_path / Path(f"Aggregated_Results//ref_values_{variable2plot}_env{env}_scenarios.xlsx"))

        # mange labels etc.
        ax.set_xticklabels(list(scenario.keys()), rotation=45)
        if variable2plot == "Year_PVSelfConsumptionRate":
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
            blue_patch = matplotlib.patches.Patch(color=CONS().dark_blue,
                                                  label='variable price')
            red_patch = matplotlib.patches.Patch(color=CONS().dark_red, label='flat price')
            plt.legend(handles=[red_patch, blue_patch])#, fontsize=14, bbox_to_anchor=(0, 1.02, 1, 0.1), bbox_transform=ax.transAxes,
                       # loc='lower left', ncol=3, borderaxespad=0, mode='expand', frameon=True)
        else:
            ax.set_yscale('log')
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
            ax_twin.plot(xticks, line_plot_values_1, "D", color=line_plot_color_1, label="variable price")
            ax_twin.plot(xticks, line_plot_values_2, "X", color=line_plot_color_2, label="flat price")
            low, high = ax_twin.get_ylim()
            ax_twin.set_ylim(low, high)
            ax_twin.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
            ax_twin.spines["right"].set_visible(True)
            ax_twin.tick_params(axis="y", color="black")
            ax_twin.set_ylabel(twinx_ylabel)
            ax_twin.yaxis.label.set_color("black")
            ax_twin.spines["right"].set_edgecolor("black")
            [t.set_color("black") for t in ax_twin.yaxis.get_ticklabels()]
        ax.set_ylabel(ylabel)
        ax.set_xlabel("configurations")
        plt.legend(fontsize=14, bbox_to_anchor=(0, 1.02, 1, 0.1), bbox_transform=ax.transAxes,
                   loc='lower left', ncol=3, borderaxespad=0, mode='expand', frameon=True)
        plt.tight_layout()
        fig1.savefig(self.figure_path / Path(f"Aggregated_Results//change_{variable2plot}_env{env}_scenarios.png"))
        fig1.savefig(self.figure_path / Path(f"Aggregated_Results//change_{variable2plot}_env{env}_scenarios.svg"))

        blue_patch = matplotlib.patches.Patch(color="blue",
                                              label='reference consumption')
        orange_patch = matplotlib.patches.Patch(color="orange",
                                                label='increase in consumption \n through optimization')
        green_patch = matplotlib.patches.Patch(color=CONS().green,
                                               label='decrease in consumption \n through optimization')

        fig1.show()
        # fig2.show()

    def define_scenarios_for_upscaling(self):
        """3 scenarios for up scaling to national level are created for austria"""
        # conservative
        # percentage of the buildings with HP who also have PV adopted
        percentage_10kW = 0.014  # 1.4% in 3 scenarios
        percentage_5kW = 0.176  # 17.6%
        all_HP_buildings = self.visualize_total_number_of_buildings()
        scenario_numbers = self.calculate_scenario_numbers(percentage_5kW, percentage_10kW, all_HP_buildings)

        self.visualize_number_of_buildings_scenario(scenario_numbers)

        # self.plot_difference_on_national_level(scenario_numbers, "Year_E_Grid")
        # self.plot_difference_on_national_level(scenario_numbers, "OperationCost")
        # self.plot_difference_on_national_level(scenario_numbers, "Year_E_PV2Grid")
        self.plot_difference_on_national_level(scenario_numbers, "Year_PVSelfConsumptionRate")

    def run(self):
        for household_id in range(3):
            for environment_id in range(1, 2):
                self.visualization_SystemOperation(household_id + 1, environment_id, week=8)
                self.visualization_SystemOperation(household_id + 1, environment_id, week=34)


if __name__ == "__main__":
    CONN = DB().create_Connection(CONS().RootDB)

    # Visualization(CONN).plot_average_results()
    # Visualization(CONN).violin_plots()
    # Visualization(CONN).visualize_comparison2Reference(1, 1, week=8)

    # Visualization(CONN).run()

    # Visualization(CONN).find_negativ_cost_households()

    # Visualization(CONN).plot_energy_consumption()
    # Visualization(CONN).calculate_key_numbers()
    # Visualization(CONN).plot_households_without_storages()
    Teil2Paper = UpScalingToBuildingStock()
    Teil2Paper.define_scenarios_for_upscaling()
