# -*- coding: utf-8 -*-
__author__ = 'Philipp'

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import re

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
            ylabel = "Grid demand (kWh)"
        elif label == "OperationCost":
            ylabel = "Operation cost"
        elif label == "Year_PVSelfSufficiencyRate":
            ylabel = "self sufficiency (%)"
        elif label == "Year_E_Load":
            ylabel = "total electricity demand (kWh)"

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
            figure.suptitle(
                variable2plot.replace("Year_",
                                      "") + " average difference between optimization and reference \n for different electricity prices")
            y_axis_title = variable2plot.replace("Year_", "") + "\n optimization - reference (kWh)"
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

        # def create_number_matrix():
        #     # building ID 1 to 11 is SFH, ID 12 to 22 is RH
        #     df = pd.DataFrame(index=["mean", "median", "thirst quantile", "third quantile", "standard deviation",
        #                              "variance"], columns=[optimization_results_year.columns])
        #     for environment in unique_Environment:
        #         single_building = optimization_results_year.loc[(optimization_results_year["ID_Building"] == 1) &
        #                                       (optimization_results_year["ID_Environment"] == environment)]
        #
        #
        #         pass
        #
        #
        #     pass

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

                # for house_id in configuration_IDs[configuration]:
                #     Sum_ref_hour = DB().read_DataFrame(REG_Table().Res_Reference_HeatingCooling,
                #                                        self.Conn,
                #                                        variable,
                #                                        ID_Household=house_id, ID_Environment=environment_id)
                #     Sum_ref = Sum_ref + Sum_ref_hour.to_numpy()
                #
                #     Sum_opt_hour = DB().read_DataFrame(REG_Table().Res_SystemOperationHour,
                #                                        self.Conn,
                #                                        variable,
                #                                        ID_Household=house_id, ID_Environment=environment_id)
                #     Sum_opt = Sum_opt + Sum_opt_hour.to_numpy()

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
        reference_results_OP_below_zero = self.ReferenceOperationYear.loc[
            (self.ReferenceOperationYear["OperationCost"] < 0) &
            (self.ReferenceOperationYear["ID_Building"] < 12)]
        # in the optimization model
        optimization_results_OP_below_zero = self.SystemOperationYear.loc[
            (self.SystemOperationYear["OperationCost"] < 0) &
            (self.SystemOperationYear["ID_Building"] < 12)]

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
                                  "in the reference model", reference_results_OP_below_zero)
        create_percentage_barplot("Percentage of household configurations with negative energy costs \n "
                                  "in the optimization model", optimization_results_OP_below_zero)
        create_percentage_barplot("Percentage of household configurations with negative energy costs \n "
                                  "in both models", pd.concat([optimization_results_OP_below_zero,
                                                               reference_results_OP_below_zero]))

    def violin_plots(self):

        reference_results = self.ReferenceOperationYear
        optimization_results = self.SystemOperationYear
        reference_results.loc[:, "Option"] = "Reference"
        optimization_results.loc[:, "Option"] = "Optimization"
        frame = pd.concat([reference_results, optimization_results], axis=0)

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
            fig, axes = plt.subplots(2, 3, sharey=True, figsize=[15, 10])
            fig.suptitle(title)
            axes = axes.flatten()
            sns.violinplot(x="ID_Building", y=y_achse, hue="Option", data=Frame, ax=axes[0], split=True,
                           inner="stick", palette="muted")
            axes[0].set_xlabel("Building ID")
            axes[0].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_PVPower", y=y_achse, hue="Option", data=Frame, ax=axes[1], split=True,
                           inner="stick", palette="muted")
            axes[1].set_xlabel("PV Power")
            axes[1].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_TankSize", y=y_achse, hue="Option", data=Frame, ax=axes[2], split=True,
                           inner="stick", palette="muted")
            axes[2].set_xlabel("TankSize")
            axes[2].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="Household_BatteryCapacity", y=y_achse, hue="Option", data=Frame, ax=axes[3],
                           split=True, inner="stick", palette="muted")
            axes[3].set_xlabel("Battery")
            axes[3].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="ID_AgeGroup", y=y_achse, hue="Option", data=Frame, ax=axes[4],
                           split=True, inner="stick", palette="muted")
            axes[4].set_xlabel("indoor set \n temperature")
            axes[4].set_ylabel(self.determine_ylabel(y_achse))

            sns.violinplot(x="ID_SpaceHeating", y=y_achse, hue="Option", data=Frame, ax=axes[5],
                           split=True, inner="stick", palette="muted")
            axes[5].set_xlabel("HP type")
            axes[5].set_ylabel(self.determine_ylabel(y_achse))
            axes[5].set_xticklabels([str(tick.get_text()).replace("Air_HP", "Air HP").replace("Water_HP", "Ground HP")
                                    for tick in axes[5].get_xticklabels()])

            for i in range(axes.shape[0]):
                axes[i].grid(axis="y")
                if axes[i].get_ylim()[1] > 999:
                    axes[i].get_yaxis().set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))

                axes[i].yaxis.set_tick_params(labelbottom=True)

            figure_name = "ViolinPlot " + title
            plt.tight_layout()
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".png"), dpi=200,
                        format='PNG')
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".svg"))
            plt.show()

        frame = frame.loc[frame["ID_Building"] < 12]
        for key, value in {1: "variable", 2: "flat"}.items():
            price_type_frame = frame.loc[frame["Environment_ElectricityPriceType"] == key]
            create_violin_plot_overview(price_type_frame, "OperationCost",
                                        f"Operation Costs Overview with {value} price")
            create_violin_plot_overview(price_type_frame, "Year_E_Load",
                                        f"total electricity consumption Overview with {value} price")
            create_violin_plot_overview(price_type_frame, "Year_PVSelfSufficiencyRate",
                                        f"PV self sufficiency rate Overview with {value} price")
            create_violin_plot_overview(price_type_frame, "Year_E_Grid",
                                        f"total electricity from the Grid Overview with {value} price")

        ref_list = pd.DataFrame(columns=optimization_results.columns)
        opt_list = pd.DataFrame(columns=optimization_results.columns)
        for index in range(optimization_results.shape[0]):
            if reference_results.loc[index, "OperationCost"] < optimization_results.loc[index, "OperationCost"]:
                print("Household ID: " + str(self.SystemOperationYear.loc[index, "ID_Household"]))
                print("Battery: " + str(reference_results.loc[index, "Household_BatteryCapacity"]))
                print("PV Peak: " + str(reference_results.loc[index, "Household_PVPower"]))
                print("Tank: " + str(reference_results.loc[index, "Household_TankSize"]))

                ref_list = ref_list.append(reference_results.iloc[index, :], ignore_index=True)
                opt_list = opt_list.append(optimization_results.iloc[index, :], ignore_index=True)

        frame_for_violin = pd.concat([ref_list, opt_list], axis=0)

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

    def visualize_number_of_buildings(self) -> None:
        """creates plot that shows how many buildings exist and how many buildings have heat pumps
        in each building category"""
        buildings_frame = DB().read_DataFrame(REG_Table().Gen_OBJ_ID_Building, self.Conn)
        invert_index = pd.to_numeric(buildings_frame.index_invert)
        normal_index = pd.to_numeric(buildings_frame["index"]).to_numpy()
        matplotlib.rc("font", size=14)
        fig = plt.figure(figsize=(9, 6))
        ax = plt.gca()
        fig.suptitle("Number of buildings in each representative category")
        for i, index in enumerate(invert_index):
            total_number, HP_number = self.get_total_number_of_buildings(index)
            ax.bar(i + 1, total_number, color=self.COLOR.dark_blue)
            ax.bar(i + 1, HP_number, color=self.COLOR.green)

        plt.xlabel("building category")
        plt.ylabel("number of buildings")
        ax.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
        # create legend
        blue_patch = matplotlib.patches.Patch(color=self.COLOR.dark_blue, label='total number of buildings')
        gree_path = matplotlib.patches.Patch(color=self.COLOR.green, label='number of buildings \n with heat pump')
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
                (self.SystemOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 1) &
                (self.SystemOperationYear.loc[:, REG_Var().ID_AgeGroup] == 1)]

            res_optimization_year_elec2 = self.SystemOperationYear.loc[
                (self.SystemOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 2) &
                (self.SystemOperationYear.loc[:, REG_Var().ID_AgeGroup] == 1)]

            res_reference_year_elec1 = self.ReferenceOperationYear.loc[
                (self.ReferenceOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 1) &
                (self.ReferenceOperationYear.loc[:, REG_Var().ID_AgeGroup] == 1)]

            res_reference_year_elec2 = self.ReferenceOperationYear.loc[
                (self.ReferenceOperationYear.loc[:, REG_Var().Environment_ElectricityPriceType] == 2) &
                (self.ReferenceOperationYear.loc[:, REG_Var().ID_AgeGroup] == 1)]

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
                                     "ID_SpaceHeating": {"Air_HP": "", "Water_HP": ""}}

            fig = plt.figure()
            plt.suptitle(title)
            ax = plt.gca()

            xticks = np.arange(5)
            plt.xticks(np.arange(5))
            colors = [self.COLOR.dark_blue, self.COLOR.dark_red, self.COLOR.blue, self.COLOR.brown, self.COLOR.grey,
                      self.COLOR.green, self.COLOR.dark_green, self.COLOR.dark_grey, self.COLOR.yellow, self.COLOR.red,
                      self.COLOR.purple]
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
                                color=colors[i], alpha=0.1)

                    plt.boxplot(e_grid, meanline=True, positions=[xticks[tick] + spacing[space_tick]])
                    patches.append(matplotlib.patches.Patch(color=colors[i],
                                                            label=parameter.replace("Household_", "").
                                                            replace("ID_", "") +
                                                            " " + str(option_name).replace("Air_HP", "Air HP").
                                                            replace("Water_HP", "Ground HP") + str(option_unit)))
                    i += 1
                    space_tick += 1
                tick += 1
            xticks = [param.replace("Household_", "").replace("ID_", "") for param in parameters_to_analyze.keys()]
            plt.xticks(np.arange(5), xticks, rotation=45)
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

        fixed_variables = ["ID_AgeGroup", "Environment_ElectricityPriceType"]
        remaining_variables = remaining_scenario_variables(fixed_variables)
        fixed_variables_dict = remaining_scenario_variables(list(remaining_variables.keys()))

        optimization_elec1, optimization_elec2, reference_elec1, reference_elec2 = create_subresults_frame()

        create_boxplot_with_scatter_plot(optimization_elec1,
                                         "Impact of scenario parameters on grid demand \n with variable price, optimization model")
        create_boxplot_with_scatter_plot(optimization_elec2,
                                         "Impact of scenario parameters on grid demand \n with flat price, optimization model")

        create_boxplot_with_scatter_plot(reference_elec1,
                                         "Impact of scenario parameters on grid demand \n with variable price, reference model")
        create_boxplot_with_scatter_plot(reference_elec2,
                                         "Impact of scenario parameters on grid demand \n with flat price, reference model")

    def calculate_key_numbers(self) -> None:
        """this function calculates important values for the paper and is here to play around"""
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
        pass


class UpScalingToBuildingStock:

    def __init__(self, percentage_optimized_buildings):
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
        self.percentage_optimized_buildings = percentage_optimized_buildings

    def get_total_number_of_building(self, building_index: int) -> (float, float):
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

    def calculate_HP_PV_numbers(self, percentage: float) -> dict:
        """calculates the number of each building ID with HP and PV"""
        HP_buildings_dict_percentage, HP_buildings_dict = self.calculate_percentage_of_hp_buildings()
        kWp_5 = {x: y * percentage for x, y in HP_buildings_dict_percentage.items()}
        number_buildings = {index: kWp_5[index] * number for index, number in HP_buildings_dict.items()}
        return number_buildings

    def calculate_percentage_numbers(self, percentage: float, total_numbers_dict: dict) -> dict:
        """calculate the total number of buildings with certain appliance percentage based eg numer of buildings that
        have PV and ALSO have battery storage"""
        return {index: value * percentage for index, value in total_numbers_dict.items()}

    def select_grid_electricity_consumption(self, buildings_dict: dict, PVPower, TankSize, BatteryCapacity,
                                            environment) -> (dict, dict):
        """returns the annual grid electricity consumption for the households in the dict for reference and
        optimization. NO COOLING, fixed indoor temp, 20% ground source HP"""
        cooling_adoption = 0
        id_agegroup = 1
        results_ref = self.ReferenceOperationYear
        results_opt = self.SystemOperationYear

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
        def calculate_total_elec(frame) -> dict:
            perc = self.percentage_optimized_buildings
            return {i: (
                               frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                         (frame.loc[:, "ID_SpaceHeating"] == "Air_HP")].Year_E_Grid.to_numpy() +
                               frame.loc[(frame.loc[:, "ID_Building"] == i + 1) &
                                         (frame.loc[:, "ID_SpaceHeating"] == "Water_HP")].Year_E_Grid.to_numpy()
                       ) * round(nr * perc) for i, nr in buildings_dict.items()}

        opt_elec_grid = calculate_total_elec(opt)
        ref_elec_grid = calculate_total_elec(ref)
        return opt_elec_grid, ref_elec_grid


    def calculate_scenario_numbers(self, percentage_5kW, percentage_10kW):
        """calculate all nessecary numbers for each scenario"""
        # 30% of Households with PV have battery storage!
        percentage_battery_storage = 0.3
        # 50% of Households with PV have a thermal storage:
        percentage_tank_storage = 0.5
        # 10% if Households with PV have Battery and thermal storage:
        percentage_tank_battery = 0.1
        # percentage of households that have only PV and no other appliance:
        percentage_PV_nothing_else = 1 - percentage_tank_battery - percentage_tank_storage - percentage_battery_storage
        environment = [1, 2]
        # 50% of households with HP and without PV adopt a thermal storage:
        percentage_tank_no_PV = 0.3
        percentage_buildings_no_PV = 1 - percentage_10kW - percentage_5kW


        # calculate the number of hp buildings that have PV for certain scenario
        number_buildings_5kWp = self.calculate_HP_PV_numbers(percentage_5kW)
        number_buildings_10kWp = self.calculate_HP_PV_numbers(percentage_10kW)

        number_buildings_battery_5kWp = self.calculate_percentage_numbers(percentage_battery_storage,                                                                   number_buildings_5kWp)
        number_buildings_battery_10kWp = self.calculate_percentage_numbers(percentage_battery_storage,
                                                                           number_buildings_10kWp)

        number_buildings_tank_5kWp = self.calculate_percentage_numbers(percentage_tank_storage,                                                                     number_buildings_5kWp)
        number_buildings_tank_10kWp = self.calculate_percentage_numbers(percentage_tank_storage,
                                                                        number_buildings_10kWp)


        number_buildings_tank_battery_5kWp = self.calculate_percentage_numbers(percentage_tank_battery,                                                                         number_buildings_5kWp)
        number_buildings_tank_battery_10kWp = self.calculate_percentage_numbers(percentage_tank_battery,
                                                                                number_buildings_10kWp)

        number_buildings_only_5kWp = self.calculate_percentage_numbers(percentage_PV_nothing_else,
                                                                       number_buildings_5kWp)
        number_buildings_only_10kWp = self.calculate_percentage_numbers(percentage_PV_nothing_else,
                                                                        number_buildings_10kWp)

        number_buildings_no_PV = self.calculate_HP_PV_numbers(percentage_buildings_no_PV)
        number_buildings_tank_no_PV = self.calculate_percentage_numbers(percentage_tank_no_PV, number_buildings_no_PV)
        number_buildings_no_tank_no_PV = self.calculate_percentage_numbers((1 - percentage_tank_no_PV),
                                                                           number_buildings_no_PV)
        configuration_numbers = {"PV0B0T0": number_buildings_no_tank_no_PV,
                                 "PV0B0T1500": number_buildings_tank_no_PV,
                                 "PV5B0T0": number_buildings_only_5kWp,
                                 "PV5B7T0": number_buildings_battery_5kWp,
                                 "PV5B0T1500": number_buildings_tank_5kWp,
                                 "PV5B7T1500": number_buildings_tank_battery_5kWp,
                                 "PV10B0T0": number_buildings_only_10kWp,
                                 "PV10B7T0": number_buildings_battery_10kWp,
                                 "PV10B0T1500": number_buildings_tank_10kWp,
                                 "PV10B7T1500": number_buildings_tank_battery_10kWp}

        def plot_grid_electricity_difference_on_national_level(configurations):
            fig = plt.figure()
            fig.suptitle(f"Annual electricity demand change \n {int(100*self.percentage_optimized_buildings)}% "
                         f"of buildings with HP are optimized")
            ax = plt.gca()
            for env in environment:
                total_change = 0
                if env == 1:
                    color = CONS().dark_red
                else:
                    color = CONS().dark_blue
                # calculate the grid electricity consumption for the different buildings
                for config, config_dict in configurations.items():
                    opt_elec_grid, ref_elec_grid = self.select_grid_electricity_consumption(
                        configuration_numbers[config],
                        PVPower=config_dict["PVPower"],
                        TankSize=config_dict["TankSize"],
                        BatteryCapacity=config_dict["BatteryCapacity"],
                        environment=env
                    )
                    total_change += (sum(opt_elec_grid.values()) - sum(ref_elec_grid.values())) / 1_000
                    ax.bar(config, (sum(opt_elec_grid.values()) - sum(ref_elec_grid.values())) / 1_000, color=color)  # MWh
                    ax.text(config, 0, str(round(sum(configuration_numbers[config].values()))), ha="center")
                print(f"total change in demand for env {env}: {total_change}")
            ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))
            red_patch = matplotlib.patches.Patch(color=CONS().dark_red, label='variable price')
            blue_patch = matplotlib.patches.Patch(color=CONS().dark_blue, label='flat price')
            plt.legend(handles=[red_patch, blue_patch])
            plt.ylabel("Grid demand (MWh/year)")
            plt.xlabel("configurations")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

        configurations = {"PV0B0T0": {"PVPower": 0, "TankSize": 0, "BatteryCapacity": 0},
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
        plot_grid_electricity_difference_on_national_level(configurations)


    def define_scenarios_for_upscaling(self):
        """3 scenarios for up scaling to national level are created for austria"""
        # conservative
        # percentage of the buildings with HP who also have PV adopted
        percentage_10kW = 0.2
        percentage_5kW = 0.5
        self.calculate_scenario_numbers(percentage_5kW, percentage_10kW)


    def run(self):
        for household_id in range(3):
            for environment_id in range(1, 2):
                self.visualization_SystemOperation(household_id + 1, environment_id, week=8)
                self.visualization_SystemOperation(household_id + 1, environment_id, week=34)




if __name__ == "__main__":
    CONN = DB().create_Connection(CONS().RootDB)
    # Visualization(CONN).visualize_number_of_buildings()

    # Visualization(CONN).plot_average_results()
    # Visualization(CONN).violin_plots()
    # Visualization(CONN).visualize_comparison2Reference(1, 1, week=8)

    # Visualization(CONN).run()

    # Visualization(CONN).find_negativ_cost_households()

    Visualization(CONN).plot_energy_consumption()
    # Visualization(CONN).calculate_key_numbers()

    percentage_optimized_buildings = 0.3
    Teil2Paper = UpScalingToBuildingStock(percentage_optimized_buildings)
    Teil2Paper.define_scenarios_for_upscaling()
