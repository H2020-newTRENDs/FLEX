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

        path2buildingsnumber = CONS().ProjectPath / Path("_Philipp/inputdata/AUT") / Path(
            "040_aut__2__BASE__1_zzz_short_bc_seg__b_building_segment_sh.csv")
        number_of_buildings_frame = pd.read_csv(path2buildingsnumber, sep=";", encoding="ISO-8859-1", header=0)
        self.number_of_buildings_frame = number_of_buildings_frame.iloc[4:, :]

    def adjust_ylim(self, lim_array):

        tiny = (lim_array[1] - lim_array[0]) * 0.02
        left = lim_array[0] - tiny
        right = lim_array[1] + tiny

        return np.array((left, right))

    def get_total_number_of_buildings(self, building_index: int) -> (float, float):
        """takes the invert building index and returns the number of buildings
        with heat pumps and total number of buildings."""
        total_number_of_buildings = self.number_of_buildings_frame.loc[
                        pd.to_numeric(self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index, :
                        ].number_of_buildings.sum()

        number_of_buildings_with_HP = self.number_of_buildings_frame.loc[
            (pd.to_numeric(self.number_of_buildings_frame.loc[:, "buildingclasscsvid"]) == building_index) &
            ((self.number_of_buildings_frame.loc[:, "heatingsystem"] == "geothermal_central_heatpump(air/water)_Heat pump air/water") |
            (self.number_of_buildings_frame.loc[:, "heatingsystem"] == "geothermal_central_heatpump(water/water)_Heat pump brine/water shallow"))
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
                        config_ids_opt[("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity/1_000), TankSize),
                                        "E{:n}".format(environment))] = optim_ids
                        config_ids_ref[("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity/1_000), TankSize, environment),
                                        "E{:n}".format(environment))] = ref_ids

                        # save same household configuration total results to dict
                        config_frame_opt_year[
                            ("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity/1_000), TankSize),
                             "E{:n}".format(environment))] = optim_year
                        config_frame_ref_year[("PV{:n}B{:n}T{:n}".format(PVPower, int(BatteryCapacity/1_000), TankSize),
                                              "E{:n}".format(environment))] = ref_year
                        i += 1

        def plot_barplot(frame_opt, frame_ref, variable2plot):
            # create barplot with total amounts:
            figure = plt.figure()
            figure.suptitle(
                variable2plot.replace("Year_", "") + " average difference between optimization and reference \n for different electricity prices")
            y_axis_title = variable2plot.replace("Year_", "") + "\n optimization - reference (kWh)"
            barplot_frame = pd.DataFrame(
                columns=["Configuration", y_axis_title, "Scenario"])  # dataframe for seaborn:
            for key in frame_opt.keys():
                Sum_opt = frame_opt[key][variable2plot].sum() / number_of_profiles_aggregated
                Sum_ref = frame_ref[key][variable2plot].sum() / number_of_profiles_aggregated
                Differenz = Sum_opt - Sum_ref

                # check what the environment means:
                if key[1] == "E1":
                    scenarioname = "flat electricity price"
                elif key[1] == "E2":
                    scenarioname = "variable electricity price"
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
            figure.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(fig_name + ".png"), dpi=200, format='PNG')
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
                DataFrame_opt = pd.read_sql('select ' + variable + ' from ' + REG_Table().Res_SystemOperationHour + condition, con=self.Conn)
                Sum_opt = DataFrame_opt.sum()
                DataFrame_ref = pd.read_sql('select ' + variable + ' from ' + REG_Table().Res_Reference_HeatingCooling + condition, con=self.Conn)
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


    def find_special_households(self):
        # households that have negative operation costs:
        reference_results = self.ReferenceOperationYear
        optimization_results = self.SystemOperationYear
        # in the reference model
        reference_results_OP_below_zero = self.ReferenceOperationYear.loc[
            self.ReferenceOperationYear["OperationCost"] < 0]
        # in the optimization model
        optimization_results_OP_below_zero = self.SystemOperationYear.loc[
            self.SystemOperationYear["OperationCost"] < 0]

        def create_percentage_barplot(title, dataframe):
            analyzed_parameters = {"Household_TankSize": {0: "l", 1500: "l"},
                                   "Household_CoolingAdoption": {0: "", 1: ""},
                                   "Household_PVPower": {0: "kWp", 5: "kWp", 10: "kWp"},
                                   "Household_BatteryCapacity":  {0: "kWh", 7000: "kWh"},
                                   "Environment_ElectricityPriceType": {1: "variable", 2: "flat"}}
            fig = plt.figure()
            fig.suptitle(title)
            ax = plt.gca()
            for parameter in analyzed_parameters.keys():
                total_number_ref = len(dataframe)
                unit = analyzed_parameters[parameter]
                x_label = parameter.replace("Household_", "")
                x_label = x_label.replace("Environment_", "")
                # create frames with the
                uniques = np.unique(dataframe[parameter])
                # TODO implement the number of the buildings with HP!
                # count the number of households in each column
                if len(uniques) == 1:
                    plt.bar(x_label, 1, color=self.COLOR.red, label=uniques[0])
                    plt.text(x_label, 0.5, str(int(uniques[0])) + str(unit[int(uniques[0])]), ha="center")
                elif len(uniques) == 2:
                    first_bar_ref = len(
                        dataframe.loc[dataframe[parameter] == uniques[0]])
                    percentage_ref = first_bar_ref / total_number_ref

                    plt.bar(x_label, percentage_ref, color=self.COLOR.red, label=uniques[0])
                    plt.bar(x_label, 1-percentage_ref, bottom=percentage_ref, color=self.COLOR.blue, label=uniques[1])
                    plt.text(x_label, percentage_ref/2, str(int(uniques[0])) + str(unit[int(uniques[0])]), ha="center")
                    plt.text(x_label, percentage_ref + (1-percentage_ref) / 2, str(int(uniques[1])) + str(unit[int(uniques[1])]), ha="center")
                elif len(uniques) == 3:
                    first_bar_ref = len(dataframe.loc[dataframe[parameter] == uniques[0]])
                    second_bar_ref = len(dataframe.loc[dataframe[parameter] == uniques[1]])
                    percentage_ref_1 = first_bar_ref / total_number_ref
                    percentage_ref_2 = second_bar_ref / total_number_ref
                    percentage_ref_3 = 1 - percentage_ref_1 - percentage_ref_2

                    plt.bar(x_label, percentage_ref_1, color=self.COLOR.red, label=uniques[0])
                    plt.bar(x_label, percentage_ref_2, bottom=percentage_ref_1, color=self.COLOR.blue, label=uniques[1])
                    plt.bar(x_label, percentage_ref_3, bottom=percentage_ref_2, color=self.COLOR.green, label=uniques[2])
                    plt.text(x_label, percentage_ref_1 / 2, str(int(uniques[0])) + str(unit[int(uniques[0])]), ha="center")
                    plt.text(x_label, percentage_ref_1 + percentage_ref_2 / 2, str(int(uniques[1])) + str(unit[int(uniques[1])]), ha="center")
                    plt.text(x_label, percentage_ref_1 + percentage_ref_2 + percentage_ref_3 / 2, str(int(uniques[2])) + str(unit[int(uniques[2])]), ha="center")

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

        for house_type in ["SFH", "RH"]:
            create_single_violin_plot(frame, "OperationCost", "Operation Cost for buildings",
                                      house_type, x="ID_Building", hue="Option")
            create_single_violin_plot(frame, "Year_Q_HeatPump", "Heat demand for buildings",
                                      house_type, x="ID_Building", hue="Option")

        create_single_violin_plot(reference_results, "OperationCost",
                                  "Influence of Battery on reference buildings operation cost", x="ID_Building",
                                  hue="Household_BatteryCapacity")
        create_single_violin_plot(optimization_results, "OperationCost",
                                  "Influence of Battery on optimized buildings operation cost", x="ID_Building",
                                  hue="Household_BatteryCapacity")



        def create_violin_plot_overview(Frame, y_achse, title):
            fig, axes = plt.subplots(2, 2, sharey=True, figsize=[15, 10])
            fig.suptitle(title)
            axes = axes.flatten()
            sns.violinplot(x="ID_Building", y=y_achse, hue="Option", data=Frame, ax=axes[0], split=True,
                           inner="stick", palette="muted")
            sns.violinplot(x="Household_PVPower", y=y_achse, hue="Option", data=Frame, ax=axes[1], split=True,
                           inner="stick", palette="muted")

            sns.violinplot(x="Household_TankSize", y=y_achse, hue="Option", data=Frame, ax=axes[2], split=True,
                           inner="stick", palette="muted")
            sns.violinplot(x="Household_BatteryCapacity", y=y_achse, hue="Option", data=Frame, ax=axes[3],
                           split=True, inner="stick", palette="muted")
            for i in range(axes.shape[0]):
                axes[i].grid(axis="y")
                if axes[i].get_ylim()[1] > 999:
                    axes[i].get_yaxis().set_major_formatter(
                        matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',').replace(",", " ")))

                axes[i].yaxis.set_tick_params(labelbottom=True)

            figure_name = "ViolinPlot " + title
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".png"), dpi=200, format='PNG')
            plt.savefig(CONS().FiguresPath / Path("Aggregated_Results") / Path(figure_name + ".svg"))
            # plt.show()

        create_violin_plot_overview(frame, "OperationCost", "Operation Costs")
        create_violin_plot_overview(frame, "Year_E_Load", "total electricity consumption")
        create_violin_plot_overview(frame, "Year_PVSelfSufficiencyRate", "PV self sufficiency rate")
        create_violin_plot_overview(frame, "Year_E_Grid", "total electricity from the Grid")

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
            ax.bar(i+1, total_number, color=self.COLOR.dark_blue)
            ax.bar(i+1, HP_number, color=self.COLOR.green)

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




    def run(self):
        for household_id in range(3):
            for environment_id in range(1, 2):
                self.visualization_SystemOperation(household_id + 1, environment_id, week=8)
                self.visualization_SystemOperation(household_id + 1, environment_id, week=34)


if __name__ == "__main__":
    CONN = DB().create_Connection(CONS().RootDB)
    Visualization(CONN).visualize_number_of_buildings()

    # Visualization(CONN).plot_average_results()
    # Visualization(CONN).violin_plots()
    # Visualization(CONN).box_plots()
    # Visualization(CONN).visualize_comparison2Reference(1, 1, week=8)

    # Visualization(CONN).run()

    # Visualization(CONN).find_special_households()