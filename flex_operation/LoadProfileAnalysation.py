import json
import logging
import time
from functools import wraps
from pathlib import Path
from typing import List, Union, Dict

from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable, OperationScenarioComponent
from create_individual_scenarios import filter_only_electrified_buildings, reduce_number_of_buildings, pv_kWp_to_area
from dash_visualization import data_extraction
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


global PLOT_ON
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)


def performance_counter(func):
    @wraps(func)
    def wrapper(*args):
        t_start = time.perf_counter()
        result = func(*args)
        t_end = time.perf_counter()
        print("function >>{}<< time for execution: {}".format(func.__name__, t_end - t_start))
        return result

    return wrapper


class LoadProfileAnalyser:
    def __init__(self, project_name: str, country: str, year: int, filter: str = None):
        self.project_name = project_name
        self.country = country
        self.year = year
        self.cfg = Config(project_name)
        self.db = DB(config=self.cfg)
        self.scenario_table = self.db.read_dataframe(table_name=OperationTable.Scenarios)
        self.add_building_numbers_to_scenario_table()
        self.path_to_figures = self.cfg.fig

        # the number of entries in the dict is dependent on the number of scenarios being compared
        number_scenario_ids = self.get_energy_price_ids()
        self.total_benchmark_load = {number: np.zeros((8760,)) for number in number_scenario_ids}
        self.total_sems_load = {number: np.zeros((8760,)) for number in number_scenario_ids}

        self.ref_p2p_filename = self.cfg.output / f"Peak2Peak_REF.parquet.gzip"
        self.sems_p2p_filename = self.cfg.output / f"Peak2Peak_SEMS.parquet.gzip"
        self.aggregated_load_profile_filename = self.cfg.output / f"total_load_profile.csv"
        self.load_shift_filename = self.cfg.output / Path(f"shifted_demand.xlsx")
        # self.create_load_analyzation_files()

        # filter to select only buildings of a specific type
        self.filter = self.translate_the_filter(filter)

    def translate_the_filter(self, original_filter: str) -> Union[Dict[str, int], None]:
        """
        This function takes the original filter which are the options in the EU Map dropdown and translates them
        into a list of scenario IDs.
        :param original_filter: str, allowed options are:
        "All buildings"
        "buildings with PV"
        "buildings without PV"
        "buildings with battery"
        "buildings without battery"
        "buildings with thermal storages"
        "buildings without thermal storages"
        :return: List[int], ID_Scenario for the specific case
        """
        if original_filter == None:
            return None
        elif original_filter == "buildings with PV":
            numbers_df = data_extraction.create_scenario_selection_table_with_numbers(project_name=self.project_name)
            selection = numbers_df.query("ID_PV > 0")
            dictionary = {"ID_Scenario": scen_id for scen_id in selection["ID_Scenario"]}



    def peak_load(self):
        """
        The Peak Load is the highest level of demand observed over a given period of time – typically a day – divided
        into i time steps, for a given demand load j. Said demand load may correspond to an individual consumer, or it
        may correspond to the overall maximum demand at the aggregate level.
        𝑃 𝑗𝑚𝑎𝑥 = max 𝑃𝑗(𝑡𝑖)
        Returns:

        """

        pass

    def mean_daily_peak_load(self, profile: np.array) -> float:
        """
        The Mean Daily Peak Load is the average of the daily peak loads observed throughout the year. This is important
        for dimensioning of generation, network, and supply infrastructure.
        𝑃𝑀𝐷 = 1/365 Σ (𝑃𝑖𝑚𝑎𝑥)
        Returns: numpy array of 8760 entries for the year

        """
        peak_loads = 0
        number_of_days = 0
        for start_hour in np.arange(0, len(profile), 24):
            end_hour = start_hour + 24
            day_load = profile[start_hour:end_hour]
            peak_loads += day_load.max()
            number_of_days += 1
        mean_daily_peak_load = peak_loads / number_of_days
        return mean_daily_peak_load

    def load_factor(self, profile: np.array) -> np.array:
        """
        profile: np.array of 8760 entries
        The Load Factor is the ratio of the average of the load registered over a given period of time to the maximum
        load registered during the same period. The load factor is also considered to be a measure of the “peakyness”
        of the load profile, where a higher load factor indicates a smoother load profile. The higher the load factor,
        the better.
        𝐿𝐹 = mean(𝑃(𝑡)) / 𝑃𝑚𝑎𝑥
        Returns: np.array of 365 daily values

        """
        load_factor = []
        for start_hour in np.arange(0, len(profile), 24):
            end_hour = start_hour + 24
            day_load = profile[start_hour:end_hour]
            load_factor.append(day_load.mean() / day_load.max())

        return np.array(load_factor)

    def pv_self_consumption(self):
        """calculate the pv self consumption rate"""

    def cost_savings_on_country_level(self) -> dict:
        """ calculates the relative cost savings on a country level for each price ID
        returns a dict where the price IDs are the keys and values represent the relative cost savings in %"""
        cost_saving_table = self.operation_cost_savings()
        # group into price IDs:
        groups = cost_saving_table.groupby("ID_EnergyPrice")
        cost_dict = {}
        for i, group in groups:
            weights = group.loc[:, "total savings €"] / group.loc[:, "total savings €"].sum()
            percentage_shifted_country = (group.loc[:, "relative savings %"] * weights).sum()
            price_id = int(group.loc[0, "ID_EnergyPrice"])
            cost_dict[price_id] = percentage_shifted_country
        return cost_dict

    def operation_cost_savings(self) -> pd.DataFrame:
        """calculate the operation costs savings by comparing the total operation costs from the optimisation case
        to the reference case."""
        # calculate the cost savings
        # yearly data:
        yearly_results_benchmark = self.db.read_dataframe(table_name=OperationTable.ResultRefYear)
        yearly_results_optimization = self.db.read_dataframe(table_name=OperationTable.ResultOptYear)

        savings = (yearly_results_benchmark.loc[:, "TotalCost"] -
                   yearly_results_optimization.loc[:, "TotalCost"]) / 100  # €

        # calculate total savings
        total_savings = savings * self.scenario_table.loc[:, "number_of_buildings"]  # €

        # total operation cost
        # import the costs from reference table
        total_ref_cost = yearly_results_benchmark.loc[:, "TotalCost"] / 100 * \
                         self.scenario_table.loc[:, "number_of_buildings"]  # €
        relative_savings = total_savings / total_ref_cost * 100  # %

        cost_saving_table = self.scenario_table.loc[:, ["ID_Scenario", "number_of_buildings", "ID_EnergyPrice"]]
        cost_saving_table.loc[:, "total savings €"] = total_savings
        cost_saving_table.loc[:, "relative savings %"] = relative_savings
        return cost_saving_table

    def shifted_electricity_percentage(self):
        """calculate the shifted electricity as a percentage of the reference load in the same time"""

    def create_shifted_electricity_file(self):
        """Calculate how much electricity is shifted through SEMS"""
        # calculate the numbers of how many hours the electricity consumption is higher than in the benchmark and
        # how many hours it is lower.
        print(f"creating shifted electricity file for {self.country} {self.year}")
        numbers_df = pd.DataFrame(columns=["ID_Scenario",
                                           "energy_charging kWh", "energy_charging_stock MWh",
                                           "energy_shifted kWh", "energy_shifted_stock MWh",
                                           "percentage_of_shifted_electricity %",
                                           "percentage_of_shifted_electricity_stock %",
                                           "total_electricity_demand_grid kWh",
                                           "total_electricity_demand_grid_stock MWh"])
        numbers_df.loc[:, "ID_Scenario"] = self.scenario_table.loc[:, "ID_Scenario"]
        load_generator = self.grab_grid_loads_from_results()
        for id in self.scenario_table.ID_Scenario.values:
            ref_load, opt_load = next(load_generator)
            number_of_buildings = int(self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id,
                                                              "number_of_buildings"])

            # calculate the difference between the loads:
            load_difference = opt_load - ref_load
            total_electricity_demand_grid = ref_load.sum() / 1_000  # kWh
            total_electricity_demand_grid_stock = total_electricity_demand_grid * number_of_buildings / 1_000  # MWh

            # create variables:
            energy_charging = 0  # sum of energy that is higher in optimization compared to Benchmark
            energy_shifted = 0  # sum of energy that is lower in the optimization compared to benchmark
            reference_energy = 0  # sum of energy that is used by the reference model while demand is shifted
            for i, load in enumerate(load_difference):
                # if load difference is positive, the optimization is "charging"
                if load > 0:
                    energy_charging += load / 1_000  # kWh
                # if load difference is negative, the optimization is "shifting"
                elif load < 0:
                    energy_shifted += abs(load) / 1_000  # kWh
                    reference_energy += ref_load[i] / 1_000  # kWh
                else:  # the load difference is 0, nothing happens
                    pass

            energy_charging_stock = energy_charging * number_of_buildings / 1_000  # MWh
            energy_shifted_stock = energy_shifted * number_of_buildings / 1_000  # MWh
            reference_energy_stock = reference_energy * number_of_buildings / 1_000  # MWh
            percentage_of_shifted_electricity = energy_shifted / reference_energy * 100  # %
            percentage_of_shifted_electricity_stock = energy_shifted_stock / reference_energy_stock * 100  # %

            # save numbers to pandas df:
            numbers_df.loc[numbers_df.loc[:, "ID_Scenario"] == id,
                           ["energy_charging kWh",
                            "energy_charging_stock MWh",
                            "energy_shifted kWh",
                            "energy_shifted_stock MWh",
                            "percentage_of_shifted_electricity %",
                            "percentage_of_shifted_electricity_stock %",
                            "total_electricity_demand_grid kWh",
                            "total_electricity_demand_grid_stock MWh"
                            ]] = [energy_charging,
                                  energy_charging_stock,
                                  energy_shifted,
                                  energy_shifted_stock,
                                  percentage_of_shifted_electricity,
                                  percentage_of_shifted_electricity_stock,
                                  total_electricity_demand_grid,
                                  total_electricity_demand_grid_stock]

        numbers_df.to_excel(self.load_shift_filename, index=False)

    def load_shifted_electricity_building_stock(self) -> pd.DataFrame:
        """
        loads the load shifting file and adds the ID of energy prices from the scenario table to the table
        :return:
        """
        # calculate the energy that is shifted in each building:
        load_shift_df = pd.read_excel(self.load_shift_filename, engine="openpyxl")
        # add the price ID to the df
        load_shift_df.loc[:, "ID_EnergyPrice"] = self.scenario_table.loc[:, "ID_EnergyPrice"]
        return load_shift_df

    def calc_total_shifted_electricity_building_stock(self) -> dict:
        """
        :return: dict[price_id, float], total amount of electricity shifted from the stock in MWh
        """
        df = self.load_shifted_electricity_building_stock()
        df_prices = df.groupby("ID_EnergyPrice")
        shifted = {}
        for i, group in df_prices:
            shifted_mwh = group["energy_shifted_stock MWh"].sum()
            price_id = int(group.loc[0, "ID_EnergyPrice"])
            shifted[price_id] = shifted_mwh
        return shifted

    def calc_percentage_shifted_electricity_building_stock(self) -> dict:
        """ returns a dictionary with the price ID as key and the percentage of electricity shifted through heat pumps
        in a country when optimized."""
        df = self.load_shifted_electricity_building_stock()
        # calculate the percentage of shifted electricity for the whole country for each price:
        # group the df into the different prices:
        df_prices = df.groupby("ID_EnergyPrice")
        percentages = {}
        for i, group in df_prices:
            weights = group.loc[:, "energy_shifted_stock MWh"] / group.loc[:, "energy_shifted_stock MWh"].sum()
            percentage_shifted_country = (group.loc[:, "percentage_of_shifted_electricity_stock %"] * weights).sum()
            price_id = int(group.loc[0, "ID_EnergyPrice"])
            percentages[price_id] = percentage_shifted_country

        return percentages

    def calc_pv_share_building_stock_optimization(self) -> pd.Series:
        """
        the pv share is calculated by summing up the produced PV over the whole year of all buildings within the stock
        and dividing it by the total amount of electricity used by the building stock. The PV share is calculated for
        the optimization results.
        """
        # read the yearly values
        results_df = self.db.read_dataframe(table_name=OperationTable.ResultOptYear,
                                            column_names=["ID_Scenario", "PhotovoltaicProfile", "Load"])
        # add energy price ID
        id_df = self.db.read_dataframe(table_name=OperationTable.Scenarios,
                                       column_names=["ID_Scenario", "ID_EnergyPrice"])
        df_merged = pd.merge(left=results_df, right=id_df, on="ID_Scenario")
        total_pv_generation = df_merged.groupby("ID_EnergyPrice")["PhotovoltaicProfile"].sum()
        total_demand = df_merged.groupby("ID_EnergyPrice")["Load"].sum()
        return total_pv_generation / total_demand * 100  # %

    def grab_grid_loads_from_results(self) -> (np.array, np.array):
        for id in self.scenario_table.ID_Scenario.values:
            # grab the loads
            reference_load = self.db.read_parquet(table_name=OperationTable.ResultRefHour,
                                                  scenario_ID=id,
                                                  ).loc[:, "Grid"].to_numpy()
            sems_load = self.db.read_parquet(table_name=OperationTable.ResultOptHour,
                                             scenario_ID=id,
                                             ).loc[:, "Grid"].to_numpy()
            yield reference_load, sems_load

    def grab_indoor_temperature_from_results(self, id):
        sems_temperature = self.db.read_dataframe(
            self.opt_db_name_hourly, column_names=["T_room"], filter={"ID_Scenario": id}
        ).to_numpy().flatten()
        benchmark_temperature = self.db.read_dataframe(
            self.ref_db_name_hourly, column_names=["T_room"], filter={"ID_Scenario": id}
        ).to_numpy().flatten()
        return benchmark_temperature, sems_temperature

    def grab_heating_demand_from_results(self, id):
        benchmark_heating_demand = self.db.read_dataframe(
            self.ref_db_name_hourly, column_names=["Q_room_heating"], filter={"ID_Scenario": id}
        ).to_numpy().flatten()
        return benchmark_heating_demand

    def define_boiler_id(self) -> Dict[int, str]:
        df = self.db.read_dataframe(OperationScenarioComponent.Boiler.table_name)
        return dict(zip(df["ID_Boiler"], df["type"]))

    def load_invert_parquet_file(self):
        # load the parquet file:
        path_2_parquet = self.cfg.project_folder / self.country / f"{self.year}.parquet.gzip"
        df = pd.read_parquet(path_2_parquet)
        df_heat_pumps = filter_only_electrified_buildings(df)  # filter only buildings with heat pumps
        df_reduced = reduce_number_of_buildings(df_heat_pumps)  # reduce the number of buildings
        return df_reduced

    def add_building_numbers_to_scenario_table(self) -> None:
        # load the hp numbers from the DB
        numbers = self.db.read_dataframe(table_name="ScenarioNumbers")
        self.scenario_table.loc[:, "number_of_buildings"] = numbers.loc[:, "number_of_buildings"]

    def calculate_total_load_profiles(self, scenario_id: int, benchmark_load: np.array, sems_load: np.array):
        number_of_buildings = self.scenario_table.loc[
            self.scenario_table.loc[:, "ID_Scenario"] == scenario_id, "number_of_buildings"
        ].values
        price_id = self.scenario_table.loc[
            self.scenario_table.loc[:, "ID_Scenario"] == scenario_id, "ID_EnergyPrice"
        ].values

        self.total_benchmark_load[int(price_id)] += benchmark_load * number_of_buildings
        self.total_sems_load[int(price_id)] += sems_load * number_of_buildings

    def save_load_profile_results(self, load_profile_file: Path):
        load_profile_np = np.zeros((3, 1))
        price_ids = self.get_energy_price_ids()

        for price_id in price_ids:
            matrix = np.vstack([self.total_sems_load[price_id] / 1e9,
                                self.total_benchmark_load[price_id] / 1e9,
                                np.full((8760,), price_id)])
            load_profile_np = np.hstack([load_profile_np, matrix])
            # update the results table:

        load_profile_np = load_profile_np[:, 1:]  # delete row with zeros
        load_profile_df = pd.DataFrame(load_profile_np.T, columns=["sems load (GW)", "reference load (GW)", "price ID"])
        # save the total load profiles:
        load_profile_df.to_csv(load_profile_file, sep=";", index=False)

    def define_seasons_from_hours(self, hourly_profile: np.array):
        # 21 March start of spring -> 0 - 80 days (winter) and 355 - 365
        winter = np.hstack([hourly_profile[0: 80 * 24], hourly_profile[355 * 24: 365 * 24]]).flatten()
        spring = hourly_profile[81 * 24: 173 * 24].flatten()
        summer = hourly_profile[173 * 24: 268 * 24].flatten()
        autumn = hourly_profile[268 * 24: 355 * 24].flatten()
        return [winter, spring, summer, autumn]

    def define_seasons(self, df: Union[pd.DataFrame, np.array]) -> list:
        """

        Args:
            df: 365 days peak to peak df

        Returns: np.arrays (winter, spring, summer, autumn) as flattend array from df

        """
        if isinstance(df, pd.DataFrame):
            # 21 March start of spring -> 0 - 80 days (winter) and 355 - 365
            winter = pd.concat([df.iloc[:, 0:80], df.iloc[:, 355:365]], axis=1).to_numpy().flatten()
            # 21. June start of summer -> 81 - 172 (spring)
            spring = df.iloc[:, 81:173].to_numpy().flatten()
            # 23. September start of Autumn -> 173 - 267 (Summer)
            summer = df.iloc[:, 173:268].to_numpy().flatten()
            # 21. December start of Winter -> 268 - 355 (Autumn)
            autumn = df.iloc[:, 268:355].to_numpy().flatten()

            return [winter, spring, summer, autumn]
        # else the data is a Series:
        elif isinstance(df, np.ndarray):
            winter = np.hstack([df[0:80], df[355:365]])
            spring = df[81:173].flatten()
            summer = df[173:268].flatten()
            autumn = df[268:355].flatten()

            return [winter, spring, summer, autumn]
        else:
            raise TypeError

    def create_box_plot_positions(self, list_price_ids: List[str]) -> list:
        positions = []
        increment = len(list_price_ids) + 1
        for i, label in enumerate(list_price_ids):
            positions.append([j * increment + i for j in range(4)])  # 4 for the 4 seasons
        return positions

    def mean_list(self, lists: List[List[float]]) -> List[float]:
        n = len(lists[0])
        mean_list = [0] * n
        for i in range(n):
            mean_list[i] = sum(l[i] for l in lists) / len(lists)
        return mean_list

    def create_bar_plot(self,
                        sems_dictionary: Dict[int, np.array],
                        figure_name: str,
                        ref_profile: List[np.array] = None):
        colors = ["#95a5a6", "#3498db", "orange", "#2ecc71", "#e74c3c"]
        # create new dict with the labels as keys:
        if ref_profile:  # if there is a ref_profile we need to add it as the first entry to the sems_dictionary:
            labels = ["reference"] + [f"price {int(id)}" for id in sems_dictionary.keys()]
            new_dict = dict(zip(labels, [ref_profile] + [prof for prof in sems_dictionary.values()]))
        else:  # otherwise the new dict will just consist of the sems_dict
            labels = [f"price {id}" for id in sems_dictionary.keys()]
            new_dict = dict(zip(labels, list(sems_dictionary.values())))
        # define the positions in the plot
        positions = self.create_box_plot_positions(labels)
        # define the xticks through the positions:
        xticks = self.mean_list(positions)
        x_axis = ["winter", "spring", "summer", "autumn"]
        legend_elements = []
        bar_list = []
        for id, (name, peak2peak_profile) in enumerate(new_dict.items()):
            bar = plt.bar(x=positions[id], height=peak2peak_profile, color=colors[id], label=labels[id])
            bar_list.append(bar)

        plt.legend(loc="upper right", ncol=1)
        plt.xticks(xticks)
        ax = plt.gca()
        ax.set_xticklabels(x_axis)

        plt.ylabel(f"Mean Daily Peak Load (MW)")
        plt.tight_layout()
        plt.savefig(self.path_to_figures / Path(f"{figure_name}.svg"))
        plt.show()

    def create_boxplot_p_to_p_difference(self,
                                         sems_dictionary: Dict[int, np.array],
                                         figure_name: str,
                                         ref_profile: Dict[int, np.array] = None,
                                         ):

        sems_color = "#3498db"
        colors = ["#95a5a6", "#3498db", "orange", "#2ecc71", "#e74c3c"]
        # create new dict with the labels as keys:
        if ref_profile:  # if there is a ref_profile we need to add it as the first entry to the sems_dictionary:
            # the ref profile is identical for all price ids -> only take the first one:
            labels = ["reference"] + [f"price {int(id)}" for id in sems_dictionary.keys()]
            new_dict = dict(zip(labels, [ref_profile[1]] + [prof for prof in sems_dictionary.values()]))
        else:  # otherwise the new dict will just consist of the sems_dict
            labels = [f"price {id}" for id in sems_dictionary.keys()]
            new_dict = dict(zip(labels, list(sems_dictionary.values())))
        # define the positions in the plot
        positions = self.create_box_plot_positions(labels)
        # define the xticks through the positions:
        xticks = self.mean_list(positions)
        legend_elements = []
        # make figure
        x_axis = ["winter", "spring", "summer", "autumn"]
        bp_list = []
        for id, (name, peak2peak_profile) in enumerate(new_dict.items()):
            bp = plt.boxplot(peak2peak_profile, patch_artist=True, positions=positions[id])
            bp_list.append(bp)

        for i, bp in enumerate(bp_list):
            for box in (bp["boxes"]):
                box.set(color="black")
                box.set(facecolor=colors[i])
            for median in bp['medians']:
                median.set_color('black')

            legend_elements.append(Patch(facecolor=colors[i], label=labels[i]))
        plt.legend(handles=legend_elements, loc="upper right", ncol=1)
        plt.xticks(xticks)
        ax = plt.gca()
        ax.set_xticklabels(x_axis)
        if figure_name == f"Peak_to_peak_boxplot_total_load":
            # plt.ylim(0, 350)
            plt.ylabel("peak to peak \n difference (MW)")
        elif "price" in figure_name:
            plt.ylabel("peak to peak \n difference (€/kWh)")
        elif "load_factor" in figure_name.lower():
            plt.ylabel("Load factor")
        else:
            # plt.ylim(0, 2_700)
            plt.ylabel("peak to peak \n difference (W)")
        plt.tight_layout()

        plt.savefig(self.path_to_figures / Path(f"{figure_name}.svg"))
        plt.show()

    def peak_to_peak_difference(self, load_profile: np.array) -> np.array:
        """

        Args:
            load_profile: load profile of 8760 values

        return: array of 365 values which correspond to the difference of the highest and lowest load each day
        """
        peak_to_peak_difference = []
        for start_hour in np.arange(0, 8760, 24):
            end_hour = start_hour + 24
            day_load = load_profile[start_hour:end_hour]
            peak_to_peak_difference.append(day_load.max() - day_load.min())

        return np.array(peak_to_peak_difference)

    def create_load_profile_files(self):
        # check if those files exist:
        print(f"creating peak-to-peak files and total load profiles for {self.country} {self.year}")
        p_to_p_benchmark = []
        p_to_p_sems = []
        results_generator = self.grab_grid_loads_from_results()
        for id in self.scenario_table.ID_Scenario.values:
            benchmark_load, sems_load = next(results_generator)
            # calculate peak to peak difference for benchmark load:
            p_to_p_benchmark.append(self.peak_to_peak_difference(benchmark_load))
            p_to_p_sems.append(self.peak_to_peak_difference(sems_load))

            # calculate the total load profiles
            self.calculate_total_load_profiles(id, benchmark_load, sems_load)
            # print(f"{id} scenario done")
        self.save_load_profile_results(self.aggregated_load_profile_filename)

        df_sems = pd.DataFrame(p_to_p_sems, columns=[f"{i}" for i in range(365)])
        df_ref = pd.DataFrame(p_to_p_benchmark, columns=[f"{i}" for i in range(365)])
        # save the results to json:

        df_ref.to_parquet(self.ref_p2p_filename, compression='gzip', index=False)
        df_sems.to_parquet(self.sems_p2p_filename, compression='gzip', index=False)

    def analyze_peak_to_peak_difference_single_buildings(self):
        # load the parquet files:
        sems_df = pd.read_parquet(self.sems_p2p_filename)
        ref_df = pd.read_parquet(self.ref_p2p_filename)
        # get number of electricity scenarios:
        scenario_ids = self.get_energy_price_ids()
        number_of_profiles = len(scenario_ids)
        ref_profiles = ref_df.iloc[np.arange(0, len(ref_df), number_of_profiles), :]  # profile is always the same

        sems_profiles = {}
        # filter out the profiles for each price scenario:
        for number in scenario_ids:
            sems_profiles[number] = sems_df.iloc[np.arange(1, len(ref_df), number_of_profiles), :]

        # create list of numpy arrays with the values for each season
        ref_list = self.define_seasons(ref_profiles)
        sems_lists = {}
        for id, profile in sems_profiles.items():
            sems_lists[id] = self.define_seasons(profile)

        # create boxplot
        self.create_boxplot_p_to_p_difference(sems_dictionary=sems_lists,
                                              figure_name=f"Peak_to_peak_boxplot_single_buildings",
                                              ref_profile=ref_list)

    def get_energy_price_ids(self) -> list:
        return self.db.read_dataframe(OperationScenarioComponent.EnergyPrice.table_name)["ID_EnergyPrice"].to_list()

    def get_energy_price_profiles(self) -> Dict[int, np.array]:
        """
        Returns: dict of the profiles
        """
        price_ids = self.get_energy_price_ids()
        profile_dict = {}
        price_df = self.db.read_dataframe(table_name=OperationTable.EnergyPriceProfile)
        for id in price_ids:
            profile_dict[id] = price_df.loc[:, f"electricity_{id}"].to_numpy()

        return profile_dict

    def analyze_peak_to_peak_difference_prices(self):
        # first get the different energy prices:
        price_profiles = self.get_energy_price_profiles()
        p2p_dict = {}
        # iterate through the different price profiles and get the peak to peak values for each day:
        for id, price_profile in price_profiles.items():
            p2p_dict[id] = self.peak_to_peak_difference(price_profile)

        # iterate through the peak to peak days and define to which season they belong:
        season_dict = {}
        for id, p2p_price in p2p_dict.items():
            season_dict[id] = self.define_seasons(p2p_price)

        # create boxplot
        self.create_boxplot_p_to_p_difference(season_dict, f"Peak_to_peak_boxplot_prices")

    def load_aggregated_load_profiles(self) -> (pd.DataFrame, pd.DataFrame):
        """ loads the aggregated load profiles from csv and returns the ref and sems profile as pd.Dataframe
        if they don't exist they will be created"""
        df = pd.read_csv(self.aggregated_load_profile_filename, sep=";")
        sems_load = df.loc[:, ["price ID", "sems load (GW)"]]
        ref_load = df.loc[:, ["price ID", "reference load (GW)"]]
        return ref_load, sems_load

    def analyze_peak_to_peak_difference_total_load_profiles(self, plot_on=False) -> (
            Dict[int, List[np.array]], Dict[int, List[np.array]]):
        # load demand profiles from csv:
        df = pd.read_csv(self.aggregated_load_profile_filename, sep=";")

        ids = list(df.loc[:, "price ID"].unique())
        peak_to_peak_day_sems = {}
        peak_to_peak_day_ref = {}
        for id in ids:
            sems_load = df.loc[df.loc[:, "price ID"] == id, "sems load (GW)"].to_numpy() * 1_000  # MW
            peak_to_peak_day_sems[id] = self.define_seasons(self.peak_to_peak_difference(sems_load))

            ref_load = df.loc[df.loc[:, "price ID"] == id, "reference load (GW)"].to_numpy() * 1_000  # MW
            peak_to_peak_day_ref[id] = self.define_seasons(self.peak_to_peak_difference(ref_load))

        # create boxplot
        if plot_on:
            self.create_boxplot_p_to_p_difference(sems_dictionary=peak_to_peak_day_sems,
                                                  figure_name=f"Peak_to_peak_boxplot_total_load",
                                                  ref_profile=peak_to_peak_day_ref)
        return peak_to_peak_day_ref, peak_to_peak_day_sems

    def analyze_mean_daily_peak_load(self) -> Dict[int, list]:
        # load demand profiles from csv:
        df = pd.read_csv(self.aggregated_load_profile_filename, sep=";")
        ids = list(df.loc[:, "price ID"].unique())
        mean_daily_sems_dict = {}
        for id in ids:
            sems_load = df.loc[df.loc[:, "price ID"] == id, "sems load (GW)"].to_numpy() * 1_000  # MW
            sems_load_season = self.define_seasons_from_hours(sems_load)
            mean_daily_sems = []
            for season in [0, 1, 2, 3]:
                mean_daily_sems.append(self.mean_daily_peak_load(sems_load_season[season]))
            mean_daily_sems_dict[id] = mean_daily_sems

        ref_load = df.loc[df.loc[:, "price ID"] == 1, "reference load (GW)"].to_numpy() * 1_000  # MW
        ref_load_season = self.define_seasons_from_hours(ref_load)
        mean_daily_ref = []
        for season in [0, 1, 2, 3]:
            mean_daily_ref.append(self.mean_daily_peak_load(ref_load_season[season]))

        self.create_bar_plot(sems_dictionary=mean_daily_sems_dict,
                             figure_name="Mean_Daily_Peak_Loads",
                             ref_profile=mean_daily_ref)

    def analyze_load_factor(self, plot_on=False) -> (Dict[int, np.array], Dict[int, np.array]):
        df = pd.read_csv(self.aggregated_load_profile_filename, sep=";")
        ids = list(df.loc[:, "price ID"].unique())
        sems_load_factor_season_dict = {}
        ref_load_factor_season_dict = {}
        # iterate through different price IDs:
        for id in ids:
            sems_load = df.loc[df.loc[:, "price ID"] == id, "sems load (GW)"].to_numpy() * 1_000  # MW
            sems_load_factor = self.load_factor(sems_load)
            sems_load_factor_seasons = self.define_seasons(sems_load_factor)
            sems_load_factor_season_dict[id] = sems_load_factor_seasons

            # reference profile is always the same since price does not affect it:
            ref_load = df.loc[df.loc[:, "price ID"] == id, "reference load (GW)"].to_numpy() * 1_000  # MW
            ref_load_factor = self.load_factor(ref_load)
            ref_load_factor_season = self.define_seasons(ref_load_factor)
            ref_load_factor_season_dict[id] = ref_load_factor_season

        if plot_on:
            self.create_boxplot_p_to_p_difference(sems_dictionary=sems_load_factor_season_dict,
                                                  figure_name="Load_Factor_total_load",
                                                  ref_profile=ref_load_factor_season_dict)
        return ref_load_factor_season_dict, sems_load_factor_season_dict

    def calculate_consumption_weights(self, id_scenario: pd.Series) -> pd.Series:
        """ select the ids from the self.scenario table and calculate the weights through the
        number of buildings."""
        filtered_scenario = self.scenario_table[self.scenario_table[f"ID_Scenario"].isin(list(id_scenario))]
        # weighted by the number of buildings:
        weights = filtered_scenario.loc[:, "number_of_buildings"] / \
                  filtered_scenario.loc[:, "number_of_buildings"].sum()
        return weights

    def add_pv_self_consumption_to_table(self, table: pd.DataFrame) -> pd.DataFrame:
        # self consumption is sum(PV2Bat, PV2EV, PV2Load) / Photovoltaic Profile
        nominator = table.loc[:, ["PV2Load", "PV2Bat", "PV2EV"]].sum(axis=1)
        denominator = table.loc[:, "PhotovoltaicProfile"]
        table.loc[:, "PV_self_consumption %"] = nominator / denominator * 100
        # drop the buildings without PV (drop nan in axis=0)
        return_table = table.dropna(axis=0)
        weights = self.calculate_consumption_weights(id_scenario=return_table["ID_Scenario"])
        return_table.loc[:, "weights"] = weights
        return_table.loc[:, "ID_EnergyPrice"] = self.scenario_table[
            self.scenario_table[f"ID_Scenario"].isin(list(return_table["ID_Scenario"]))
        ]["ID_EnergyPrice"]
        return return_table.dropna(axis=0)

    def calc_country_self_consumption(self, filter: None) -> (Dict[int, float], Dict[int, float]):
        """
        calculates the PV self consumption on country level.
        only considers the buildings that have a PV installed!
        The output is a dictionary with the key being the price index and the value the PV self consumption
         for the reference case and the optimization
        """
        ref_table = self.db.read_dataframe(table_name=OperationTable.ResultRefYear, filter=filter)
        opt_table = self.db.read_dataframe(table_name=OperationTable.ResultOptYear, filter=filter)

        ref = self.add_pv_self_consumption_to_table(ref_table)
        opt = self.add_pv_self_consumption_to_table(opt_table)

        opt_groups = opt.groupby("ID_EnergyPrice")
        ref_groups = ref.groupby("ID_EnergyPrice")
        opt_dict = {}
        ref_dict = {}
        for i, group in opt_groups:
            total_self_consumption = (group.loc[:, "PV_self_consumption %"] * group.loc[:, "weights"]).sum()
            price_id = list(group.loc[:, "ID_EnergyPrice"])[0]
            opt_dict[price_id] = total_self_consumption

            # in the reference case the results are independent from the electricity price
            ref_group = ref_groups.get_group(i)
            total_self_consumption_ref = (ref_group.loc[:, "PV_self_consumption %"] * ref_group.loc[:, "weights"]).sum()
            ref_dict[price_id] = total_self_consumption_ref

        return ref_dict, opt_dict

    def create_load_analyzation_files(self, overwrite: bool = False):
        """ creates the load profile files if they do not already exist
        overwrite: if True than the files will be overwritten, if False, it will be checked if the files exist and only
        create them if they don't
        """
        if overwrite:
            self.create_load_profile_files()
            self.create_shifted_electricity_file()
        else:
            if not self.aggregated_load_profile_filename.exists() or not self.sems_p2p_filename.exists() or not self.ref_p2p_filename.exists():
                self.create_load_profile_files()
            else:
                logger.info(f"peak-to-peak and load profile files for {self.project_name} already exist.")

            if not self.load_shift_filename.exists():
                self.create_shifted_electricity_file()
            else:
                logger.info(f"shifted electricity files for {self.project_name} already exist.")


if __name__ == "__main__":
    PLOT_ON = False
    # LoadProfileAnalyser(project_name="D5.4_AUT_2020", year=2020, country="AUT", filter="buildings with PV")

    country = "AUT"
    year = 2020
    YEARS = [2020, 2030, 2040, 2050]
    COUNTRIES = [
        'BEL',
        'AUT',
        'HRV',
        'CZE',
        'DNK',
        'FRA',
        'DEU',
        'LUX',
        'HUN',
        'IRL',
        'ITA',
        'NLD',
        'POL',
        'PRT',
        'SVK',
        'SVN',
        'ESP',
        'SWE',
        # 'GRC',
        'LVA',
        # 'LTU',
        # 'MLT',
        'ROU',
        'BGR',
        'FIN',
        'EST',
    ]

    # for country in COUNTRIES:
    #     for year in YEARS:
    #         print(f"{country}_{year}")
    #         LoadProfileAnalyser(project_name=f"D5.4_{country}_{year}", country=country, year=year).db.read_dataframe(OperationTable.ResultRefYear)

    country = "AUT"
    year = 2020
    LoadProfileAnalyser(project_name=f"D5.4_{country}_{year}", country=country, year=year).calc_total_shifted_electricity_building_stock()
