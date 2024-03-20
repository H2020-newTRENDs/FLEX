import shutil
from pathlib import Path
import pandas as pd
import numpy as np
import random

random.seed(42)
from typing import List
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from matplotlib.cm import get_cmap
import joblib

matplotlib.use('Agg')

from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable


class ECEMFBuildingComparison:
    def __init__(self, region: str):
        self.region = region
        self.path_to_osm = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM")
        self.path_to_project = Path(
            r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects") / f"ECEMF_T4.3_{region}_{year}"
        self.original_building_ids = self.load_original_building_ids()
        self.clustered_building_df = self.load_clustered_building_df()
        self.original_building_df = self.load_original_building_df()
        self.db = DB(config=Config(project_name=f"ECEMF_T4.3_{region}_{year}"))
        self.scenario_table = self.db.read_dataframe(OperationTable.Scenarios)
        # add the cluster ID to the original building df
        self.add_cluster_id_to_original_building_df()

    def load_original_building_ids(self):
        return pd.read_excel(self.path_to_project / f"Original_Building_IDs_to_clusters_{self.region}.xlsx")

    def load_clustered_building_df(self):
        return pd.read_excel(self.path_to_project / f"OperationScenario_Component_Building_small_{self.region}.xlsx")

    def load_original_building_df(self):
        # add the construction period, and hwb from invert, height from Urban3R
        column_names = [
            "ID_Building",
            "construction_period_start",
            "construction_period_end",
            "hwb",
            "numero_viv",
            "height",
            "demanda_ca"
        ]
        df_combined = pd.read_excel(self.path_to_osm / f"combined_building_df_{self.region}.xlsx",
                                    usecols=column_names).rename(columns={"numero_viv": "number of persons"})
        df_5r1c = pd.read_excel(self.path_to_osm / f"OperationScenario_Component_Building_{self.region}.xlsx")
        df = df_5r1c.merge(df_combined, on="ID_Building")
        return df

    def add_cluster_id_to_original_building_df(self):
        for cluster_id, building_ids in self.original_building_ids.iteritems():
            list_of_ids = building_ids.dropna().astype(int).tolist()
            self.original_building_df.loc[
                self.original_building_df.loc[:, "ID_Building"].isin(list_of_ids), "Cluster ID"
            ] = cluster_id

    def select_original_buildings_for_cluster(self, cluster_id: int):
        real_building_ids = self.original_building_ids.loc[:, cluster_id].dropna().astype(int).tolist()
        return self.original_building_df.query(f"ID_Building in {real_building_ids}")

    def show_variance_within_clusters(self):
        columns_to_plot = [
            "Cluster ID",
            "Af",
            "Hop",
            "Htr_w",
            "Hve",
            "CM_factor",
            "effective_window_area_south",
            "construction_period_start",
            "hwb",
            "demanda_ca",
            "number of persons",
        ]

        # drop buildings that are alone in a cluster as they obviously align with the original data:
        value_counts = self.original_building_df['Cluster ID'].value_counts()
        values_to_drop = value_counts[value_counts == 1].index
        filtered_df = self.original_building_df[~self.original_building_df['Cluster ID'].isin(values_to_drop)]
        plot_df = filtered_df[columns_to_plot].sort_values("Cluster ID").melt(id_vars="Cluster ID", )

        number_columns = len(columns_to_plot) - 1
        fig = px.box(data_frame=plot_df,
                     color="Cluster ID",
                     facet_col="variable",
                     facet_col_wrap=1,
                     facet_col_spacing=0.01,
                     facet_row_spacing=0.02,
                     width=800,
                     height=350 * number_columns, )
        fig.update_yaxes(matches=None)
        fig.show()

    def main(self):
        self.show_variance_within_clusters()


class ECEMFPostProcess:
    def __init__(self,
                 year: int,
                 region: str,
                 pv_installation_percentage: float,
                 dhw_storage_percentage: float,
                 buffer_storage_percentage: float,
                 heating_element_percentage: float,
                 air_hp_percentage: float,
                 ground_hp_percentage: float,
                 direct_electric_heating_percentage: float,
                 gases_percentage: float,
                 ac_percentage: float,
                 battery_percentage: float,
                 prosumager_percentage: float,
                 baseline: dict
                 ):
        """
        :param year: int of the year that corresponds to the renovation of buildings and lowered heat demand
        :param region: name of the region (str)
        :param pv_installation_percentage: percentage of buildings having a PV installed (SFH = 5kWp, MFH = 15kWp)
        :param dhw_storage_percentage: percentage of buildings having a domestic hot water storage (SFH=300l, MFH=700l)
        :param buffer_storage_percentage: percentage of buildings having a heating buffer storage (SFH=700l, MFH=1500)
        :param heating_element_percentage: percentage of buildings having a heating element (max power 5kWh SFH with
                PV or 15kWh MFH with PV)
        :param air_hp_percentage: percentage of buildings having an air sourced heat pump
        :param ground_hp_percentage: percentage of buildings having a ground sourced heat pump
        :param direct_electric_heating_percentage: percentage of buildings having a direct electric heating system
        :param ac_percentage: percentage of buildings having an air conditioner for cooling (COP=4)
        """
        self.baseline = baseline
        self.region = region
        self.year = year
        self.path_to_project = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{region}_{year}"
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{region}/data_output/"
        self.clustered_building_df = self.load_clustered_building_df()
        self.db = DB(config=Config(project_name=f"ECEMF_T4.3_{region}_{year}"))
        self.scenario_table = self.db.read_dataframe(OperationTable.Scenarios)
        self.pv_sizes = {
            1: 0,  # kWp
            2: 5,
            3: 15,
            4: 5,
            5: 15,
            6: 5,
            7: 15
        }
        self.battery_sizes = {
            1: 0,
            2: 7,  # kWh
            3: 15
        }
        self.buffer_sizes = {
            1: 0,
            2: 700,
            3: 1500,  # l
        }
        self.DHW_sizes = {
            1: 0,
            2: 300,
            3: 700,  # l
        }
        self.boiler = {
            1: "direct electric",
            2: "air HP",
            3: "ground HP",
            4: "no heating",
            5: "gas",
        }
        self.cooling = {
            1: "cooling",
            2: "no cooling",
        }
        self.pv_installation_percentage = pv_installation_percentage
        self.dhw_storage_percentage = dhw_storage_percentage
        self.buffer_storage_percentage = buffer_storage_percentage
        self.heating_element_percentage = heating_element_percentage
        self.air_hp_percentage = air_hp_percentage
        self.ground_hp_percentage = ground_hp_percentage
        self.direct_electric_heating_percentage = direct_electric_heating_percentage
        self.gases_percentage = gases_percentage
        self.ac_percentage = ac_percentage
        self.battery_percentage = battery_percentage
        self.prosumager_percentage = prosumager_percentage
        self.no_heating_percentage = 1 - (
                self.air_hp_percentage + self.ground_hp_percentage + self.direct_electric_heating_percentage + self.gases_percentage)
        assert 0 <= self.pv_installation_percentage <= 1
        assert 0 <= self.dhw_storage_percentage <= 1
        assert 0 <= self.buffer_storage_percentage <= 1
        assert 0 <= self.heating_element_percentage <= 1
        assert 0 <= self.air_hp_percentage <= 1
        assert 0 <= self.ground_hp_percentage <= 1
        assert 0 <= self.direct_electric_heating_percentage <= 1
        assert 0 <= self.ac_percentage <= 1
        assert 0 <= self.battery_percentage <= 1
        assert 0 <= self.prosumager_percentage <= 1
        assert 0 <= self.no_heating_percentage <= 1
        assert self.air_hp_percentage + self.ground_hp_percentage + self.direct_electric_heating_percentage + gases_percentage <= 1

    def load_clustered_building_df(self):
        return pd.read_excel(self.path_to_project / f"OperationScenario_Component_Building.xlsx")

    def percentage_of_total_buildings(self):
        """
        Percentage of total buildings that belong to this type at the studied location in this scenario
        :return:
        """

    @staticmethod
    def assign_id(list_of_percentages: List[float], ) -> int:
        """
        if list of percentages is only containing one element, the decision is taken between 1 and 0.
        :param total_buildings: number of buildings for the specific cluster
        :param list_of_percentages: percentages of the each choice. eg: (no PV, PV east, PV west, PV optimal) =
                (0.5, 0.15, 0.15, 0.2), Has to sum up to 1!
        :return:
        """

        if len(list_of_percentages) == 1:
            choices = [0, 1]  # random choice is taken between 0 and 1 (no and yes)
            weights = [1 - list_of_percentages[0], list_of_percentages[0]]
        elif len(list_of_percentages) > 1:
            choices = [i for i in range(len(list_of_percentages))]
            weights = list_of_percentages
        else:
            assert "list of percentages is empty"
        choice = random.choices(choices, weights=weights)[0]
        return choice

    def scenario_generator(self) -> pd.DataFrame:
        """
        generates a scenario based on the percentage values that are provided for each installation. The values have
        to be between 0 and 1. 0 = no buildings are equipped with tech, 1 = all buildings are equipped.

        :return: returns a list of all the scenarios IDs for the whole building stock
        """

        dict_of_inputs = {
            "ID_PV": [1 - self.pv_installation_percentage, self.pv_installation_percentage * 0.5,
                      self.pv_installation_percentage * 0.25, self.pv_installation_percentage * 0.25],
            "ID_HotWaterTank": [self.dhw_storage_percentage],
            "ID_Boiler": [self.air_hp_percentage, self.ground_hp_percentage, self.direct_electric_heating_percentage,
                          self.gases_percentage, self.no_heating_percentage],
            "ID_SpaceCoolingTechnology": [self.ac_percentage]

        }
        translation_2_id = {
            "ID_PV": {
                0: [1],  # IDs with no PV
                1: [2, 3],  # PVs with optimal angles
                2: [4, 5],  # PVs east
                3: [6, 7],  # PVs west
            },
            "ID_HotWaterTank": {
                0: [1],  # no DHW
                1: [2, 3]  # with DHW
            },
            "ID_Boiler": {
                0: [2],  # Air HP
                1: [3],  # Ground HP
                2: [1],  # Electric
                3: [5],  # gases
                4: [4],  # no_heating
            },
            "ID_SpaceCoolingTechnology": {
                0: [2],  # without AC
                1: [1]  # with AC
            },
        }
        total_scenarios = pd.DataFrame()
        random.seed(42)
        for _, row in tqdm(self.clustered_building_df.iterrows(), desc=f"creating scenario based on the probabilities"):
            number_buildings = row["number_of_buildings"]
            building_id = row["ID_Building"]
            building_scenario = self.scenario_table.query(f"ID_Building == {building_id}")
            for n_building in range(number_buildings):
                chosen_building_attributes = {}
                for attribute_name, perc_list in dict_of_inputs.items():
                    chosen_id = self.assign_id(list_of_percentages=perc_list)
                    # translate the chosen ID to an ID in the scenario table:
                    # get unique labels for the specific building
                    possible_labels = building_scenario[attribute_name].unique().tolist()
                    # grab the matching number within both lists:
                    ID = list(set(possible_labels).intersection(translation_2_id[attribute_name][chosen_id]))[0]
                    # add the ID and to chosen_building_attributes
                    chosen_building_attributes[attribute_name] = ID

                # select the correct scenario for the chosen IDs:
                query_list = [f"{key}=={value}" for key, value in chosen_building_attributes.items()]
                query = f" and ".join(query_list)

                # in the second round we check if multiple options exist for the SpaceHeatingTank (related to heating
                # system), the Battery, and the Heating Element (related to PV):
                possible_bat_ids = building_scenario.query(query)["ID_Battery"].unique().tolist()
                possible_spacetank_ids = building_scenario.query(query)["ID_SpaceHeatingTank"].unique().tolist()
                possible_heatelement_ids = building_scenario.query(query)["ID_HeatingElement"].unique().tolist()

                # check if multiple scenarios exist:
                # If multiple exist, battery is an option otherwise there is no PV and thus no battery can be chosen.
                if len(possible_bat_ids) > 1:
                    # decide if battery is used
                    if self.assign_id([self.battery_percentage]) == 1:  # battery is used
                        id_battery = list(set(possible_bat_ids).intersection([2, 3]))[0]  # select ID based on building
                    else:
                        id_battery = 1  # battery is not used
                else:
                    _ = self.assign_id(
                        [self.battery_percentage])  # call it to have the same amount of calls for random choice
                    id_battery = 1
                if len(possible_spacetank_ids) > 1:
                    if self.assign_id([self.buffer_storage_percentage]) > 1:
                        id_buffer = list(set(possible_spacetank_ids).intersection([2, 3]))[0]
                    else:
                        id_buffer = 1
                else:
                    _ = self.assign_id(
                        [self.buffer_storage_percentage])  # call it to have the same amount of calls in every run
                    id_buffer = 1
                if len(possible_heatelement_ids) > 1:
                    if self.assign_id([self.heating_element_percentage]) > 1:
                        id_heatelement = list(set(possible_heatelement_ids).intersection([2, 3]))[0]
                    else:
                        id_heatelement = 1
                else:
                    _ = self.assign_id(
                        [self.heating_element_percentage])  # call it to have the same amount of calls for random choice
                    id_heatelement = 1
                full_query = f"{query} and ID_Battery == {id_battery} " \
                             f"and ID_SpaceHeatingTank == {id_buffer} " \
                             f"and ID_HeatingElement == {id_heatelement}"

                scenario = building_scenario.query(full_query)
                total_scenarios = pd.concat([total_scenarios, scenario], axis=0)

        return total_scenarios

    @staticmethod
    def shorten_scenario_list(list_of_scenarios: List[int]):
        """
        find out the doubles in the list
        :return: dictionary containing the scenario ID as key and the amounts of occurrences as value
        """
        print("shortening the scenario list and creating dict...")
        return_dict = {}
        for scen_id in set(list_of_scenarios):
            return_dict[scen_id] = list_of_scenarios.count(scen_id)
        return return_dict

    def worker(self, id_scenario, number_of_occurences) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
        ref_loads = self.db.read_parquet(table_name=OperationTable.ResultRefHour,
                                         scenario_ID=id_scenario,
                                         column_names=["Grid", "Feed2Grid", "E_Heating_HP_out", "E_RoomCooling"])
        # check if optimisation exists for this scenario (dependent on the heating system)
        heating_system = self.scenario_table.loc[self.scenario_table["ID_Scenario"] == id_scenario, "ID_Boiler"].values[
            0]
        # air hp = 2 and ground hp = 3
        if heating_system == 2 or heating_system == 3:
            prosumager_possibility = self.prosumager_percentage
            opt_loads = self.db.read_parquet(table_name=OperationTable.ResultOptHour,
                                             scenario_ID=id_scenario,
                                             column_names=["Grid", "Feed2Grid", "E_Heating_HP_out", "E_RoomCooling"])
        else:
            prosumager_possibility = 0

        multi_index = pd.MultiIndex.from_tuples([(id_scenario, i) for i in range(number_of_occurences)])
        result_matrix_demand = pd.DataFrame(index=range(8760), columns=multi_index)
        result_matrix_feed2grid = pd.DataFrame(index=range(8760), columns=multi_index)
        result_matrix_cooling = pd.DataFrame(index=range(8760), columns=multi_index)
        result_matrix_heating = pd.DataFrame(index=range(8760), columns=multi_index)
        for i in range(number_of_occurences):
            prosumager = self.assign_id([prosumager_possibility])
            if prosumager == 0:
                result_matrix_demand.loc[:, (id_scenario, i)] = ref_loads["Grid"].to_numpy()
                result_matrix_feed2grid.loc[:, (id_scenario, i)] = ref_loads["Feed2Grid"].to_numpy()
                result_matrix_cooling.loc[:, (id_scenario, i)] = ref_loads["E_RoomCooling"].to_numpy()
                result_matrix_heating.loc[:, (id_scenario, i)] = ref_loads["E_Heating_HP_out"].to_numpy()
            else:
                result_matrix_demand.loc[:, (id_scenario, i)] = opt_loads["Grid"].to_numpy()
                result_matrix_feed2grid.loc[:, (id_scenario, i)] = opt_loads["Feed2Grid"].to_numpy()
                result_matrix_cooling.loc[:, (id_scenario, i)] = opt_loads["E_RoomCooling"].to_numpy()
                result_matrix_heating.loc[:, (id_scenario, i)] = opt_loads["E_Heating_HP_out"].to_numpy()

        return result_matrix_demand, result_matrix_feed2grid, result_matrix_heating, result_matrix_cooling

    def calculate_total_load_profiles(self, scenario_dictionary: dict):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.worker, id_scenario, number_of_occurences)
                       for id_scenario, number_of_occurences in scenario_dictionary.items()]

            results_demand = []
            results_feed2grid = []
            results_heating = []
            results_cooling = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="calculating total load profiles"):
                demand, feed2grid, heating, cooling = future.result()
                results_demand.append(demand)
                results_feed2grid.append(feed2grid)
                results_heating.append(heating)
                results_cooling.append(cooling)

        result_matrix_demand = pd.concat(results_demand, axis=1)
        result_matrix_feed2grid = pd.concat(results_feed2grid, axis=1)
        result_matrix_heating = pd.concat(results_heating, axis=1)
        result_matrix_cooling = pd.concat(results_cooling, axis=1)

        return result_matrix_demand, result_matrix_feed2grid, result_matrix_heating, result_matrix_cooling

    @staticmethod
    def find_net_peak_demand_day(data: pd.DataFrame) -> int:
        """
        finds the index of the day with the highest peak demand
        :param profile: numpy array with shape (8760,)
        :return: integer which day has highest demand out of 364 (starting at 0)
        """
        print("searching for net peak demand day")
        # Sum up all columns to get the net demand for each hour
        profile = data.sum(axis=1)
        # Resample the data to get average daily demand
        # ---------------------------------- only for highest average demand day
        # daily_average = profile.groupby(profile.index // 24).mean()
        # Find the day with the highest average demand
        # max_average_demand_day = pd.to_numeric(daily_average, errors="coerce").idxmax()
        # -------------------------------------
        max_hour = pd.to_numeric(profile, errors="coerce").idxmax()
        max_demand_day = max_hour // 24
        return max_demand_day

    def show_chosen_dist(self, df: pd.DataFrame):
        # Create a subplot with shared x-axis
        n_cols = 3
        n_rows = int(np.ceil(len(df.columns) / n_cols))
        fig = make_subplots(rows=n_rows, cols=n_cols)

        # Add histograms to the subplot
        for idx, column in enumerate(df.columns, start=1):
            row = (idx - 1) // n_cols + 1
            col = (idx - 1) % n_cols + 1
            histogram = go.Histogram(x=df[column], name=column, hovertext="number buildings")
            fig.add_trace(histogram, row=row, col=col)

        # Update subplot layout
        fig.update_layout(title="Distribution of Values in Each Column",
                          height=200 * len(df.columns),
                          width=900
                          )
        fig.show()

    def select_max_days(self, demand: pd.DataFrame) -> (int, int):
        max_grid_day_index = self.find_net_peak_demand_day(demand)
        start_hour = max_grid_day_index * 24
        end_hour = (max_grid_day_index + 1) * 24 - 1
        return start_hour, end_hour

    def calculate_contracted_power(self, array: np.array):
        """ contracted power is the maximum demand """
        return max(array.sum(axis=1))

    def calculate_installed_capacity(self, scenario_series: pd.Series, translation_dict: dict) -> float:
        capacity = np.vectorize(translation_dict.get)(scenario_series.to_numpy())
        return capacity.sum()

    def calculate_real_percentage_of_installation(self, df: pd.DataFrame, column_name: str, name_dict: dict):
        if column_name not in df.columns:
            print(f"Column '{column_name}' does not exist in the DataFrame.")
        # Calculate the count of each unique value in the specified column
        value_counts = df[column_name].value_counts()
        # Calculate the percentage of each unique value
        percentages = dict((value_counts / value_counts.sum()) * 100)
        new_dict = {}
        for key, value in name_dict.items():
            try:
                new_dict[f"{value} %"] = percentages[key]
            except:  # 0 % of the variable so it doesnt show up in percentages
                new_dict[f"{value} %"] = 0
        return new_dict

    def save_hourly_csv(self,
                        demand_d_day: pd.DataFrame,
                        demand_f_day: pd.DataFrame,
                        feed2grid_d_day: pd.DataFrame,
                        feed2grid_f_day: pd.DataFrame,
                        ):
        """

        :param demand_d_day: demand on max demand day
        :param demand_f_day: demand on max feed2grid day
        :param feed2grid_d_day:  feed on max demand day
        :param feed2grid_f_day: feed on max feed2grid day
        :param file_name: str
        :return:
        """
        # check if feed2grid is empty and if its is create a zero array:
        if len(feed2grid_d_day) == 0:
            feed2grid_d_day = pd.DataFrame(np.zeros(shape=demand_d_day.shape))
        if len(feed2grid_f_day) == 0:
            feed2grid_f_day = pd.DataFrame(np.zeros(shape=demand_d_day.shape))

        total_data = pd.concat([
            demand_d_day,
            feed2grid_d_day * -1,
            demand_f_day,
            feed2grid_f_day * -1
        ], axis=0)

        total_data.insert(loc=0,
                          column="unit",
                          value="Wh")
        total_data.insert(loc=0,
                          column="type",
                          value=np.concatenate((np.full(shape=(24,), fill_value="grid demand on max demand day"),
                                                np.full(shape=(24,), fill_value="feed to grid on max demand day"),
                                                np.full(shape=(24,), fill_value="grid demand on max feed day"),
                                                np.full(shape=(24,), fill_value="feed to grid on max feed day"),
                                                )
                                               )
                          )
        total_data.insert(loc=0,
                          column="hours",
                          value=np.concatenate(
                              (np.arange(1, 25), np.arange(1, 25), np.arange(1, 25), np.arange(1, 25))))
        return total_data

    def add_real_building_ids(self,
                              scenario_df: pd.DataFrame,
                              total_grid_demand,
                              total_grid_feed,
                              heating_demand,
                              cooling_demand,
                              ):

        # load the table where real IDs are matched to the clustered IDs:
        match = pd.read_excel(self.path_to_project /
                              f"Original_Building_IDs_to_clusters_{self.region}_{self.year}.xlsx",
                              engine="openpyxl")
        # create a dict of the matches:
        match_dict = match.to_dict(orient="list")
        # remove nan from the dict
        for key in match_dict.keys():
            match_dict[key] = [int(x) for x in match_dict[key] if not pd.isna(x)]
        # create a dict that contains the building ID for each scenario ID
        clustered_ids = scenario_df[["ID_Building", "ID_Scenario"]].set_index("ID_Scenario", drop=True).to_dict()[
            "ID_Building"]

        # for each clustered Building ID draw a real building ID from match.
        # To avoid drawing the same building twice use pop
        column_names = list(total_grid_demand.columns)
        for j, (scenario_id, i) in enumerate(column_names):
            # get the building ID for the scenario ID
            cluster_id = clustered_ids[scenario_id]
            # with the building ID (which is the ID of the clustered buildings) select a real building ID
            real_id = int(match_dict[cluster_id].pop())
            # change the column name with the real ID:
            column_names[j] = real_id

        # reset the column ids to the real building IDs: column names are the same in both dataframes
        total_grid_demand.columns = column_names
        total_grid_feed.columns = column_names
        heating_demand.columns = column_names
        cooling_demand.columns = column_names
        return total_grid_demand, total_grid_feed, heating_demand, cooling_demand

    def create_overall_information_csv(self,
                                       scenario_df: pd.DataFrame,
                                       total_grid_demand: pd.DataFrame,
                                       max_demand_day: int,
                                       max_feed_day: int,
                                       file_name: str):
        # create the corresponding overall information csv:
        contracted_power = self.calculate_contracted_power(total_grid_demand)
        total_pv = self.calculate_installed_capacity(scenario_series=scenario_df["ID_PV"],
                                                     translation_dict=self.pv_sizes)
        total_battery = self.calculate_installed_capacity(scenario_series=scenario_df["ID_Battery"],
                                                          translation_dict=self.battery_sizes)
        total_buffer = self.calculate_installed_capacity(scenario_series=scenario_df["ID_SpaceHeatingTank"],
                                                         translation_dict=self.buffer_sizes)
        total_dhw = self.calculate_installed_capacity(scenario_series=scenario_df["ID_HotWaterTank"],
                                                      translation_dict=self.DHW_sizes)

        add_info_df = {
            "Region": self.region,
            "Peak Demand (kWh)": round(contracted_power / 1_000),
            "Installed PV (kWp)": total_pv,
            "Installed Battery (kWh)": total_battery,
            "Installed Buffer (l)": total_buffer,
            "Installed DHW storage (l)": total_dhw,
            "Day of max demand": max_demand_day,
            "Day of max feed to grid": max_feed_day,
        }

        total_cooling = self.calculate_real_percentage_of_installation(df=scenario_df,
                                                                       column_name="ID_SpaceCoolingTechnology",
                                                                       name_dict=self.cooling)
        total_boilers = self.calculate_real_percentage_of_installation(df=scenario_df,
                                                                       column_name="ID_Boiler",
                                                                       name_dict=self.boiler)

        final_info = pd.DataFrame.from_dict(data={**add_info_df, **total_cooling, **total_boilers},
                                            orient="index")
        final_info.to_csv(self.data_output / f"INFO_{file_name}.csv", sep=";")
        print("saved add information csv")

    def __file_name__(self):
        return f"{self.year}_" \
               f"{self.region}_" \
               f"PV-{round(self.pv_installation_percentage * 100)}%_" \
               f"DHW-{round(self.dhw_storage_percentage * 100)}%_" \
               f"Buffer-{round(self.buffer_storage_percentage * 100)}%_" \
               f"HE-{round(self.heating_element_percentage * 100)}%_" \
               f"AirHP-{round(self.air_hp_percentage * 100)}%_" \
               f"GroundHP-{round(self.ground_hp_percentage * 100)}%_" \
               f"directE-{round(self.direct_electric_heating_percentage * 100)}%_" \
               f"Conventional-{round(self.gases_percentage * 100)}%_" \
               f"AC-{round(self.ac_percentage * 100)}%_" \
               f"Battery-{round(self.battery_percentage * 100)}%_" \
               f"Prosumager-{round(self.prosumager_percentage * 100)}%"

    def create_output_csv(self):
        scenarios = self.scenario_generator()
        scenario_list = scenarios["ID_Scenario"].values.tolist()
        scenario_dict = self.shorten_scenario_list(scenario_list)

        total_grid_demand, total_grid_feed, heating, cooling = self.calculate_total_load_profiles(
            scenario_dictionary=scenario_dict
        )
        total_grid_demand_real, total_grid_feed_real, heating_real, cooling_real = self.add_real_building_ids(
            scenario_df=scenarios,
            total_grid_demand=total_grid_demand,
            total_grid_feed=total_grid_feed,
            heating_demand=heating,
            cooling_demand=cooling
        )

        # define filename:
        file_name = self.__file_name__()

        # save the total profiles to parquet
        total_grid_demand_real.columns = total_grid_demand_real.columns.astype(str)
        total_grid_feed_real.columns = total_grid_feed_real.columns.astype(str)
        heating_real.columns = heating_real.columns.astype(str)
        cooling_real.columns = cooling_real.columns.astype(str)
        total_grid_demand_real.to_parquet(self.data_output / f"Demand_{file_name}.parquet.gzip")
        total_grid_feed_real.to_parquet(self.data_output / f"Feed_{file_name}.parquet.gzip")
        heating_real.to_parquet(self.data_output / f"Heating_{file_name}.parquet.gzip")
        cooling_real.to_parquet(self.data_output / f"Cooling_{file_name}.parquet.gzip")
        # save the profiles to csv
        demand_start_hour, demand_end_hour = self.select_max_days(total_grid_demand_real)
        demand_max_demand_day = total_grid_demand_real.loc[demand_start_hour: demand_end_hour, :]
        feed_max_demand_day = total_grid_feed_real.loc[demand_start_hour: demand_end_hour, :]

        feed_start_hour, feed_end_hour = self.select_max_days(total_grid_feed_real)
        demand_max_feed_day = total_grid_demand_real.loc[feed_start_hour: feed_end_hour, :]
        feed_max_feed_day = total_grid_feed_real.loc[feed_start_hour: feed_end_hour, :]

        hourly_df = self.save_hourly_csv(
            demand_d_day=demand_max_demand_day,
            demand_f_day=demand_max_feed_day,
            feed2grid_d_day=feed_max_demand_day,
            feed2grid_f_day=feed_max_feed_day,
        )
        self.add_baseline_data_to_hourly_csv(hourly_df)

        print("saved hourly csv file")

        self.create_overall_information_csv(scenario_df=scenarios,
                                            total_grid_demand=total_grid_demand_real,
                                            max_demand_day=int(demand_start_hour / 24),
                                            max_feed_day=int(feed_start_hour / 24),
                                            file_name=file_name,

                                            )

    def get_hourly_csv(self) -> pd.DataFrame:
        file_name = self.__file_name__()
        path_2_file = self.data_output / f"{file_name}.csv"
        return pd.read_csv(path_2_file, sep=";")

    def get_info_csv(self) -> pd.DataFrame:
        file_name = self.__file_name__()
        path_2_file = self.data_output / f"INFO_{file_name}.csv"
        return pd.read_csv(path_2_file, sep=";")

    def add_baseline_data_to_hourly_csv(self, hourly_csv_scen: pd.DataFrame):
        base = ECEMFPostProcess(**self.baseline, baseline=self.baseline)
        try:  # if baseline does not exist yet:
            hourly_csv_base = base.get_hourly_csv()
            hourly_csv_base["type"] = hourly_csv_base["type"] + " baseline"
            df = pd.concat([hourly_csv_scen, hourly_csv_base], axis=0)
        except:
            df = hourly_csv_scen

        df.to_csv(self.data_output / f"{self.__file_name__()}.csv", sep=";", index=False)


def create_figure_worker(scenario_list: list,
                         demand_feed: str,
                         file_loc: Path,
                         Season: str,
                         changed_parameter: str,
                         output_folder: Path
                         ):
    matplotlib.rc("font", **{"size": 28})
    fig = plt.figure(figsize=(18, 16))
    ax = plt.gca()
    cmap = get_cmap('Set1')
    colors = [cmap(i) for i in np.linspace(0, 1, len(scenario_list))]

    for i, scen in enumerate(scenario_list):

        data = pd.read_parquet(file_loc / f"{demand_feed}_{get_file_name(scen)}.parquet.gzip", engine="pyarrow")
        data_enhanced = get_season_and_datetime(data)
        if changed_parameter in scen.keys():
            nh = f"{round(scen[changed_parameter] * 100)} %"
        else:
            nh = f"{scen['year']}"

        median = data_enhanced.query("season==@Season").drop(columns=["season"]).groupby("hour").median().mean(axis=1)
        # stds = data_enhanced.query("season==@Season").drop(columns=["season"]).groupby("hour").std().mean(axis=1)
        percentile_75 = data_enhanced.query("season==@Season").drop(columns=["season"]).groupby("hour").quantile(
            0.75).mean(axis=1)
        percentile_25 = data_enhanced.query("season==@Season").drop(columns=["season"]).groupby("hour").quantile(
            0.25).mean(axis=1)

        ax.plot(np.arange(24),
                median,
                color=colors[i],
                linewidth=2,
                label=f'{changed_parameter}: {nh} Median'.replace("_", " "),
                )

        ax.fill_between(np.arange(24),
                        percentile_75,
                        percentile_25,
                        alpha=0.3,
                        # label=f'{changed_parameter}: {round(scen[changed_parameter] * 100)}% 25 & 75 percentile'.replace("_", " "),
                        color=colors[i])
    # Formatting the seasonal plot
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel(f'Grid {demand_feed} (Wh)')
    ax.set_title(f'Median Hourly {demand_feed} for {Season.capitalize()}')
    ax.legend(loc="upper left")
    plt.tight_layout()
    if not output_folder.exists():
        # Create the folder
        output_folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_folder / f"{Season.capitalize()}_Daily_Mean_{demand_feed}_{changed_parameter}_Comparison.png")
    plt.close(fig)
    print(
        f"created {Season.capitalize()}_Daily_Median_{demand_feed}_{changed_parameter}_Comparison.png plot")


def get_file_name(dictionary: dict):
    return f"{dictionary['year']}_" \
           f"{dictionary['region']}_" \
           f"PV-{round(dictionary['pv_installation_percentage'] * 100)}%_" \
           f"DHW-{round(dictionary['dhw_storage_percentage'] * 100)}%_" \
           f"Buffer-{round(dictionary['buffer_storage_percentage'] * 100)}%_" \
           f"HE-{round(dictionary['heating_element_percentage'] * 100)}%_" \
           f"AirHP-{round(dictionary['air_hp_percentage'] * 100)}%_" \
           f"GroundHP-{round(dictionary['ground_hp_percentage'] * 100)}%_" \
           f"directE-{round(dictionary['direct_electric_heating_percentage'] * 100)}%_" \
           f"Conventional-{round(dictionary['gases_percentage'] * 100)}%_" \
           f"AC-{round(dictionary['ac_percentage'] * 100)}%_" \
           f"Battery-{round(dictionary['battery_percentage'] * 100)}%_" \
           f"Prosumager-{round(dictionary['prosumager_percentage'] * 100)}%"


def get_season(date):
    seasons = {'spring': pd.date_range(start='2023-03-21 00:00:00', end='2023-06-20 23:00:00', freq="H"),
               'summer': pd.date_range(start='2023-06-21 00:00:00', end='2023-09-22 23:00:00', freq="H"),
               'autumn': pd.date_range(start='2023-09-23 00:00:00', end='2023-12-20 23:00:00', freq="H")}

    if date in seasons['spring']:
        return 'spring'
    elif date in seasons['summer']:
        return 'summer'
    elif date in seasons['autumn']:
        return 'autumn'
    else:
        return 'winter'


def get_season_and_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > 365:
        datetime_index = pd.date_range(start='2023-01-01 00:00:00', periods=8760, freq='H')
    else:
        datetime_index = pd.date_range(start='2023-01-01 ', periods=365, freq='D')
    df.index = datetime_index
    df['season'] = df.index.map(get_season)
    df["hour"] = df.index.hour
    return df.reset_index(drop=True)


def check_if_lists_in_dict_are_same_length(dictionary: dict) -> bool:
    # check that the lists are the same length because each scenario will be one index of the lists:
    value_list = []
    for value in dictionary.values():
        if isinstance(value, list):
            value_list.append(value)
    if any(len(v) != value_list[0] for v in value_list):
        return True
    else:
        return False


class ECEMFFigures:
    def __init__(self, scenario: dict, scenario_name: str = None):
        self.baseline = scenario["baseline"]
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{self.baseline['region']}/data_output/"
        self.path_2_figure = Path(__file__).parent / r"data/figure" / f"ECEMF_T4.3_{self.baseline['region']}"
        nr_lists = 0
        for value in scenario.values():
            if isinstance(value, list):
                nr_lists += 1

        if nr_lists == 1:
            # we have multiple scenarios
            changing_parameter, values = [(key, value) for key, value in scenario.items() if isinstance(value, list)][0]
            self.changing_parameter = changing_parameter
            scenarios = []
            for parameter in values:
                # copy the scenario multiple times:
                new_scen = scenario.copy()
                new_scen[changing_parameter] = parameter
                scenarios.append(new_scen)
            self.scenario = scenarios

        elif nr_lists > 1:
            assert check_if_lists_in_dict_are_same_length(scenario), "lists do not have the same length"
            scenarios = []
            changing_parameters = [(key, value) for key, value in scenario.items() if isinstance(value, list)]
            for i in range(len(changing_parameters[0][1])):
                new_scen = scenario.copy()
                for param, values in changing_parameters:
                    new_scen[param] = values[i]
                scenarios.append(new_scen)
            self.scenario = scenarios
            # if this scenario is chosen, scneario name has to be provided:
            assert scenario_name is not None, "please provide scenario name"
            self.changing_parameter = scenario_name

        else:
            self.scenario = scenario

        self.check_if_scenarios_exist(self.scenario)

    def path_to_gzip(self, filename):
        return Path(self.data_output) / f"{filename}.parquet.gzip"

    def check_if_scenarios_exist(self, scenarios):
        if isinstance(scenarios, list):
            for scen in scenarios:
                # check if baseline csv exists and check if the scenario csv exists. If not, create it:
                if not self.path_to_gzip(f"Demand_{get_file_name(scen)}").exists() or not \
                        self.path_to_gzip(f"Feed_{get_file_name(scen)}").exists():
                    print(
                        f"{get_file_name(scen)} \n parquet files do not exist. create_output_csv"
                    )
                    ECEMFPostProcess(**scen).create_output_csv()
        else:
            if not self.path_to_gzip(f"Demand_{get_file_name(scenarios)}").exists() or not \
                    self.path_to_gzip(f"Feed_{get_file_name(scenarios)}").exists():
                print(
                    f"{get_file_name(scenarios)} \n parquet files do not exist. create_output_csv"
                )
                ECEMFPostProcess(**scenarios).create_output_csv()

        # check if baseline exists
        if not self.path_to_gzip(f"Demand_{get_file_name(self.baseline)}").exists() or not \
                self.path_to_gzip(f"Feed_{get_file_name(self.baseline)}").exists():
            print(
                f"{get_file_name(self.baseline)} \n parquet files do not exist. create_output_csv"
            )
            ECEMFPostProcess(**self.baseline, baseline=self.baseline).create_output_csv()

    def copy_scenario_outputs_into_specific_folder(self):
        """
        copy the csv files of a specific scenario into one folder to make it easier to send to other model
        :return: None
        """
        if isinstance(self.scenario, list):
            destination_path = self.data_output / f"{str(self.changing_parameter).upper()}_{get_file_name(self.scenario[0])}"
            info_destination_path = destination_path / "INFO"
            if not destination_path.exists():
                destination_path.mkdir()
                info_destination_path.mkdir()
            names_to_copy = [get_file_name(self.scenario[i]) for i in range(len(self.scenario))]

            for file in self.data_output.glob("*.csv"):
                for name in names_to_copy:
                    if name in file.stem:
                        if "INFO" in file.stem:
                            shutil.copy(file, info_destination_path)
                        else:
                            shutil.copy(file, destination_path)

    def create_boxplot_p_to_p_difference(self,
                                         baseline: pd.DataFrame,
                                         scenario: pd.DataFrame,
                                         demand_or_feed: str
                                         ):

        # create a plot df in longformat:
        base_p2p = pd.DataFrame(self.peak_to_peak_difference(baseline.sum(axis=1).to_numpy() / 1_000 / 1_000)).rename(
            columns={0: "peak to peak MW"})
        scenario_p2p = pd.DataFrame(
            self.peak_to_peak_difference(scenario.sum(axis=1).to_numpy() / 1_000 / 1_000)).rename(
            columns={0: "peak to peak MW"})

        base_p2p["name"] = "baseline"
        scenario_p2p["name"] = "scenario"
        fig = plt.figure(figsize=(18, 16))
        plot_df = pd.concat([get_season_and_datetime(base_p2p), get_season_and_datetime(scenario_p2p)], axis=0)
        sns.boxplot(data=plot_df,
                    x="season",
                    y="peak to peak MW",
                    hue="name",
                    palette=sns.color_palette("hls", 5),

                    )
        plt.title("daily peak to peak demand")
        plt.tight_layout()
        folder = self.path_2_figure / get_file_name(self.scenario)
        if not folder.exists():
            # Create the folder
            folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / Path(f"Peak_to_peak_{demand_or_feed}.png"))
        plt.close(fig)
        print("created Peak_to_peak boxplot")

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

        return peak_to_peak_difference

    def create_scenario_comparison_daily_mean_plot_for_each_season(self):
        seasons = ["spring", "summer", "autumn", "winter"]
        # for demand_or_feed in ["Demand", "Feed"]:
        #     for season in seasons:
        #         create_figure_worker(
        #              scenario_list=self.scenario,
        #              demand_feed=demand_or_feed,
        #              Season=season,
        #              file_loc=self.data_output,
        #              changed_parameter=self.changing_parameter,
        #              output_folder=self.path_2_figure / "scenario_comparisons"
        #         )
        #         pass

        arguments = [
            (
                self.scenario,
                demand_or_feed,
                self.data_output,
                season,
                self.changing_parameter,
                self.path_2_figure / f"scenario_comparisons_{self.changing_parameter}_{get_file_name(self.scenario[0])}"
            ) for season in seasons for demand_or_feed in ["Demand", "Feed"]
        ]
        joblib.Parallel(n_jobs=8)(joblib.delayed(create_figure_worker)(*args) for args in arguments)

    def get_max_day_feed_and_demand_sum(self) -> (dict, dict, dict, dict):
        """
        loads the dataframes for the max day demand data, and returns two dicts with the baseline, and
        all scenarios that are included in self.scenario. one dict is the grid demand and the other the feed to grid.
        :return:
        """
        demand_dict_d = {}
        feed_dict_d = {}
        demand_dict_f = {}
        feed_dict_f = {}
        nr_lists = 0
        for value in scenario.values():
            if isinstance(value, list):
                nr_lists += 1

        if nr_lists == 0:
            data = pd.read_csv(self.data_output / f"{get_file_name(self.scenario)}.csv", sep=";")
            demand_dict_d["scenario"] = data.query("type=='grid demand on max demand day'").iloc[:, 3:]
            feed_dict_d["scenario"] = data.query("type=='feed to grid on max demand day'").iloc[:, 3:]
            demand_dict_f["scenario"] = data.query("type=='grid demand on max feed day'").iloc[:, 3:]
            feed_dict_f["scenario"] = data.query("type=='feed to grid on max feed day'").iloc[:, 3:]
        elif nr_lists == 1:
            for scen in self.scenario:
                data = pd.read_csv(self.data_output / f"{get_file_name(scen)}.csv", sep=";")
                demand_dict_d[f"{self.changing_parameter}_{round(scen[self.changing_parameter] * 100)} %"] = data.query(
                    "type=='grid demand on max demand day'"
                ).iloc[:, 3:]
                feed_dict_d[f"{self.changing_parameter}_{round(scen[self.changing_parameter] * 100)} %"] = data.query(
                    "type=='feed to grid on max demand day'"
                ).iloc[:, 3:]
                demand_dict_f[f"{self.changing_parameter}_{round(scen[self.changing_parameter] * 100)} %"] = data.query(
                    "type=='grid demand on max feed day'"
                ).iloc[:, 3:]
                feed_dict_f[f"{self.changing_parameter}_{round(scen[self.changing_parameter] * 100)} %"] = data.query(
                    "type=='feed to grid on max feed day'"
                ).iloc[:, 3:]
        else:
            for scen in self.scenario:
                data = pd.read_csv(self.data_output / f"{get_file_name(scen)}.csv", sep=";")
                demand_dict_d[f"{self.changing_parameter}_{scen['year']}"] = data.query(
                    "type=='grid demand on max demand day'"
                ).iloc[:, 3:]
                feed_dict_d[f"{self.changing_parameter}_{scen['year']}"] = data.query(
                    "type=='feed to grid on max demand day'"
                ).iloc[:, 3:]
                demand_dict_f[f"{self.changing_parameter}_{scen['year']}"] = data.query(
                    "type=='grid demand on max feed day'"
                ).iloc[:, 3:]
                feed_dict_f[f"{self.changing_parameter}_{scen['year']}"] = data.query(
                    "type=='feed to grid on max feed day'"
                ).iloc[:, 3:]

        baseline = pd.read_csv(self.data_output / f"{get_file_name(self.baseline)}.csv", sep=";")
        demand_dict_d["baseline"] = baseline.query("type=='grid demand on max demand day'").iloc[:, 3:]
        feed_dict_d["baseline"] = baseline.query("type=='feed to grid on max demand day'").iloc[:, 3:]
        demand_dict_f["baseline"] = baseline.query("type=='grid demand on max feed day'").iloc[:, 3:]
        feed_dict_f["baseline"] = baseline.query("type=='feed to grid on max feed day'").iloc[:, 3:]

        return demand_dict_d, feed_dict_d, demand_dict_f, feed_dict_f

    def visualize_peak_day(self):
        """
        creates plot of grid demand and feed of baseline and all scenarios
        :return:
        """
        if isinstance(self.scenario, dict):
            folder = self.path_2_figure / get_file_name(self.scenario)
            region = ECEMFPostProcess(**self.scenario).region
        else:
            folder = self.path_2_figure / f"scenario_comparisons_{self.changing_parameter}_{get_file_name(self.scenario[0])}"
            region = ECEMFPostProcess(**self.scenario[0]).region
        if not folder.exists():
            # Create the folder
            folder.mkdir(parents=True, exist_ok=True)

        demand_dict_max_demand, feed_dict_max_demand, demand_dict_max_feed, feed_dict_max_feed = self.get_max_day_feed_and_demand_sum()
        matplotlib.rc("font", **{"size": 36})
        for key, scen_dict in {"Demand": {"demand": demand_dict_max_demand,
                                          "feed": feed_dict_max_demand},
                               "Feed": {"demand": demand_dict_max_feed,
                                        "feed": feed_dict_max_feed}
                               }.items():
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(36, 18))
            cmap = get_cmap('tab20c')
            colors = [cmap(i) for i in np.linspace(0, 1, len(scen_dict["demand"]))]
            for i, (scen_name, df) in enumerate(scen_dict["demand"].items()):
                if scen_name == "baseline":
                    line_style = "--"
                else:
                    line_style = "-"
                ax1.plot(
                    np.arange(1, 25),
                    df.sum(axis=1) / 1_000 / 1_000,  # GWh
                    color=colors[i],
                    linewidth=2,
                    label=f'{scen_name.replace("_", " ")}',
                    linestyle=line_style
                )

            for i, (scen_name, df) in enumerate(scen_dict["feed"].items()):
                if scen_name == "baseline":
                    line_style = "--"
                else:
                    line_style = "-"
                ax2.plot(
                    np.arange(1, 25),
                    df.sum(axis=1) / 1_000 / 1_000 * -1,  # GWh
                    color=colors[i],
                    linewidth=2,
                    label=f'{scen_name.replace("_", " ")}',
                    linestyle=line_style
                )

            ax1.set_xlabel('Hour of Day')
            ax2.set_xlabel('Hour of Day')
            ax1.set_ylabel(f'Grid demand (MW)')
            ax2.set_ylabel(f'Feed to grid (MW)')
            ax1.set_title(f'Total demand for {region} on peak {key} day')
            ax2.set_title(f'Total feed to grid for {region} on peak {key} day')
            ax1.legend(loc="upper left")
            ax2.legend(loc="lower left")

            # plt.xticks(range(0, 24))
            # ax.grid(True)
            plt.tight_layout()

            fig.savefig(folder / f"Max_Day_{key}_Comparison.png")
            plt.close(fig)

    def create_figures(self):
        print("creating figures...")
        self.visualize_peak_day()
        if isinstance(self.scenario, dict):
            # load the data from gzip files
            demand = pd.read_parquet(self.path_to_gzip(f"Demand_{get_file_name(self.scenario)}"), engine="pyarrow")
            feed = pd.read_parquet(self.path_to_gzip(f"Feed_{get_file_name(self.scenario)}"), engine="pyarrow")
            base_demand = pd.read_parquet(self.path_to_gzip(f"Demand_{get_file_name(self.baseline)}"), engine="pyarrow")
            base_feed = pd.read_parquet(self.path_to_gzip(f"Feed_{get_file_name(self.baseline)}"), engine="pyarrow")

            # create the figures
            self.create_boxplot_p_to_p_difference(baseline=base_demand, scenario=demand, demand_or_feed="Demand")
            self.plot_seasonal_daily_means(df_baseline=base_demand, df_scenario=demand, demand_or_feed="Demand")
            self.plot_seasonal_daily_means(df_baseline=base_feed, df_scenario=feed, demand_or_feed="Feed")
        else:  # self.scenario is a list of scenarios:
            # create a graph that shows the difference between the scenarios:
            # load data
            self.visualize_heating_and_cooling_load()
            self.create_scenario_comparison_daily_mean_plot_for_each_season()

    def visualize_heating_and_cooling_load(self):
        fig_path = self.path_2_figure / f"scenario_comparisons_{self.changing_parameter}_{get_file_name(self.scenario[0])}"
        heating_dict = {}
        cooling_dict = {}
        for i, scen in enumerate(self.scenario):
            heating = pd.read_parquet(self.data_output / f"Heating_{get_file_name(scen)}.parquet.gzip",
                                      engine="pyarrow")
            cooling = pd.read_parquet(self.data_output / f"Cooling_{get_file_name(scen)}.parquet.gzip",
                                      engine="pyarrow")
            if self.changing_parameter in scen.keys():
                nh = f"{round(scen[self.changing_parameter] * 100)} %"
            else:
                nh = f"{scen['year']}"
            heating_dict[
                f"{self.changing_parameter}_{nh}"] = heating.sum().sum() / 1000 / 1000  # MWh
            cooling_dict[
                f"{self.changing_parameter}_{nh}"] = cooling.sum().sum() / 1000 / 1000  # MWh

        x_heating, y_heating = list(heating_dict.keys()), list(heating_dict.values())
        x_cooling, y_cooling = list(cooling_dict.keys()), list(cooling_dict.values())

        fig = plt.figure(figsize=(16, 20))
        ax = plt.gca()
        ax.plot(x_heating, y_heating, label="heating", color="firebrick")
        ax.plot(x_cooling, y_cooling, label="cooling", color="blue")
        plt.legend()
        ax.set_xlabel("scenario")
        ax.set_ylabel("demand in MWh")
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(fig_path / "Total_Heating_Cooling.png")

    def plot_seasonal_daily_means(self,
                                  df_baseline: pd.DataFrame,
                                  df_scenario: pd.DataFrame,
                                  demand_or_feed: str
                                  ):
        # add seasons to df:
        season_groups_baseline = get_season_and_datetime(df_baseline).groupby("season")
        season_groups_scenario = get_season_and_datetime(df_scenario).groupby("season")
        folder = self.path_2_figure / get_file_name(self.scenario)

        # Separate plots for each season
        seasons = ["spring", "summer", "autumn", "winter"]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(26, 22), sharey=True)
        axes = axes.flatten()
        for i, season in enumerate(seasons):
            ax = axes[i]
            season_baseline = season_groups_baseline.get_group(season)
            season_scenario = season_groups_scenario.get_group(season)
            # Filter the dataframe for the season and aggregate data by hour
            # seasonal_hourly_means_baseline = season_baseline.drop(columns=["season"]).groupby("hour").mean().mean(axis=1)
            seasonal_hourly_median_baseline = season_baseline.drop(columns=["season"]).groupby("hour").median().mean(
                axis=1)
            seasonal_hourly_pec_75_baseline = season_baseline.drop(columns=["season"]).groupby("hour").quantile(
                0.75).mean(axis=1)
            seasonal_hourly_pec_25_baseline = season_baseline.drop(columns=["season"]).groupby("hour").quantile(
                0.25).mean(axis=1)

            # seasonal_hourly_means_scenario = season_scenario.drop(columns=["season"]).groupby("hour").mean().mean(axis=1)
            seasonal_hourly_median_scenario = season_scenario.drop(columns=["season"]).groupby("hour").median().mean(
                axis=1)
            seasonal_hourly_pec_75_scenario = season_scenario.drop(columns=["season"]).groupby("hour").quantile(
                0.75).mean(axis=1)
            seasonal_hourly_pec_25_scenario = season_scenario.drop(columns=["season"]).groupby("hour").quantile(
                0.25).mean(axis=1)

            # Plot seasonal mean and standard deviation
            ax.plot(np.arange(24),
                    seasonal_hourly_median_baseline,
                    color="blue",
                    linewidth=2,
                    label=f'{season.capitalize()} Median baseline',
                    )
            ax.fill_between(np.arange(24),
                            seasonal_hourly_pec_75_baseline,
                            seasonal_hourly_pec_25_baseline,
                            alpha=0.3,
                            # label=f'{season.capitalize()} 25 & 75 percentile',
                            color="cyan")

            ax.plot(np.arange(24),
                    seasonal_hourly_median_scenario,
                    color="red",
                    linewidth=2,
                    label=f'{season.capitalize()} Median scenario',
                    )
            ax.fill_between(np.arange(24),
                            seasonal_hourly_pec_75_scenario,
                            seasonal_hourly_pec_25_scenario,
                            alpha=0.3,
                            # label=f'{season.capitalize()} 25 & 75 percentile',
                            color="lightcoral")

            # Formatting the seasonal plot
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel(f'Grid {demand_or_feed} (Wh)')
            ax.set_title(f'Median grid {demand_or_feed} for {season.capitalize()}')
            ax.legend(loc="upper left")
            # plt.xticks(range(0, 24))
            # ax.grid(True)
        plt.tight_layout()
        if not folder.exists():
            # Create the folder
            folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"Daily_Median_{demand_or_feed}_Comparison.png")
        plt.close(fig)
        print(f"created Daily Median {demand_or_feed} plot")


if __name__ == "__main__":
    # ECEMFBuildingComparison(region="Murcia").main()

    # Battery is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating element is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating buffer storage is only installed in buildings with HPs. Probability only refers to buildings with HP
    baseline = {
        "year": 2020,
        "region": "Murcia",
        "pv_installation_percentage": 0.015,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.2,
        "ground_hp_percentage": 0,
        "direct_electric_heating_percentage": 0.39,
        "gases_percentage": 0.19,
        "ac_percentage": 0.5,
        "battery_percentage": 0.1,
        "prosumager_percentage": 0,
    }
    pv_increase = np.arange(0.1, 1.1, 0.1).tolist()
    prosumager_increase = np.arange(0, 1.1, 0.1).tolist()
    direct_electric_heating_increase = np.arange(0, 0.45, 0.05).tolist()
    AC_increase = np.arange(0, 1.1, 0.1).tolist()
    air_hp_increase = np.arange(0, 0.7, 0.1).tolist()

    # to only analyse one parameter use ECEMFFigures wiht on parameter as list:
    # scenario = {
    #     "year": 2020,
    #     "region": "Murcia",
    #     "pv_installation_percentage": pv_increase,
    #     "dhw_storage_percentage": 0.5,
    #     "buffer_storage_percentage": 0,
    #     "heating_element_percentage": 0,
    #     "air_hp_percentage": 0.2,
    #     "ground_hp_percentage": 0,
    #     "direct_electric_heating_percentage": 0.39,
    #     "gases_percentage": 0.19,
    #     "ac_percentage": 0.5,
    #     "battery_percentage": 0.1,
    #     "prosumager_percentage": 0,
    #     "baseline": baseline
    # }
    #
    # ECEMFFigures(scenario=scenario).create_figures()
    # ECEMFFigures(scenario=scenario).copy_scenario_outputs_into_specific_folder()

    # calculate a complete scenario run:
    scenario = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Murcia",
        "pv_installation_percentage": [0.015, 0.1, 0.4, 0.6],
        "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
        "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
        "heating_element_percentage": 0,
        "air_hp_percentage": [0.2, 0.3, 0.5, 0.7],
        "ground_hp_percentage": [0, 0.02, 0.04, 0.06],
        "direct_electric_heating_percentage": [0.39, 0.3, 0.2, 0.1],
        "gases_percentage": [0.19, 0.15, 0.07, 0.02],
        "ac_percentage": [0.5, 0.6, 0.8, 0.9],
        "battery_percentage": [0.1, 0.12, 0.2, 0.3],
        "prosumager_percentage": [0, 0.05, 0.2, 0.5],
        "baseline": baseline
    }
    scen_name = "no_change"
    # ECEMFFigures(scenario=scenario, scenario_name=scen_name).create_figures()
    ECEMFFigures(scenario=scenario, scenario_name=scen_name).copy_scenario_outputs_into_specific_folder()



    # scenario = {
    #     "year": [2020, 2030, 2040, 2050],
    #     "region": "Murcia",
    #     "pv_installation_percentage": [0.015, 0.15, 0.3, 0.5],
    #     "dhw_storage_percentage": 0.5,
    #     "buffer_storage_percentage": 0,
    #     "heating_element_percentage": 0,
    #     "air_hp_percentage": [0.2, 0.3, 0.5, 0.7],
    #     "ground_hp_percentage": [0, 0.05, 0.1, 0.15],
    #     "direct_electric_heating_percentage": [0.39, 0.3, 0.2, 0.1],
    #     "gases_percentage": [0.19, 0.15, 0.1, 0.05],
    #     "ac_percentage": [0.5, 0.65, 0.8, 1],
    #     "battery_percentage": 0.1,
    #     "prosumager_percentage": [0, 0.1, 0.3, 0.5],
    #     "baseline": baseline
    # }
    # todo check that buildings are not switching place!! Building ID has to stay the same
    # todo send miguel sample scenario where nothing changes but building renovation.
