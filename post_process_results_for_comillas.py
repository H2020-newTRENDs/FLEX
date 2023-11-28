from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import List
import plotly.express as px
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import seaborn as sns

from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable


class ECEMFBuildingComparison:
    def __init__(self, region: str):
        self.region = region
        self.path_to_osm = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM")
        self.path_to_project = Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects") / f"ECEMF_T4.3_{region}"
        self.original_building_ids = self.load_original_building_ids()
        self.clustered_building_df = self.load_clustered_building_df()
        self.original_building_df = self.load_original_building_df()
        self.db = DB(config=Config(project_name=f"ECEMF_T4.3_{region}"))
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
                 region: str,
                 pv_installation_percentage: float,
                 dhw_storage_percentage: float,
                 buffer_storage_percentage: float,
                 heating_element_percentage: float,
                 air_hp_percentage: float,
                 ground_hp_percentage: float,
                 direct_electric_heating_percentage: float,
                 ac_percentage: float,
                 battery_percentage: float,
                 prosumager_percentage: float,
                 ):
        """

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
        self.region = region
        self.path_to_project = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{region}"
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{region}/data_output/"
        self.clustered_building_df = self.load_clustered_building_df()
        self.db = DB(config=Config(project_name=f"ECEMF_T4.3_{region}"))
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
        self.baseline = baseline
        self.pv_installation_percentage = pv_installation_percentage
        self.dhw_storage_percentage = dhw_storage_percentage
        self.buffer_storage_percentage = buffer_storage_percentage
        self.heating_element_percentage = heating_element_percentage
        self.air_hp_percentage = air_hp_percentage
        self.ground_hp_percentage = ground_hp_percentage
        self.direct_electric_heating_percentage = direct_electric_heating_percentage
        self.ac_percentage = ac_percentage
        self.battery_percentage = battery_percentage
        self.prosumager_percentage = prosumager_percentage
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
        assert self.air_hp_percentage + self.ground_hp_percentage + self.direct_electric_heating_percentage <= 1

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

        gases_percentage = 1 - (
                self.air_hp_percentage + self.ground_hp_percentage + self.direct_electric_heating_percentage)
        dict_of_inputs = {
            "ID_PV": [1 - self.pv_installation_percentage, self.pv_installation_percentage * 0.5,
                      self.pv_installation_percentage * 0.25, self.pv_installation_percentage * 0.25],
            "ID_HotWaterTank": [self.dhw_storage_percentage],
            "ID_Boiler": [self.air_hp_percentage, self.ground_hp_percentage, self.direct_electric_heating_percentage,
                          gases_percentage],
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
                3: [4]  # gases
            },
            "ID_SpaceCoolingTechnology": {
                0: [2],  # without AC
                1: [1]  # with AC
            },
        }
        total_scenarios = pd.DataFrame()
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
                    id_battery = 1
                if len(possible_spacetank_ids) > 1:
                    if self.assign_id([self.buffer_storage_percentage]) > 1:
                        id_buffer = list(set(possible_spacetank_ids).intersection([2, 3]))[0]
                    else:
                        id_buffer = 1
                else:
                    id_buffer = 1
                if len(possible_heatelement_ids) > 1:
                    if self.assign_id([self.heating_element_percentage]) > 1:
                        id_heatelement = list(set(possible_heatelement_ids).intersection([2, 3]))[0]
                    else:
                        id_heatelement = 1
                else:
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

    def load_all_load_profiles(self,
                               ):
        pass

    def worker(self, id_scenario, number_of_occurences) -> (pd.DataFrame, pd.DataFrame):
        ref_loads = self.db.read_parquet(table_name=OperationTable.ResultRefHour,
                                         scenario_ID=id_scenario,
                                         column_names=["Load", "Feed2Grid"])
        opt_loads = self.db.read_parquet(table_name=OperationTable.ResultOptHour,
                                         scenario_ID=id_scenario,
                                         column_names=["Load", "Feed2Grid"])

        multi_index = pd.MultiIndex.from_tuples([(id_scenario, i) for i in range(number_of_occurences)])
        result_matrix_demand = pd.DataFrame(index=range(8760), columns=multi_index)
        result_matrix_feed2grid = pd.DataFrame(index=range(8760), columns=multi_index)
        for i in range(number_of_occurences):
            prosumager = self.assign_id([self.prosumager_percentage])
            if prosumager == 0:
                result_matrix_demand.loc[:, (id_scenario, i)] = ref_loads["Load"].to_numpy()
                result_matrix_feed2grid.loc[:, (id_scenario, i)] = ref_loads["Feed2Grid"].to_numpy()
            else:
                result_matrix_demand.loc[:, (id_scenario, i)] = opt_loads["Load"].to_numpy()
                result_matrix_feed2grid.loc[:, (id_scenario, i)] = opt_loads["Feed2Grid"].to_numpy()

        return result_matrix_demand, result_matrix_feed2grid

    def calculate_total_load_profiles(self, scenario_dictionary: dict):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.worker, id_scenario, number_of_occurences)
                       for id_scenario, number_of_occurences in scenario_dictionary.items()]

            results_demand = []
            results_feed2grid = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="calculating total load profiles"):
                demand, feed2grid = future.result()
                results_demand.append(demand)
                results_feed2grid.append(feed2grid)

        result_matrix_demand = pd.concat(results_demand, axis=1)
        result_matrix_feed2grid = pd.concat(results_feed2grid, axis=1)

        return result_matrix_demand, result_matrix_feed2grid

    @staticmethod
    def find_net_peak_demand_day(data: pd.DataFrame) -> int:
        """
        finds the index of the day with the highest average demand
        :param profile: numpy array with shape (8760,)
        :return: integer which day has highest demand out of 364 (starting at 0)
        """
        print("searching for net peak demand day")
        # Sum up all columns to get the net demand for each hour
        profile = data.sum(axis=1)
        # Resample the data to get average daily demand
        daily_average = profile.groupby(profile.index // 24).mean()
        # Find the day with the highest average demand
        max_average_demand_day = daily_average.idxmax()
        return max_average_demand_day

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

    def select_max_days(self, demand: pd.DataFrame) -> pd.DataFrame:
        max_grid_day_index = ecemf.find_net_peak_demand_day(demand)
        start_hour = max_grid_day_index * 24
        end_hour = (max_grid_day_index + 1) * 24 - 1
        selected_day_demand = demand.loc[start_hour: end_hour, :]
        return selected_day_demand

    def calculate_contracted_power(self, array: np.array):
        """ contracted power is the maximum demand """
        return max(array.sum(axis=1))

    def calculate_installed_capacity(self, scenario_series: pd.Series, translation_dict: dict) -> float:
        capacity = np.vectorize(translation_dict.get)(scenario_series.to_numpy())
        return capacity.sum()

    def save_hourly_csv(self,
                        demand: pd.DataFrame,
                        feed2grid: pd.DataFrame,
                        file_name: str):
        # check if feed2grid is empty and if its is create a zero array:
        if len(feed2grid) == 0:
            feed2grid = pd.DataFrame(np.zeros(shape=demand.shape))
        total_data = pd.concat([demand, feed2grid * -1], axis=0)
        total_data.insert(loc=0,
                          column="unit",
                          value="Wh")
        total_data.insert(loc=0,
                          column="type",
                          value=np.concatenate((np.full(shape=(24,), fill_value="grid demand"),
                                                np.full(shape=(24,), fill_value="feed to grid")))
                          )
        total_data.insert(loc=0,
                          column="hours",
                          value=np.concatenate((np.arange(1, 25), np.arange(1, 25))))
        total_data.to_csv(self.data_output / f"{file_name}.csv", sep=";", index=False)
        print("saved hourly csv file")

    def add_real_building_ids(self,
                              scenario_df: pd.DataFrame,
                              total_grid_demand,
                              total_grid_feed):

        # load the table where real IDs are matched to the clustered IDs:
        match = pd.read_excel(self.path_to_project / f"Original_Building_IDs_to_clusters_{self.region}.xlsx",
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
        return total_grid_demand, total_grid_feed

    def create_overall_information_csv(self,
                                       scenario_df: pd.DataFrame,
                                       total_grid_demand: pd.DataFrame,
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
        add_info_df = pd.DataFrame.from_dict(data={
            "Region": self.region,
            "Peak Demand (kWh)": round(contracted_power / 1_000),
            "Installed PV (kWp)": total_pv,
            "Installed Battery (kWh)": total_battery,
            "Installed Buffer (l)": total_buffer,
            "Installed DHW storage (l)": total_dhw
        }, orient="index")
        add_info_df.to_csv(self.data_output / f"INFO_{file_name}.csv", sep=";")
        print("saved add information csv")

    def __file_name__(self):
        return f"{self.region}_" \
               f"PV-{round(self.pv_installation_percentage * 100)}%_" \
               f"DHW-{round(self.dhw_storage_percentage * 100)}%_" \
               f"Buffer-{round(self.buffer_storage_percentage * 100)}%_" \
               f"HE-{round(self.heating_element_percentage * 100)}%_" \
               f"AirHP-{round(self.air_hp_percentage * 100)}%_" \
               f"GroundHP-{round(self.ground_hp_percentage * 100)}%_" \
               f"directE-{round(self.direct_electric_heating_percentage * 100)}%_" \
               f"AC-{round(self.ac_percentage * 100)}%_" \
               f"Battery-{round(self.battery_percentage * 100)}%_" \
               f"Prosumager-{round(self.prosumager_percentage * 100)}%"

    def create_output_csv(self):
        scenarios = self.scenario_generator()
        scenario_list = scenarios["ID_Scenario"].values.tolist()
        scenario_dict = self.shorten_scenario_list(scenario_list)

        total_grid_demand, total_grid_feed = ecemf.calculate_total_load_profiles(scenario_dictionary=scenario_dict)
        total_grid_demand_real, total_grid_feed_real = self.add_real_building_ids(scenario_df=scenarios,
                                                                                  total_grid_demand=total_grid_demand,
                                                                                  total_grid_feed=total_grid_feed)

        # define filename:
        file_name = self.__file_name__()

        # save the profiles to csv
        max_day_demand = self.select_max_days(total_grid_demand_real)
        max_day_feed2grid = self.select_max_days(total_grid_feed_real)
        self.save_hourly_csv(demand=max_day_demand, feed2grid=max_day_feed2grid, file_name=file_name)

        self.create_overall_information_csv(scenario_df=scenarios,
                                            total_grid_demand=total_grid_demand_real,
                                            file_name=file_name)


class ECEMFFigures:
    def __init__(self, baseline_scenario: dict, scenario: dict):
        self.baseline = baseline_scenario
        self.scenario = scenario
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{baseline_scenario['region']}/data_output/"
        self.path_2_figure = Path(__file__).parent / r"data/figure" / f"ECEMF_T4.3_{baseline_scenario['region']}"

    @staticmethod
    def __file_name__(dictionary: dict):
        return f"{dictionary['region']}_" \
               f"PV-{round(dictionary['pv_installation_percentage'] * 100)}%_" \
               f"DHW-{round(dictionary['dhw_storage_percentage'] * 100)}%_" \
               f"Buffer-{round(dictionary['buffer_storage_percentage'] * 100)}%_" \
               f"HE-{round(dictionary['heating_element_percentage'] * 100)}%_" \
               f"AirHP-{round(dictionary['air_hp_percentage'] * 100)}%_" \
               f"GroundHP-{round(dictionary['ground_hp_percentage'] * 100)}%_" \
               f"directE-{round(dictionary['direct_electric_heating_percentage'] * 100)}%_" \
               f"AC-{round(dictionary['ac_percentage'] * 100)}%_" \
               f"Battery-{round(dictionary['battery_percentage'] * 100)}%_" \
               f"Prosumager-{round(dictionary['prosumager_percentage'] * 100)}%"

    def create_boxplot_p_to_p_difference(self,
                                         baseline: pd.DataFrame,
                                         scenario: pd.DataFrame,
                                         demand_or_feed: str
                                         ):

        # create a plot df in longformat:
        base_p2p = pd.DataFrame(self.peak_to_peak_difference(baseline.sum(axis=1).to_numpy() / 1_000 / 1_000)).rename(columns={0: "peak to peak MW"})
        scenario_p2p = pd.DataFrame(self.peak_to_peak_difference(scenario.sum(axis=1).to_numpy() / 1_000 / 1_000)).rename(columns={0: "peak to peak MW"})

        base_p2p["name"] = "baseline"
        scenario_p2p["name"] = "scenario"
        plot_df = pd.concat([self.get_season_and_datetime(base_p2p), self.get_season_and_datetime(scenario_p2p)], axis=0)

        sns.boxplot(data=plot_df,
                    x="season",
                    y="peak to peak MW",
                    hue="name",
                    palette=sns.color_palette("hls", 5)
                    )
        plt.title("daily peak to peak demand")
        plt.tight_layout()
        folder = self.path_2_figure / self.__file_name__(self.scenario)
        if not folder.exists():
            # Create the folder
            folder.mkdir(parents=True, exist_ok=True)
        plt.savefig(folder / Path(f"Peak_to_peak_{demand_or_feed}.png"))

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

    def create_figures(self):
        def path_to_gzip(filename):
            return Path(self.data_output) / f"{filename}.parquet.gzip"

        # check if baseline csv exists and check if the scenario csv exists. If not, create it:
        assert path_to_gzip(f"Demand_{self.__file_name__(self.scenario)}").exists() or \
               path_to_gzip(f"Feed_{self.__file_name__(self.scenario)}").exists(), \
            f"{self.__file_name__(self.scenario)} \n parquet files do not exist. create_output_csv need to be run on " \
            f"the server again."

        assert path_to_gzip(f"Demand_{self.__file_name__(self.baseline)}").exists() or \
               path_to_gzip(f"Feed_{self.__file_name__(self.baseline)}").exists(), \
            f"{self.__file_name__(self.baseline)} \n parquet files do not exist. create_output_csv need to be run on " \
            f"the server again. "

        if self.__file_name__(self.scenario) == self.__file_name__(self.baseline):
            # plot only baseline scenario
            pass
        else:
            demand = pd.read_parquet(path_to_gzip(f"Demand_{self.__file_name__(self.scenario)}"), engine="pyarrow")
            feed = pd.read_parquet(path_to_gzip(f"Feed_{self.__file_name__(self.scenario)}"), engine="pyarrow")
            base_demand = pd.read_parquet(path_to_gzip(f"Demand_{self.__file_name__(self.baseline)}"), engine="pyarrow")
            base_feed = pd.read_parquet(path_to_gzip(f"Feed_{self.__file_name__(self.baseline)}"), engine="pyarrow")

            self.create_boxplot_p_to_p_difference(baseline=base_demand, scenario=demand, demand_or_feed="Demand")
            self.plot_seasonal_daily_means(df_baseline=base_demand, df_scenario=demand, demand_or_feed="Demand")
            self.plot_seasonal_daily_means(df_baseline=base_feed, df_scenario=feed, demand_or_feed="Feed")

    @staticmethod
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

    def get_season_and_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        if len(df) > 365:
            datetime_index = pd.date_range(start='2023-01-01 00:00:00', periods=8760, freq='H')
        else:
            datetime_index = pd.date_range(start='2023-01-01 ', periods=365, freq='D')
        df.index = datetime_index
        df['season'] = df.index.map(self.get_season)
        df["hour"] = df.index.hour
        return df.reset_index(drop=True)

    def plot_seasonal_daily_means(self,
                                  df_baseline: pd.DataFrame,
                                  df_scenario: pd.DataFrame,
                                  demand_or_feed: str
                                  ):
        # add seasons to df:
        season_groups_baseline = self.get_season_and_datetime(df_baseline).groupby("season")
        season_groups_scenario = self.get_season_and_datetime(df_scenario).groupby("season")

        # Separate plots for each season
        seasons = ["spring", "summer", "autumn", "winter"]
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 6), sharey=True)
        axes = axes.flatten()
        for i, season in enumerate(seasons):
            ax = axes[i]
            season_baseline = season_groups_baseline.get_group(season)
            season_scenario = season_groups_scenario.get_group(season)
            # Filter the dataframe for the season and aggregate data by hour
            seasonal_hourly_means_baseline = season_baseline.groupby("hour").mean().mean(axis=1)
            seasonal_hourly_std_baseline = season_baseline.groupby("hour").std().mean(axis=1)

            seasonal_hourly_means_scenario = season_scenario.groupby("hour").mean().mean(axis=1)
            seasonal_hourly_std_scenario = season_scenario.groupby("hour").std().mean(axis=1)

            # Plot seasonal mean and standard deviation
            ax.plot(np.arange(24),
                    seasonal_hourly_means_baseline,
                    color="blue",
                    linewidth=2,
                    label=f'{season.capitalize()} Mean baseline',
                    )
            ax.fill_between(np.arange(24),
                            seasonal_hourly_means_baseline - seasonal_hourly_std_baseline,
                            seasonal_hourly_means_baseline + seasonal_hourly_std_baseline,
                            alpha=0.3,
                            label=f'{season.capitalize()} Std Dev baseline',
                            color="cyan")

            ax.plot(np.arange(24),
                    seasonal_hourly_means_scenario,
                    color="red",
                    linewidth=2,
                    label=f'{season.capitalize()} Mean scenario',
                    )
            ax.fill_between(np.arange(24),
                            seasonal_hourly_means_scenario - seasonal_hourly_std_scenario,
                            seasonal_hourly_means_scenario + seasonal_hourly_std_scenario,
                            alpha=0.3,
                            label=f'{season.capitalize()} Std Dev scenario',
                            color="lightcoral")

            # Formatting the seasonal plot
            ax.set_xlabel('Hour of Day')
            ax.set_ylabel('Mean Profile Value')
            ax.set_title(f'Mean Hourly Profile for {season.capitalize()}')
            ax.legend()
            # plt.xticks(range(0, 24))
            # ax.grid(True)
        plt.tight_layout()
        folder = self.path_2_figure / self.__file_name__(self.scenario)
        if not folder.exists():
            # Create the folder
            folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(folder / f"Daily_Mean_{demand_or_feed}_Comparison.png")
        plt.close(fig)


if __name__ == "__main__":
    # ECEMFBuildingComparison(region="Murcia").main()

    # Battery is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating element is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating buffer storage is only installed in buildings with HPs. Probability only refers to buildings with HP
    baseline = {
        "region": "Murcia",
        "pv_installation_percentage": 0.05,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.08,
        "ground_hp_percentage": 0,
        "direct_electric_heating_percentage": 0.5,
        "ac_percentage": 0.1,
        "battery_percentage": 0.1,
        "prosumager_percentage": 0,
    }
    scenario = {
        "region": "Murcia",
        "pv_installation_percentage": 0.5,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0.1,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.7,
        "ground_hp_percentage": 0.05,
        "direct_electric_heating_percentage": 0.1,
        "ac_percentage": 0.3,
        "battery_percentage": 0.3,
        "prosumager_percentage": 0.2,
    }

    ecemf = ECEMFPostProcess(**scenario)
    # ecemf.create_output_csv()

    ECEMFFigures(baseline_scenario=baseline, scenario=scenario).create_figures()

# TODO zu jedem einzelnen Gebäude im original df die geclusterten dazufügen + Ergebnis und dann den
#  heat demand vergeleichen, Außerdem die Abweichung in Floor area plotten! (wegen clustering)
