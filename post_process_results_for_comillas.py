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
from db_init import ProjectDatabaseInit
from flex_operation.run import main as operation_main


TRANSLATION_2_ID = {
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


def convert_to_numeric(column):
    return pd.to_numeric(column, errors="ignore")


def delete_result_task_folders(conf):
    for item in Path(conf.output).parent.iterdir():
        if item.is_dir():
            if conf.project_name in item.name and conf.project_name != item.name:
                print(f"Deleting directory and all contents: {item}")
                for sub_item in item.iterdir():
                    # Check if the sub_item is a file
                    if sub_item.is_file():
                        sub_item.unlink()  # Delete the file
                shutil.rmtree(item)


def delete_input_task_folders(conf):
    for item in Path(conf.input_operation).parent.iterdir():
        if item.is_dir():
            if conf.project_name in item.name and conf.project_name != item.name:
                print(f"Deleting directory and all contents: {item}")
                for sub_item in item.iterdir():
                    # Check if the sub_item is a file
                    if sub_item.is_file():
                        sub_item.unlink()  # Delete the file
                shutil.rmtree(item)


def delete_figure_task_folders(conf):
    for item in Path(conf.input_operation).parent.parent / "figure".iterdir():
        if item.is_dir():
            if conf.project_name in item.name and conf.project_name != item.name:
                print(f"Deleting directory and all contents: {item}")
                for sub_item in item.iterdir():
                    # Check if the sub_item is a file
                    if sub_item.is_file():
                        sub_item.unlink()  # Delete the file
                shutil.rmtree(item)


def delete_result_folder(conf):
    for item in Path(conf.output).parent.iterdir():
        if item.is_dir():
            if conf.project_name == item.name:
                print(f"Deleting directory and all contents: {item}")
                for sub_item in item.iterdir():
                    # Check if the sub_item is a file
                    if sub_item.is_file():
                        sub_item.unlink()  # Delete the file
                shutil.rmtree(item)

class ECEMFPostProcess:
    def __init__(self,
                 year: int,
                 region: str,
                 building_scenario: str,
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
                 baseline: dict,
                 previous_scenario: dict,
                 scenario_name: str,
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
        :param baseline: dictionary containing the parameters of the baseline scenario
        :param previous_scenario: dictionary containing the parameters of the scenario on which this scenario is built on
        """
        self.baseline = baseline
        self.previous_scenario = previous_scenario
        self.region = region
        self.year = year
        self.building_scenario = building_scenario
        self.data_base_name = f"ECEMF_T4.3_{region}_{year}_{building_scenario}"
        self.path_to_results = Path(__file__).parent / "data" / "output" / self.data_base_name
        self.path_to_model_input = Path(__file__).parent / "data" / "input_operation" / self.data_base_name
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{region}/data_output/"
        self.data_output.mkdir(parents=True, exist_ok=True)
        self.building_df = pd.read_excel(self.path_to_model_input /f"OperationScenario_Component_Building.xlsx", engine="openpyxl")
        self.scenario_name = scenario_name
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
        return pd.read_excel(self.path_to_model_input / f"OperationScenario_Component_Building.xlsx")

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
    

    def choose_appliances(self, building_type: str):
        all_ids = {}
        first_assignment = {
            "ID_PV": [1 - self.pv_installation_percentage, 
                    self.pv_installation_percentage * 0.5,
                    self.pv_installation_percentage * 0.25, 
                    self.pv_installation_percentage * 0.25],
            "ID_HotWaterTank": [self.dhw_storage_percentage],
            "ID_Boiler": [self.air_hp_percentage, 
                        self.ground_hp_percentage, 
                        self.direct_electric_heating_percentage,
                        self.gases_percentage, 
                        self.no_heating_percentage],
            "ID_SpaceCoolingTechnology": [self.ac_percentage],
        }

        for attribute_name, perc_list in first_assignment.items():
            chosen_id = self.assign_id(list_of_percentages=perc_list)
            trans_id = TRANSLATION_2_ID[attribute_name][chosen_id]
            if len(trans_id) > 1:
                if building_type == "SFH":  
                    all_ids[attribute_name] = trans_id[0]
                else:  # MFH
                    all_ids[attribute_name] = trans_id[1]
            else:
                all_ids[attribute_name] = trans_id[0]

        # based in the first assignment of the attributes that are independent we choose the non-independent ones:
        # battery only if there is PV:
        if all_ids["ID_PV"] != 1:
            yes_no = self.assign_id([self.battery_percentage])
            if building_type == "SFH":
                id_battery = [1, 2][yes_no] 
            else:  # MFH
                id_battery = [1, 3][yes_no]
        else:
            _ = self.assign_id([self.battery_percentage])
            id_battery = 1
        all_ids["ID_Battery"] = id_battery

        # Space heating tank only if heat pump exists:
        if all_ids["ID_Boiler"] in [2, 3]:  # heat pump indices
            yes_no = self.assign_id([self.buffer_storage_percentage])
            if building_type == "SFH":
                id_spaceheatingtank = [1, 2][yes_no]
            else:
                id_spaceheatingtank = [1, 3][yes_no]

        else:
            _ = self.assign_id([self.buffer_storage_percentage])
            id_spaceheatingtank = 1
        all_ids["ID_SpaceHeatingTank"] = id_spaceheatingtank

        # DHW tank is only installed with heat pumps (because it only plays a role for flexibility)
        if all_ids["ID_Boiler"] in [2, 3]:  # heat pump indices
            yes_no = self.assign_id([self.dhw_storage_percentage])
            if building_type == "SFH":
                id_dhwtank = [1, 2][yes_no]
            else:
                id_dhwtank = [1, 3][yes_no]

        else:
            _ = self.assign_id([self.dhw_storage_percentage])
            id_dhwtank = 1
        all_ids["ID_HotWaterTank"] = id_dhwtank

        # Behavior is a random chosen on for direct electric heating element
        if all_ids["ID_Boiler"] in [2, 3, 5]:  # Air_HP or Ground_HP or Gas
            all_ids["ID_Behavior"] = 1
        elif all_ids["ID_Boiler"] == 1:
            all_ids["ID_Behavior"] = 3 # Electric
        elif all_ids["ID_Boiler"] == 4:
            all_ids["ID_Behavior"] = 2 # no heating
        else:
            assert "Boiler ID is not pre-defined boiler ids"

        # ID SEMS
        if all_ids["ID_Boiler"] in [2, 3]:  # Air_HP or Ground_HP
            all_ids["ID_Prosumager"] = self.assign_id([self.prosumager_percentage])
        else:
            _ = self.assign_id([self.prosumager_percentage])
            all_ids["ID_Prosumager"] = 0

        # ID Heating Element is alsways 1
        all_ids["ID_HeatingElement"] = 1

        # ID EnergyPrice is always 1
        all_ids["ID_EnergyPrice"] = 1

        # ID Region is 1
        all_ids["ID_Region"] = 1

        all_ids["ID_Vehicle"] = 1
 
        return all_ids

    def scenario_generator_baseline(self) -> pd.DataFrame:
        """
        generates a scenario based on the percentage values that are provided for each installation. The values have
        to be between 0 and 1. 0 = no buildings are equipped with tech, 1 = all buildings are equipped.

        :return: returns a list of all the scenarios IDs for the whole building stock
        """
        columns=[
            "ID_Scenario",
            "ID_Region",
            "ID_Building",
            "ID_Boiler",
            "ID_SpaceHeatingTank",
            "ID_HotWaterTank",
            "ID_SpaceCoolingTechnology",
            "ID_PV",
            "ID_Battery",
            "ID_Vehicle",
            "ID_EnergyPrice",
            "ID_Behavior",
            "ID_HeatingElement",
            "ID_Prosumager"
        ]
        random.seed(42)
        scenario_list = []
        for i, row in tqdm(self.building_df.iterrows(), desc=f"creating scenario based on the probabilities"):
            building_id = row["ID_Building"]
            chosen_ids = self.choose_appliances(building_type=row["type"])

            building_attributes = {
                "ID_Scenario": i + 1,  
                "ID_Building": building_id,
                **chosen_ids
            }
            scenario_list.append(building_attributes)

        total_scenarios = pd.DataFrame(scenario_list, columns=columns)
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

    def worker(self, id_building, id_scenario, prosumager, db, electric_heating_ids) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):

        result_matrix_demand = pd.DataFrame(index=range(8760), columns=[id_building])
        result_matrix_feed2grid = pd.DataFrame(index=range(8760), columns=[id_building])
        result_matrix_e_cooling = pd.DataFrame(index=range(8760), columns=[id_building])
        result_matrix_e_hp_heating = pd.DataFrame(index=range(8760), columns=[id_building])
        result_matrix_q_heating = pd.DataFrame(index=range(8760), columns=[id_building])

        if prosumager == 0:
            ref_loads = db.read_parquet(
                table_name=OperationTable.ResultRefHour,
                scenario_ID=id_scenario,
                folder=self.path_to_results,
                column_names=["Grid", "Feed2Grid", "E_Heating_HP_out", "E_RoomCooling", "Q_RoomHeating"]
            )

            # change the grid demand and heating of direct electric heating buildings if they are MFH with moving avg.5
            if id_scenario in electric_heating_ids:
                result_matrix_demand.loc[:, id_building] = ref_loads["Grid"].rolling(window=5, min_periods=1).mean().to_numpy()
                result_matrix_q_heating.loc[:, id_building] = ref_loads["Q_RoomHeating"].rolling(window=5, min_periods=1).mean().to_numpy()
            else:
                result_matrix_demand.loc[:, id_building] = ref_loads["Grid"].to_numpy()
                result_matrix_q_heating.loc[:, id_building] = ref_loads["Q_RoomHeating"].to_numpy()
          
            result_matrix_feed2grid.loc[:, id_building] = ref_loads["Feed2Grid"].to_numpy()
            result_matrix_e_cooling.loc[:, id_building] = ref_loads["E_RoomCooling"].to_numpy()
            result_matrix_e_hp_heating.loc[:, id_building] = ref_loads["E_Heating_HP_out"].to_numpy()

        else:
            opt_loads = db.read_parquet(
                table_name=OperationTable.ResultOptHour,
                scenario_ID=id_scenario,
                folder=self.path_to_results,
                column_names=["Grid", "Feed2Grid", "E_Heating_HP_out", "E_RoomCooling", "Q_RoomHeating"])

            if id_scenario in electric_heating_ids:
                result_matrix_demand.loc[:, id_building] = opt_loads["Grid"].rolling(window=5, min_periods=1).mean().to_numpy()
                result_matrix_q_heating.loc[:, id_building] = opt_loads["Q_RoomHeating"].rolling(window=5, min_periods=1).mean().to_numpy()
            else:
                result_matrix_demand.loc[:, id_building] = opt_loads["Grid"].to_numpy()
                result_matrix_q_heating.loc[:, id_building] = opt_loads["Q_RoomHeating"].to_numpy()
            
            result_matrix_feed2grid.loc[:, id_building] = opt_loads["Feed2Grid"].to_numpy()
            result_matrix_e_cooling.loc[:, id_building] = opt_loads["E_RoomCooling"].to_numpy()
            result_matrix_e_hp_heating.loc[:, id_building] = opt_loads["E_Heating_HP_out"].to_numpy()

        return result_matrix_demand, result_matrix_feed2grid, result_matrix_e_hp_heating, result_matrix_e_cooling, result_matrix_q_heating

    def calculate_total_load_profiles(self, total_df: pd.DataFrame):
        database = DB(path=self.path_to_results / f"{self.data_base_name}.sqlite")
        building_table = database.read_dataframe(table_name="OperationScenario_Component_Building",
                                                 column_names=["ID_Building", "type"])
        scenario_table = database.read_dataframe(table_name="OperationScenario", column_names=["ID_Scenario", "ID_Building", "ID_Boiler"])
        merged = pd.merge(scenario_table, building_table, on="ID_Building")
        electric_heating_ids = merged.loc[(merged.loc[:, "ID_Boiler"] == 1) & (merged.loc[:, "type"] == "MFH"), "ID_Scenario"].to_list()
        
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.worker, id_building, id_scenario, prosumager_value, database, electric_heating_ids)
                for (id_building, id_scenario, prosumager_value) in list(zip(
                    total_df["ID_Building"], total_df["ID_Scenario"], total_df['ID_Prosumager'])
                )
            ]

            results_demand = []
            results_feed2grid = []
            results_e_hp_heating = []
            results_q_heating = []
            results_cooling = []

            for future in tqdm(as_completed(futures), total=len(futures), desc="calculating total load profiles"):
                demand, feed2grid, heating_e_hp, cooling, heating_q = future.result()
                results_demand.append(demand)
                results_feed2grid.append(feed2grid)
                results_e_hp_heating.append(heating_e_hp)
                results_q_heating.append(heating_q)
                results_cooling.append(cooling)

        result_matrix_demand = pd.concat(results_demand, axis=1)
        result_matrix_feed2grid = pd.concat(results_feed2grid, axis=1)
        result_matrix_e_hp_heating = pd.concat(results_e_hp_heating, axis=1)
        result_matrix_cooling = pd.concat(results_cooling, axis=1)
        result_matrix_q_heating = pd.concat(results_q_heating, axis=1)

        return result_matrix_demand, result_matrix_feed2grid, result_matrix_e_hp_heating, result_matrix_cooling, result_matrix_q_heating

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

    @staticmethod
    def select_max_days(demand: pd.DataFrame) -> (int, int):
        max_grid_day_index = ECEMFPostProcess.find_net_peak_demand_day(demand)
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
                              old_column_names
                              ):

        # load the table where real IDs are matched to the clustered IDs:
        match = pd.read_excel(self.path_to_model_input /
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
        column_names = []
        scenario_and_building_id = {}
        for j, scenario_id in enumerate(old_column_names):
            # get the building ID for the scenario ID
            cluster_id = clustered_ids[scenario_id]
            # with the building ID (which is the ID of the clustered buildings) select a real building ID
            real_id = int(match_dict[cluster_id].pop())
            # change the column name with the real ID:
            column_names.append(real_id)
            scenario_and_building_id[real_id] = scenario_id

        # create a csv file containing the scenario ID and the Building ID:
        scenario_building_table = self.create_scenario_building_csv(scenario_and_building_id)

        return scenario_building_table

    def create_scenario_building_csv(self, scenario_and_building_id):
        building_scenario_ids_df = pd.DataFrame.from_dict(
            scenario_and_building_id, orient="index"
        ).sort_index().reset_index().rename(columns={0: "ID_Scenario", "index": "ID_Building"})
        table = pd.merge(
            building_scenario_ids_df, self.scenario_table.rename(columns={"ID_Building": "FLEX_ID_Building"}),
            on="ID_Scenario", how="left"
        )
        # add prosumager ID to the table:
        # create a df that contains the Scenario ID, the real Building ID and if Prosumager is in the house:
        # check if optimisation exists for this scenario (dependent on the heating system)
        # air hp = 2 and ground hp = 3
        table["ID_Prosumager"] = table["ID_Boiler"].apply(
            lambda x: self.assign_id([self.prosumager_percentage]) if x in [2, 3] else 0
        )
        return table

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
        final_info.to_csv(self.data_output / f"I_{file_name}.csv", sep=";")
        final_info.to_csv(self.data_output / f"Info_{self.region}_{self.year}_{self.scenario_name.replace('P', 'Prosumager')}.csv")
        print("saved add information csv")

    def __file_name__(self):
        return f"{self.year}_" \
               f"{self.region}_" \
               f"{self.building_scenario}_" \
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

    def building_change(self, old_building_table: pd.DataFrame) -> (list, list):
        print("implementing renovations")

        type_df = self.building_df.loc[:, ["ID_Building", "type"]]
        df = pd.merge(self.scenario_table, type_df, on="ID_Building")

        row_list = []
        for _, row in tqdm(old_building_table.iterrows(), desc=f"updating the buildings"):
            # get the real building ID and see what is the corresponding cluster from the Oiriginal Building IDS to
            # cluster file
            flex_id = [key for key, value_list in match_dict.items() if row["ID_Building"] in value_list][0]
            new_row = df.loc[
                      (df.loc[:, "ID_SpaceCoolingTechnology"] == row["ID_SpaceCoolingTechnology"]) &
                      (df.loc[:, "ID_HotWaterTank"] == row["ID_HotWaterTank"]) &
                      (df.loc[:, "ID_PV"] == row["ID_PV"]) &
                      (df.loc[:, "ID_SpaceHeatingTank"] == row["ID_SpaceHeatingTank"]) &
                      (df.loc[:, "ID_Boiler"] == row["ID_Boiler"]) &
                      (df.loc[:, "ID_Battery"] == row["ID_Battery"]) &
                      (df.loc[:, "type"] == row["type"]) &
                      (df.loc[:, "ID_Building"] == flex_id)
            , :].copy()
            # add prosumager ID from old table
            new_row["ID_Prosumager"] = row["ID_Prosumager"]
            # overwrite the ID Building (from flex) with the real ID Building from old table
            new_row["FLEX_ID_Building"] = new_row["ID_Building"]
            new_row["ID_Building"] = row["ID_Building"]
            row_list.append(new_row)

        new_scenario_table = pd.concat(row_list, axis=0).drop(columns=["number_of_buildings"]).reset_index(drop=True)
        return new_scenario_table
    
    # def create_summary_excel_for_building_type_and_number_of_inhabs(self, df: pd.DataFrame):
    #     # open excel if it already exists:
    #     file_name = f"Building_types_{self.year}_{self.region}_{self.building_scenario}.csv"
    #     if file_name not in self.data_output.iterdir():
    #         # get the building 
    #         type_df = pd.read_excel(
    #             self.path_to_model_input / f"OperationScenario_Component_Building.xlsx", engine="openpyxl"
    #         ).loc[:, ["ID_Building", "type", "person_num"]].rename(columns={"ID_Building": "FLEX_ID_Building"})
    #         df = pd.merge(df, type_df, on="FLEX_ID_Building")
    #         df.drop(columns=["FLEX_ID_Building"], inplace=True)
    #         df.to_csv(self.data_output / file_name)
        


    def heating_systems_change(self):
        if "baseline" in self.previous_scenario.keys():
            prev_scen = ECEMFPostProcess(**self.previous_scenario, previous_scenario=None, scenario_name=self.scenario_name)
        else:
            prev_scen = ECEMFPostProcess(**self.previous_scenario, previous_scenario=None, baseline=None, scenario_name=self.scenario_name)

        df = pd.read_excel(
            prev_scen.path_to_model_input / f"OperationScenario.xlsx", engine="openpyxl"
            )
        
        # add the building type (MFH SFH) to the previous df:
        type_df = pd.read_excel(
            prev_scen.path_to_model_input / f"OperationScenario_Component_Building.xlsx", engine="openpyxl"
        ).loc[:, ["ID_Building", "type"]]
        df = pd.merge(df, type_df, on="ID_Building")

        # if number is positive, more is installed in current state
        change_pv = self.pv_installation_percentage - prev_scen.pv_installation_percentage
        change_hp_air = self.air_hp_percentage - prev_scen.air_hp_percentage  # Air HP
        change_hp_ground = self.ground_hp_percentage - prev_scen.ground_hp_percentage  # Ground HP
        change_direct_e = self.direct_electric_heating_percentage - prev_scen.direct_electric_heating_percentage
        change_gases = self.gases_percentage - prev_scen.gases_percentage
        change_no_heating = self.no_heating_percentage - prev_scen.no_heating_percentage

        change_dhw_tank = self.dhw_storage_percentage - prev_scen.dhw_storage_percentage
        change_cooling = self.ac_percentage - prev_scen.ac_percentage
        change_battery = self.battery_percentage - prev_scen.battery_percentage
        change_buffer_tank = self.buffer_storage_percentage - prev_scen.buffer_storage_percentage
        change_prosumager = self.prosumager_percentage - prev_scen.prosumager_percentage

        percentage_dict_heating_systems = {
            "ID_Boiler_HP_air": change_hp_air,
            "ID_Boiler_HP_ground": change_hp_ground,
            "ID_Boiler_direct_e": change_direct_e,
            "ID_Boiler_gases": change_gases,
            "ID_Boiler_no_heating": change_no_heating,
        }
        percentage_dict_storage_systems = {
            "ID_PV": change_pv,
            "ID_HotWaterTank": change_dhw_tank,
            "ID_SpaceCoolingTechnology": change_cooling,
            "ID_SpaceHeatingTank": change_buffer_tank,
            "ID_Prosumager": change_prosumager,
            "ID_Battery": change_battery,
        }
        # Sort the dictionary by its values so the heating systems can be replaced in the next step.
        # The storage systems have to be after the heating systems because their percentage depends partly on the
        # installed systems (eg. Battery)
        sorted_dict = dict(sorted(percentage_dict_heating_systems.items(), key=lambda item: item[1]))
        percentage_dict = {**sorted_dict, **percentage_dict_storage_systems}

        for tech_name, perc_increase in percentage_dict.items():
            if "ID_Boiler" in tech_name:  # differentiate between the different heating systems
                id_name = "ID_Boiler"
                if "HP_air" in tech_name:
                    tech_id_without = {"SFH": [2], "MFH": [2]}
                    tech_id = {"SFH": [2], "MFH": [2]}
                elif "HP_ground" in tech_name:
                    tech_id_without = {"SFH": [3], "MFH": [3]}
                    tech_id = {"SFH": [3], "MFH": [3]}
                elif "direct_e" in tech_name:
                    tech_id_without = {"SFH": [1], "MFH": [1]}
                    tech_id = {"SFH": [1], "MFH": [1]}
                elif "gases" in tech_name:
                    tech_id_without = {"SFH": [5], "MFH": [5]}
                    tech_id = {"SFH": [5], "MFH": [5]}

                elif "no_heating" in tech_name:
                    tech_id_without = {"SFH": [4], "MFH": [4]}
                    tech_id = {"SFH": [4], "MFH": [4]}
                else:
                    print("wrong tech name")

            else:
                id_name = tech_name
                if tech_name == "ID_PV":  # distinguish between MFH and SFH
                    tech_id_without = {"SFH": [1], "MFH": [1]}
                    tech_id = {"SFH": [2, 4, 6], "MFH": [3, 5, 7]}
                elif tech_name == "ID_HotWaterTank":
                    tech_id_without = {"SFH": [1], "MFH": [1]}
                    tech_id = {"SFH": [2], "MFH": [3]}
                elif tech_name == "ID_SpaceHeatingTank":
                    tech_id_without = {"SFH": [1], "MFH": [1]}
                    tech_id = {"SFH": [2], "MFH": [3]}
                elif tech_name == "ID_Battery":
                    tech_id_without = {"SFH": [1], "MFH": [1]}
                    tech_id = {"SFH": [2], "MFH": [3]}
                elif tech_name == "ID_SpaceCoolingTechnology":
                    tech_id_without = {"SFH": [2], "MFH": [2]}
                    tech_id = {"SFH": [1], "MFH": [1]}
                elif tech_name == "ID_Prosumager":
                    tech_id_without = {"SFH": [0], "MFH": [0]}
                    tech_id = {"SFH": [1], "MFH": [1]}
            if perc_increase == 0:
                continue  # nothing changes

            def query_or_full_df(dataframe, query):
                if query:
                    return dataframe.query(query)
                else:
                    # Query string is empty, return the full DataFrame
                    return dataframe

            # the perc increase needs first to be set to "changing" in the buildings that dont have the tech yet
            # except if the percentage is declining
            if perc_increase < 0:  # if tech is decreasing tech_id_without and tech_id have to be switched:
                tech_id, tech_id_without = tech_id_without, tech_id  # doesnt affect the heating techs
            # for non heating technologies the perc change has to be negative and positive at the same time because
            # they replace themselves:
            filter_query = ""
            if id_name == "ID_Battery":  # only concerns buildings with PV
                filter_query = f"ID_PV in {['new_PV', 2, 3, 4, 5, 6, 7]}"
            if id_name == "ID_SpaceHeatingTank":  # only buildings with heat pumps can have buffer storage
                filter_query = f"ID_Boiler in {[2, 3]}"
            if id_name == "ID_Prosumager":
                filter_query = f"ID_Boiler in {[2, 3]}"

            if perc_increase < 0 or not "ID_Boiler" == id_name:
                b_type_group = df.groupby("type")
                for b_type, group in b_type_group:
                    if id_name == "ID_Boiler":
                        filter_query = f"ID_Boiler in {tech_id[b_type]}"
                    filtered_group = query_or_full_df(group, filter_query)
                    # changing number is the number of systems that will be exchanged
                    if id_name == "ID_Boiler":  # boiler percentages are always on the whole dataset even with filterquery
                        # if a technology goes to 0 the random initialization of buildings can lead to
                        # this code not working because the percentages are not exactly met. Therefore the
                        # changing_number is adapted if a technology goes completely to 0 or 1:
                        changing_number = max(round(group[id_name].count() * perc_increase), -len(filtered_group))
                    else:  # relative percentages are calculated
                        changing_number = min(abs(round(filtered_group[id_name].count() * perc_increase)),
                                              len(filtered_group.loc[
                                                      filtered_group.loc[:, id_name].isin(tech_id_without[b_type])]))
                    # choose the buildings with the tech id and set some of them to "change"
                    indices_to_replace = filtered_group.loc[
                        filtered_group.loc[:, id_name].isin(tech_id_without[b_type]), id_name
                    ].sample(n=abs(changing_number), random_state=42).index
                    df.loc[indices_to_replace, id_name] = "changing"

            if perc_increase > 0 or not "ID_Boiler" == id_name:
                b_type_group = df.groupby("type")
                for b_type, group in b_type_group:
                    if id_name == "ID_Boiler":
                        filter_query = f"ID_Boiler in {tech_id[b_type]}"
                    if id_name == "ID_Boiler":  # boiler percentages are always on the whole dataset even with filterquery
                        filtered_group = group.copy()
                        # if a technology goes to 0 the random initialization of buildings can lead to
                        # this code not working because the percentages are not exactly met. Therefore the
                        # changing_number is adapted if a technology goes completely to 0 or 1:
                        changing_number = min(round(group[id_name].count() * perc_increase),
                                              len(filtered_group.loc[filtered_group.loc[:, id_name] == "changing", :]))
                    else:
                        filtered_group = query_or_full_df(group, filter_query)
                        changing_number = min(round(filtered_group[id_name].count() * perc_increase),
                                              len(filtered_group.loc[filtered_group.loc[:, id_name] == "changing", :]))

                    indices_changing = filtered_group.loc[filtered_group.loc[:, id_name] == "changing", id_name].sample(
                        n=abs(changing_number), random_state=42).index
                    if id_name == "ID_PV":
                        df.loc[
                            indices_changing, id_name] = "new_PV"  # need to distinguish between the different PV orientations later
                    else:
                        df.loc[indices_changing, id_name] = tech_id[b_type][0]

        def complex_condition(row):
            if row["ID_PV"] == "new_PV":
                if row["type"] == "SFH":
                    return np.random.choice([2, 4, 6], p=[0.5, 0.25, 0.25])
                else:
                    return np.random.choice([3, 5, 7], p=[0.5, 0.25, 0.25])
            else:
                return row["ID_PV"]
            
        # replace the "new_PV" with the different PV sizes + orientation
        np.random.seed(42)
        df["ID_PV"] = df.apply(complex_condition, axis=1)
        # if (for some reason i dont understand) a "changing" is still in df (in the ID_Boiler), select a new one
        if "changing" in df["ID_Boiler"].to_list():
            df.loc[df.loc[:, "ID_Boiler"] == "changing", "ID_Boiler"] = np.random.choice([1, 2, 3, 4, 5,] ,
                                                                                         p=[self.direct_electric_heating_percentage,
                                                                                            self.air_hp_percentage,
                                                                                            self.ground_hp_percentage,
                                                                                            self.no_heating_percentage,
                                                                                            self.gases_percentage])
              
        # fix the ID Behavior
        df.loc[df.loc[:, "ID_Boiler"].isin([2, 3, 5]), "ID_Behavior"] = 1
        df.loc[df.loc[:, "ID_Boiler"] == 1, "ID_Behavior"] = 3
        df.loc[df.loc[:, "ID_Boiler"] == 4, "ID_Behavior"] = 2
        return df.apply(convert_to_numeric)
    
    def get_electricty_price_for_max_demand_day(self, start_hour: int, end_hour: int):
        # read electricity price:
        elec_price = pd.read_excel(self.path_to_model_input / "OperationScenario_EnergyPrice.xlsx", engine="openpyxl").loc[:, "electricity_1"]
        max_day_price = elec_price.iloc[start_hour: end_hour+1]
        return max_day_price
    
    def save_electricity_price_to_excel(self, price: pd.Series):
        # check if price excel existst:
        file_name = "electricity_prices.csv"
        if file_name in [f.name for f in self.data_output.iterdir()]:
            price_df = pd.read_csv(self.data_output / file_name, sep=";")
        else:
            price_df = pd.DataFrame()
        price_df.loc[:, self.__file_name__()] = price.to_numpy()
        price_df.to_csv(self.data_output / file_name, sep=";", index=False)

    def run_FLEX_model(self):
        cfg = Config(f"ECEMF_T4.3_{self.region}_{self.year}_{self.building_scenario}")
        # delete all task folder before running the model in case they exist from preivous run
        delete_input_task_folders(conf=cfg)
        delete_result_task_folders(conf=cfg)
        # delete_result_folder(conf=cfg)  # code doesnt work if i delete the sqlite file instantly and the folder which is needed
        print(f"initialising model for {self.region} {self.year} {self.building_scenario}")
        init = ProjectDatabaseInit(
            config=cfg,
            load_scenario_table=True)
        init.run()  # old results in sqlite file are deleted! parquet files are overwritten in the operation_main

        operation_main(project_name = cfg.project_name,
                use_multiprocessing = True,
                save_hourly_results = True,
                save_monthly_results = False,
                save_yearly_results = True,
                hourly_save_list = None,
                operation_scenario_ids = None,
                n_cores=20,
                )
            
        try:
            delete_result_task_folders(cfg)
            delete_input_task_folders(cfg)
            delete_figure_task_folders(cfg)
        except:
            print(f"taskfolders for {self.region} {self.year} were not deleted")
        
        print(f"FLEX run for {self.region} {self.year} {self.building_scenario} done!")
       
    
    def create_output_csv(self):
        if self.__file_name__() == get_file_name(self.baseline):
            # generate the baseline scenario
            scenarios = self.scenario_generator_baseline()
            # save scenarios table to start folder because it will be loaded by the FLEX model run
            scenarios.to_excel(self.path_to_model_input / "OperationScenario.xlsx", index=False)
        else:
            # change the technologies based on the change in percentages compared to the previous scenario
            scenarios = self.heating_systems_change()
            # drop the "type" column from teh scenario  table
            scenarios.drop(columns=["type"], inplace=True)
            scenarios.to_excel(self.path_to_model_input / "OperationScenario.xlsx", index=False)

        # calculate the  scenario using the FLEX operation model:
        self.run_FLEX_model()

        total_grid_demand, total_grid_feed, heating_e_hp, cooling, heating_q = self.calculate_total_load_profiles(
            total_df=scenarios
        )

        # save the total profiles to parquet
        total_grid_demand.columns = total_grid_demand.columns.astype(str)
        total_grid_feed.columns = total_grid_feed.columns.astype(str)
        heating_e_hp.columns = heating_e_hp.columns.astype(str)
        cooling.columns = cooling.columns.astype(str)
        heating_q.columns = heating_q.columns.astype(str)

        total_grid_demand.to_parquet(self.data_output / f"Demand_{self.__file_name__()}.parquet.gzip")
        total_grid_feed.to_parquet(self.data_output / f"Feed_{self.__file_name__()}.parquet.gzip")
        heating_e_hp.to_parquet(self.data_output / f"Heating_hp_{self.__file_name__()}.parquet.gzip")
        cooling.to_parquet(self.data_output / f"Cooling_e_{self.__file_name__()}.parquet.gzip")
        heating_q.to_parquet(self.data_output / f"Heating_q_{self.__file_name__()}.parquet.gzip")
        # save the profiles to csv
        demand_start_hour, demand_end_hour = self.select_max_days(total_grid_demand)
        demand_max_demand_day = total_grid_demand.loc[demand_start_hour: demand_end_hour, :]
        feed_max_demand_day = total_grid_feed.loc[demand_start_hour: demand_end_hour, :]

        feed_start_hour, feed_end_hour = self.select_max_days(total_grid_feed)
        demand_max_feed_day = total_grid_demand.loc[feed_start_hour: feed_end_hour, :]
        feed_max_feed_day = total_grid_feed.loc[feed_start_hour: feed_end_hour, :]
        hourly_df = self.save_hourly_csv(
            demand_d_day=demand_max_demand_day,
            demand_f_day=demand_max_feed_day,
            feed2grid_d_day=feed_max_demand_day,
            feed2grid_f_day=feed_max_feed_day,
        )
        self.add_baseline_data_to_hourly_csv(hourly_df)
        print("saved hourly csv file")

        self.create_overall_information_csv(scenario_df=scenarios,
                                            total_grid_demand=total_grid_demand,
                                            max_demand_day=int(demand_start_hour / 24),
                                            max_feed_day=int(feed_start_hour / 24),
                                            file_name=self.__file_name__(),

                                            )
        
        # self.create_summary_excel_for_building_type_and_number_of_inhabs(df=scenarios)
        # add the electricity price to an excel with elec prices for all scenarios:
        max_day_price = self.get_electricty_price_for_max_demand_day(start_hour=demand_start_hour, end_hour=demand_end_hour)
        self.save_electricity_price_to_excel(price=max_day_price)

    def get_hourly_csv(self) -> pd.DataFrame:
        file_name = self.__file_name__()
        path_2_file = self.data_output / f"{file_name}.csv"
        return pd.read_csv(path_2_file, sep=";")

    def get_info_csv(self) -> pd.DataFrame:
        file_name = self.__file_name__()
        path_2_file = self.data_output / f"I_{file_name}.csv"
        return pd.read_csv(path_2_file, sep=";")

    def add_baseline_data_to_hourly_csv(self, hourly_csv_scen: pd.DataFrame):
        base = ECEMFPostProcess(**self.baseline, baseline=self.baseline, previous_scenario=None, scenario_name=self.scenario_name)
        try:  # if baseline does not exist yet:
            hourly_csv_base = base.get_hourly_csv()
            hourly_csv_base["type"] = hourly_csv_base["type"] + " baseline"
            df = pd.concat([hourly_csv_scen, hourly_csv_base], axis=0)
        except:
            df = hourly_csv_scen

        df.to_csv(self.data_output / f"{self.__file_name__()}.csv", sep=";", index=False)
        df.to_csv(self.data_output / f"Profiles_{self.region}_{self.year}_{self.scenario_name.replace('P', 'Prosumager')}.csv")


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
           f"{dictionary['building_scenario']}_" \
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
    def __init__(self, scenario: dict, scenario_name: str):
        self.baseline = scenario["baseline"]
        self.building_scenario = scenario["building_scenario"]
        self.data_output = Path(__file__).parent / "projects" / f"ECEMF_T4.3_{self.baseline['region']}/data_output/"
        self.path_2_figure = Path(__file__).parent / r"data/figure" / f"ECEMF_T4.3_{self.baseline['region']}"
        self.scenario_name = scenario_name
        nr_lists = 0
        for value in scenario.values():
            if isinstance(value, list):
                nr_lists += 1
        self.nr_lists = nr_lists
        if self.nr_lists == 1:
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

        elif self.nr_lists > 1:
            assert check_if_lists_in_dict_are_same_length(scenario), "lists do not have the same length"
            scenarios = []
            changing_parameters = [(key, value) for key, value in scenario.items() if isinstance(value, list)]
            for i in range(len(changing_parameters[0][1])):
                new_scen = scenario.copy()
                for param, values in changing_parameters:
                    new_scen[param] = values[i]
                scenarios.append(new_scen)
            self.scenario = scenarios
            self.changing_parameter = scenario_name
        else:
            self.scenario = scenario

        self.check_if_scenarios_exist(self.scenario)

    def path_to_gzip(self, filename):
        return Path(self.data_output) / f"{filename}.parquet.gzip"

    def check_if_scenarios_exist(self, scenarios):
        # check if baseline exists
        if not self.path_to_gzip(f"Demand_{get_file_name(self.baseline)}").exists() or not \
                self.path_to_gzip(f"Feed_{get_file_name(self.baseline)}").exists():
            print(
                f"{get_file_name(self.baseline)} \n parquet files do not exist. create_output_csv"
            )
            ECEMFPostProcess(**self.baseline, baseline=self.baseline, previous_scenario=None, scenario_name=self.scenario_name).create_output_csv()

        if isinstance(scenarios, list):
            # the first scenario will be based on the baseline in case it is not the same as the baseline, making
            # the baseline the previous scenario. After that always the previous scenario from the scenario list
            # is taken as the previous scenario:
            for i, scen in enumerate(scenarios):
                if i == 0:  # make sure that the building scenario for the baseline is the same (moderate and high)
                    # need to be identical, we ignore this parameter for the dict comparison
                    a = scen.copy()
                    a.pop("building_scenario")
                    a.pop("baseline")
                    b = self.baseline.copy()
                    b.pop("building_scenario")

                    if a == b:
                        continue
                    else:
                        previous_scenario = self.baseline
                else:  # the i-1 scenario is taken as the previous one
                    # if the previous scenario scenario is the baseline, the building scenario parameter needs to be changed
                    a = scenarios[i - 1].copy()
                    a.pop("building_scenario")
                    a.pop("baseline")
                    b = self.baseline.copy()
                    b.pop("building_scenario")
                    if a == b:
                        previous_scenario = scenarios[i - 1]
                        previous_scenario["building_scenario"] = "H"
                    else:
                        previous_scenario = scenarios[i - 1]
                # check if the scenario csv exists. If not, create it:
                if not self.path_to_gzip(f"Demand_{get_file_name(scen)}").exists() or not \
                        self.path_to_gzip(f"Feed_{get_file_name(scen)}").exists():
                    print(
                        f"{get_file_name(scen)} \n parquet files do not exist. create_output_csv"
                    )
                    ECEMFPostProcess(**scen, previous_scenario=previous_scenario, scenario_name=self.scenario_name).create_output_csv()

        else:  # if its only a single scenario, base it on the baseline scenario
            if not self.path_to_gzip(f"Demand_{get_file_name(scenarios)}").exists() or not \
                    self.path_to_gzip(f"Feed_{get_file_name(scenarios)}").exists():
                print(
                    f"{get_file_name(scenarios)} \n parquet files do not exist. create_output_csv"
                )
                ECEMFPostProcess(**scenarios, previous_scenario=self.baseline, scenario_name=self.scenario_name).create_output_csv()

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
                self.path_2_figure / f"{self.scenario_name}"
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

        if self.nr_lists == 0:
            data = pd.read_csv(self.data_output / f"{get_file_name(self.scenario)}.csv", sep=";")
            demand_dict_d["scenario"] = data.query("type=='grid demand on max demand day'").iloc[:, 3:]
            feed_dict_d["scenario"] = data.query("type=='feed to grid on max demand day'").iloc[:, 3:]
            demand_dict_f["scenario"] = data.query("type=='grid demand on max feed day'").iloc[:, 3:]
            feed_dict_f["scenario"] = data.query("type=='feed to grid on max feed day'").iloc[:, 3:]
        elif self.nr_lists == 1:
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
            region = ECEMFPostProcess(**self.scenario, previous_scenario=None, scenario_name=self.scenario_name).region
        else:
            folder = self.path_2_figure / f"{self.scenario_name}"
            region = ECEMFPostProcess(**self.scenario[0], previous_scenario=None, scenario_name=self.scenario_name).region
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
            fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(25, 18))
            fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(25, 18))

            cmap = get_cmap('tab10')
            colors = [cmap(i) for i in range(4)]
            i = 0
            for scen_name, df in scen_dict["demand"].items():
                if scen_name == "baseline":
                    line_style = "--"
                else:
                    line_style = "-"
                if "2020" in scen_name:
                    continue
                ax1.plot(
                    np.arange(1, 25),
                    df.sum(axis=1) / 1_000 / 1_000,  # GWh
                    color=colors[i],
                    linewidth=2,
                    label=f'{scen_name.replace("_", " ").replace("baseline", "Baseline 2020")}',
                    linestyle=line_style
                )
                i+=1

            i = 0
            for scen_name, df in scen_dict["feed"].items():
                if scen_name == "baseline":
                    line_style = "--"
                else:
                    line_style = "-"
                if "2020" in scen_name:
                    continue
                ax2.plot(
                    np.arange(1, 25),
                    df.sum(axis=1) / 1_000 / 1_000 * -1,  # GWh
                    color=colors[i],
                    linewidth=2,
                    label=f'{scen_name.replace("_", " ").replace("baseline", "Baseline 2020")}',
                    linestyle=line_style
                )
                i+=1

            ax1.set_xlabel('Hour of Day')
            ax2.set_xlabel('Hour of Day')
            ax1.set_ylabel(f'Grid demand (MW)')
            ax2.set_ylabel(f'Feed to grid (MW)')
            if self.building_scenario == "H":
                scenario_name = "strong policy scneario"
            else:
                scenario_name = "weak policy scneario"
            ax1.set_title(f'Total load for {region} on peak {key.lower()} day \n in the {scenario_name}')
            ax2.set_title(f'Total feed to grid for {region} on peak {key.lower()} day \n in the {scenario_name}')
            ax1.legend(loc="upper left")
            ax2.legend(loc="lower left")

            # plt.xticks(range(0, 24))
            # ax.grid(True)
            plt.tight_layout()

            fig1.savefig(folder / f"Max_Demand_on_max_{key}-Day_Comparison.png")
            fig2.savefig(folder / f"Max_Feed_on_max_{key}-Day_Comparison.png")
            plt.close(fig1)
            plt.close(fig2)


    def create_figures(self):
        print("creating figures...")
        self.visualize_peak_day()
        # if isinstance(self.scenario, dict):
        #     # load the data from gzip files
        #     demand = pd.read_parquet(self.path_to_gzip(f"Demand_{get_file_name(self.scenario)}"), engine="pyarrow")
        #     feed = pd.read_parquet(self.path_to_gzip(f"Feed_{get_file_name(self.scenario)}"), engine="pyarrow")
        #     base_demand = pd.read_parquet(self.path_to_gzip(f"Demand_{get_file_name(self.baseline)}"), engine="pyarrow")
        #     base_feed = pd.read_parquet(self.path_to_gzip(f"Feed_{get_file_name(self.baseline)}"), engine="pyarrow")

        #     # create the figures
        #     self.create_boxplot_p_to_p_difference(baseline=base_demand, scenario=demand, demand_or_feed="Demand")
        #     self.plot_seasonal_daily_means(df_baseline=base_demand, df_scenario=demand, demand_or_feed="Demand")
        #     self.plot_seasonal_daily_means(df_baseline=base_feed, df_scenario=feed, demand_or_feed="Feed")
        # else:  # self.scenario is a list of scenarios:
        #     # create a graph that shows the difference between the scenarios:
        #     # load data
        #     self.visualize_heating_and_cooling_load()
            # self.create_scenario_comparison_daily_mean_plot_for_each_season()
        #     self.plot_peaks_of_all_households()
        #     self.plot_biggest_peak_profiles()
    
    def delete_all_results(self):
        path_2_delete = Path(__file__).parent / "data" / "output" #/ f"ECEMF_T4.3_{self.baseline['region']}"
        for item in Path(path_2_delete).iterdir():
            if item.is_dir():
                if f"ECEMF_T4.3_{self.scenario['region']}" in item.name and self.scenario['building_scenario'] in item.name:
                    print(f"Deleting directory and all contents: {item}")
                    for sub_item in item.iterdir():
                        # Check if the sub_item is a file
                        if sub_item.is_file():
                            sub_item.unlink()  # Delete the file
                    shutil.rmtree(item)


    def visualize_heating_and_cooling_load(self):
        fig_path = self.path_2_figure / f"{self.scenario_name}"
        heating_hp_dict = {}
        cooling_dict = {}
        heating_q_dict = {}
        for i, scen in enumerate(self.scenario):
            heating_hp = pd.read_parquet(self.data_output / f"Heating_hp_{get_file_name(scen)}.parquet.gzip",
                                         engine="pyarrow")
            cooling = pd.read_parquet(self.data_output / f"Cooling_e_{get_file_name(scen)}.parquet.gzip",
                                      engine="pyarrow")
            heating_total = pd.read_parquet(self.data_output / f"Heating_q_{get_file_name(scen)}.parquet.gzip",
                                            engine="pyarrow")
            if self.changing_parameter in scen.keys():
                nh = f"{round(scen[self.changing_parameter] * 100)} %"
            else:
                nh = f"{scen['year']}"
            heating_hp_dict[
                f"{self.changing_parameter}_{nh}"] = heating_hp.sum().sum() / 1000 / 1000  # MWh
            heating_q_dict[
                f"{self.changing_parameter}_{nh}"] = heating_total.sum().sum() / 1000 / 1000  # MWh
            cooling_dict[
                f"{self.changing_parameter}_{nh}"] = cooling.sum().sum() / 1000 / 1000  # MWh

        x_heating_hp, y_heating_hp = list(heating_hp_dict.keys()), list(heating_hp_dict.values())
        x_heating_q, y_heating_q = list(heating_q_dict.keys()), list(heating_q_dict.values())
        x_cooling, y_cooling = list(cooling_dict.keys()), list(cooling_dict.values())

        fig = plt.figure(figsize=(16, 20))
        ax = plt.gca()
        ax.plot(x_heating_hp, y_heating_hp, label="HP electricity heating", color="firebrick")
        ax.plot(x_cooling, y_cooling, label="electricity cooling", color="blue")
        plt.legend()
        ax.set_xlabel("scenario")
        ax.set_ylabel("demand in MWh")
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(fig_path / "Total_Heating_Cooling_heat_pumps.png")

        fig = plt.figure(figsize=(16, 20))
        ax = plt.gca()
        ax.bar(x_heating_q, y_heating_q, color="firebrick", label="total useful heating demand")
        ax.set_xlabel("scenario")
        ax.set_ylabel("demand in MWh")
        plt.xticks(rotation=90)
        plt.tight_layout()
        fig.savefig(fig_path / "Total_UED_Heating.png")

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


    def return_peaks_of_all_houses(self) -> dict:
        peaks_dict = {}
        for i, scen in enumerate(self.scenario):
            demand = pd.read_parquet(self.data_output / f"Demand_{get_file_name(scen)}.parquet.gzip",
                                         engine="pyarrow")
            peaks = demand.max(axis=0)
            peaks_dict[f"{scen['year']}"] = peaks
        return peaks_dict

    def plot_peaks_of_all_households(self):
        fig_path = self.path_2_figure / f"{self.scenario_name}"
        peaks_dict = {}

        for i, scen in enumerate(self.scenario):
            demand = pd.read_parquet(self.data_output / f"Demand_{get_file_name(scen)}.parquet.gzip",
                                         engine="pyarrow")
            peaks = demand.max(axis=0)
            peaks_dict[f"{scen['year']}"] = peaks

        labels = [n for n in peaks_dict.keys()]
        fig = plt.figure(figsize=(20, 16))
        ax = plt.gca()
        ax.boxplot([v / 1_000 for v in peaks_dict.values()], vert=True, positions=range(len(peaks_dict)))
        ax.set_xticklabels(labels, rotation=0)
        plt.title(f"{self.scenario_name.replace('P', 'Prosumager').replace('_', ' ')}")
        plt.ylabel("demand in kWh")
        plt.tight_layout()
        plt.savefig(fig_path / f"Peaks.png")
        plt.close()          

    def plot_biggest_peak_profiles(self):
        fig_path = self.path_2_figure / f"{self.scenario_name}"
        for i, scen in enumerate(self.scenario):
            demand = pd.read_parquet(self.data_output / f"Demand_{get_file_name(scen)}.parquet.gzip",
                                         engine="pyarrow")
            peaks = demand.max(axis=0)
            highest_5_peaks = peaks.sort_values().iloc[-5:]
            building_indices = highest_5_peaks.index
            highes_peak_profiles = demand.loc[:, building_indices]
           
            fig = px.line(data_frame=highes_peak_profiles.melt(var_name="Building ID", value_name="electricity demand in Wh", ignore_index=False).reset_index().rename(columns={"index": "hour"}), 
                       x="hour", 
                       y="electricity demand in Wh",
                       color="Building ID")
            fig.write_html(fig_path / f"highest_peak_profiles.html")


def plot_peaks_over_different_scenarios(scenarios: list[ECEMFFigures]):
    big_df_list = []
    for scen in scenarios:
        peak_dict = scen.return_peaks_of_all_houses()

        peak_df = pd.DataFrame.from_dict(peak_dict, orient="index").reset_index().rename(columns={"index": "year"})
        peak_df[f"scenario"] = scen.scenario_name.replace("P", "Prosuamger").replace("_", " ")
        melted = peak_df.melt(id_vars=["year", "scenario"], var_name="houses", value_name="peaks (kW)")
        melted["peaks (kW)"] =  melted["peaks (kW)"] / 1000
        big_df_list.append(melted)
    
    big_df = pd.concat(big_df_list, axis=0)

    matplotlib.rc("font", **{"size": 28})
    palette = sns.color_palette("Paired", 6)
    figure = plt.figure(figsize=(20, 16))
    sns.boxplot(data=big_df,
                x="year",
                y="peaks (kW)",
                hue="scenario",
                palette=palette
                )
    
    plt.savefig(scen.path_2_figure / "Peak_demand_single_houses.png")
    plt.close()

    # show the same plot but without outliers because mean etc. can not be seen
    figure = plt.figure(figsize=(20, 16))
    sns.boxplot(data=big_df,
            x="year",
            y="peaks (kW)",
            hue="scenario",
            palette=palette,
            showfliers=False
            )
    plt.savefig(scen.path_2_figure / "Peak_demand_single_houses_no_outliers.png")
    plt.close()

def plot_peak_day_profile_comparison_between_scenarios(scenarios: list[ECEMFFigures]):
    small_df_dict = {}
    for scen in scenarios:
        demand_dict_max_demand, feed_dict_max_demand, demand_dict_max_feed, feed_dict_max_feed = scen.get_max_day_feed_and_demand_sum()
        for policy_year_scen, profiles in demand_dict_max_demand.items():
            if "baseline" in policy_year_scen:
                continue
            small_df_dict[policy_year_scen] = profiles.sum(axis=1) / 1_000 / 1_000  # MW, baseline will be always the same and gets overwritten

    small_df = pd.DataFrame.from_dict(small_df_dict, orient="index").T.reset_index().rename(columns={"index": "hours"}).melt(value_name="load in (MW)", id_vars="hours")
    small_df["policy scenario"] = small_df["variable"].apply(lambda x: "weak policy" if "weak" in x.lower() else "strong policy")
    small_df["prosumager scenario"] = small_df["variable"].apply(lambda x: "High Prosumager" if "high" in x.lower() else ("Medium Prosumager" if "medium" in x.lower() else "Low Prosumager"))
    small_df["year"] = small_df["variable"].apply(lambda x: x[-4:])
    
    years_to_plot = ["2030", "2040", "2050"]
    df_2020 = small_df[small_df['year'] == "2020"].iloc[:24, :]
    df_other_years = small_df[small_df['year'].isin(years_to_plot)]
    palette = sns.color_palette("tab10")
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(years_to_plot), figsize=(60, 16), sharey=True)
    for ax, year in zip(axes, years_to_plot):
        # Filter the DataFrame for the current year
        df_year = df_other_years[df_other_years['year'] == year]

        # Combine with the 2020 data
        df_plot = pd.concat([df_2020, df_year])

        # Create the line plot
        sns.lineplot(data=df_year,
                    x="hours",
                    y="load in (MW)",
                    hue="prosumager scenario",
                    style="policy scenario",
                    palette=palette,
                    ax=ax,
                    )
        ax.plot(df_2020['hours'], df_2020['load in (MW)'], color='red', linewidth=2, label='Baseline 2020', alpha=0.8, linestyle=":")
        ax.legend().set_visible(False)
        ax.set_title(year)
    # Adjust legend to be outside of the plot and include the red line "C"
    handles, labels = ax.get_legend_handles_labels()
    # handles.append(plt.Line2D([0], [0], color='red', linewidth=2))
    # labels.append('Baseline 2020')
    fig.legend(handles, labels, loc='upper center', ncol=len(labels)//4, bbox_to_anchor=(0.475, 0.87))

    plt.tight_layout()
    plt.savefig(scen.path_2_figure / "Peak_demand_profile_comparison.png")
    plt.close()
        
            
def plot_peak_day_comparison_between_scenarios(scenarios: list[ECEMFFigures]):
    big_df_list = []
    for scen in scenarios:
        small_df_dict = {}
        for year_scen in scen.scenario:
            demand = pd.read_parquet(scen.data_output / f"Demand_{get_file_name(year_scen)}.parquet.gzip", engine="pyarrow")
            max_total_peak = demand.sum(axis=1).max() / 1_000_000  # W->kW->MW
            small_df_dict[year_scen["year"]] = max_total_peak
        small_df = pd.DataFrame.from_dict(small_df_dict, orient="index").reset_index().rename(columns={"index": "year", 0: "peaks (MW)"})
        small_df["scenario"] = scen.scenario_name.replace("P", "Prosuamger").replace("_", " ")
        big_df_list.append(small_df)

    big_df = pd.concat(big_df_list, axis=0)

    matplotlib.rc("font", **{"size": 28})
    min_value = round(big_df["peaks (MW)"].min()) - 10
    palette = sns.color_palette("Paired", 6)
    figure = plt.figure(figsize=(20, 16))
    ax = plt.gca()
    sns.barplot(data=big_df,
                x="year",
                y="peaks (MW)",
                hue="scenario",
                palette=palette
                )
    plt.ylim(min_value, ax.get_ylim()[1])
    plt.ylabel("highest peak on peak demand day (MW)")
    plt.savefig(scen.path_2_figure / "Peak_demand_total.png")
    plt.close()

if __name__ == "__main__":
    # ECEMFBuildingComparison(region="Murcia").main()

    # Battery is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating element is only installed in buildings with PV so the probability only refers to buildings with PV.
    # Heating buffer storage is only installed in buildings with HPs. Probability only refers to buildings with HP
    Low = [0, 0.05, 0.1, 0.2]
    Medium = [0, 0.1, 0.3, 0.5]
    High = [0, 0.15, 0.4, 0.8]

    baseline_leeuwarden = {
        "year": 2020,
        "region": "Leeuwarden",
        "building_scenario": "H",
        "pv_installation_percentage": 0.02,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.04,
        "ground_hp_percentage": 0,
        "direct_electric_heating_percentage": 0.02,
        "gases_percentage": 0.9,
        "ac_percentage": 0.2,
        "battery_percentage": 0.1,
        "prosumager_percentage": 0,
    }
    building_scenario_leeuwarden_H = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Leeuwarden",
        "building_scenario": "H",
        "pv_installation_percentage": 0.02,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.04,
        "ground_hp_percentage": 0,
        "direct_electric_heating_percentage": 0.02,
        "gases_percentage": 0.9,
        "ac_percentage": 0.2,
        "battery_percentage": 0.1,
        "prosumager_percentage": 0,
        "baseline": baseline_leeuwarden
    }
    building_scenario_leeuwarden_M = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Leeuwarden",
        "building_scenario": "M",
        "pv_installation_percentage": 0.02,
        "dhw_storage_percentage": 0.5,
        "buffer_storage_percentage": 0,
        "heating_element_percentage": 0,
        "air_hp_percentage": 0.04,
        "ground_hp_percentage": 0,
        "direct_electric_heating_percentage": 0.02,
        "gases_percentage": 0.9,
        "ac_percentage": 0.2,
        "battery_percentage": 0.1,
        "prosumager_percentage": 0,
        "baseline": baseline_leeuwarden
    }

    # Leeuwarden Scenarios
    # calculate a complete scenario run:
    scenario_list_leeuwarden = []
    for i, prosumager_shares in enumerate([Low, Medium, High]):
        if i == 0:
            pr = "P-low"
        elif i == 1:
            pr = "P-medium"
        elif i == 2:
            pr = "P-high" 
        scenario_high_eff_leeuwarden = {
            "year": [2020, 2030, 2040, 2050],
            "region": "Leeuwarden",
            "building_scenario": "H",
            "pv_installation_percentage": [0.02, 0.15, 0.4, 0.6],
            "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
            "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
            "heating_element_percentage": 0,
            "air_hp_percentage": [0.04, 0.18, 0.5, 0.7],
            "ground_hp_percentage": [0, 0.05, 0.1, 0.15],
            "direct_electric_heating_percentage": [0.02, 0.03, 0.02, 0.01],
            "gases_percentage": [0.9, 0.7, 0.34, 0.1],
            "ac_percentage": [0.2, 0.3, 0.5, 0.7],
            "battery_percentage": [0.1, 0.12, 0.2, 0.3],
            "prosumager_percentage": prosumager_shares,
            "baseline": baseline_leeuwarden
        }

        scenario_moderate_eff_leeuwarden = {
            "year": [2020, 2030, 2040, 2050],
            "region": "Leeuwarden",
            "building_scenario": "M",
            "pv_installation_percentage": [0.02, 0.1, 0.3, 0.5],
            "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
            "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
            "heating_element_percentage": 0,
            "air_hp_percentage": [0.04, 0.18, 0.45, 0.6],
            "ground_hp_percentage": [0, 0.02, 0.06, 0.1],
            "direct_electric_heating_percentage": [0.02, 0.03, 0.02, 0.02],
            "gases_percentage": [0.9, 0.73, 0.43, 0.24],
            "ac_percentage": [0.2, 0.35, 0.6, 0.8],
            "battery_percentage": [0.1, 0.12, 0.16, 0.25],
            "prosumager_percentage": prosumager_shares,
            "baseline": baseline_leeuwarden
        }
        # complete scenarios
        # ECEMFFigures(scenario=scenario_high_eff_leeuwarden, scenario_name=f"Strong_policy_{pr}").create_figures()
        # ECEMFFigures(scenario=scenario_moderate_eff_leeuwarden, scenario_name=f"Weak_policy_{pr}").create_figures()
        # append scenarios to list for later plots over all scenarios:
    #     scenario_list_leeuwarden.append(ECEMFFigures(scenario=scenario_moderate_eff_leeuwarden, scenario_name=f"Weak_policy_{pr}"))
    #     scenario_list_leeuwarden.append(ECEMFFigures(scenario=scenario_high_eff_leeuwarden, scenario_name=f"Strong_policy_{pr}"))
 
    # plot_peak_day_comparison_between_scenarios(scenarios=scenario_list_leeuwarden)
    # plot_peaks_over_different_scenarios(scenarios=scenario_list_leeuwarden)
        
    # building scenarios
    # ECEMFFigures(scenario=building_scenario_leeuwarden_H, scenario_name="Buildings_strong_policy").create_figures()
    # ECEMFFigures(scenario=building_scenario_leeuwarden_M, scenario_name="Buildings_weak_policy").create_figures()

    # Murcia Scenarios
    baseline_murcia = {
        "year": 2020,
        "region": "Murcia",
        "building_scenario": "H",
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
    building_scenario_murcia_h = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Murcia",
        "building_scenario": "H",
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
        "baseline": baseline_murcia
    }
    building_scenario_murcia_m = {
        "year": [2020, 2030, 2040, 2050],
        "region": "Murcia",
        "building_scenario": "M",
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
        "baseline": baseline_murcia
    }

    # calculate a complete scenario run:
    scenario_list_murcia = []
    for i, prosumager_shares in enumerate([Low, Medium, High]):
        if i == 0:
            pr = "P-low"
        elif i == 1:
            pr = "P-medium"
        elif i == 2:
            pr = "P-high"
        scenario_high_eff = {
            "year": [2020, 2030, 2040, 2050],
            "region": "Murcia",
            "building_scenario": "H",
            "pv_installation_percentage": [0.015, 0.1, 0.4, 0.6],
            "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
            "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
            "heating_element_percentage": 0,
            "air_hp_percentage": [0.2, 0.35, 0.6, 0.8],
            "ground_hp_percentage": [0, 0.02, 0.04, 0.06],
            "direct_electric_heating_percentage": [0.39, 0.3, 0.2, 0.05],
            "gases_percentage": [0.19, 0.15, 0.07, 0.02],
            "ac_percentage": [0.5, 0.6, 0.8, 0.9],
            "battery_percentage": [0.1, 0.12, 0.2, 0.3],
            "prosumager_percentage": prosumager_shares,
            "baseline": baseline_murcia
        }

        scenario_moderate_eff = {
            "year": [2020, 2030, 2040, 2050],
            "region": "Murcia",
            "building_scenario": "M",
            "pv_installation_percentage": [0.015, 0.15, 0.3, 0.5],
            "dhw_storage_percentage": [0.5, 0.55, 0.6, 0.65],
            "buffer_storage_percentage": [0, 0.05, 0.15, 0.25],
            "heating_element_percentage": 0,
            "air_hp_percentage": [0.2, 0.3, 0.5, 0.7],
            "ground_hp_percentage": [0, 0.01, 0.02, 0.03],
            "direct_electric_heating_percentage": [0.39, 0.32, 0.2, 0.1],
            "gases_percentage": [0.19, 0.16, 0.1, 0.05],
            "ac_percentage": [0.5, 0.65, 0.8, 0.95],
            "battery_percentage": [0.1, 0.12, 0.16, 0.25],
            "prosumager_percentage": prosumager_shares,
            "baseline": baseline_murcia
        }
        # complete scenarios
        # ECEMFFigures(scenario=scenario_moderate_eff, scenario_name=f"Weak_policy_{pr}").create_figures()
        # ECEMFFigures(scenario=scenario_high_eff, scenario_name=f"Strong_policy_{pr}").create_figures()
        # append scenarios to list for later plots over all scenarios:
        scenario_list_murcia.append(ECEMFFigures(scenario=scenario_moderate_eff, scenario_name=f"Weak_policy_{pr}"))
        scenario_list_murcia.append(ECEMFFigures(scenario=scenario_high_eff, scenario_name=f"Strong_policy_{pr}"))

        # ECEMFFigures(scenario=scenario_high_eff, scenario_name=f"Strong_policy_{pr}").delete_all_results()
    plot_peak_day_profile_comparison_between_scenarios(scenarios=scenario_list_murcia)
    plot_peak_day_comparison_between_scenarios(scenarios=scenario_list_murcia)
    plot_peaks_over_different_scenarios(scenarios=scenario_list_murcia)

    # create file that indicates for each building: building type, number of persons, 
    # send prices to miguel from the highest total demand day (1 new file with all the scenarios with 24 values, 1 file Murcia, 1 file Leeuwarden, column name same as for scenario)
    # create prosumager scnearios , low, high, medium  --> 6 scenarios for strong and weak policy
    # prosumager percentages:
    # Low: 0, 5, 10, 20
    # Medium: 0, 10, 30, 50
    # High: 0, 15, 40, 80


    # # building scenarios
    # ECEMFFigures(scenario=building_scenario_murcia_h, scenario_name="Buildings_strong_policy").create_figures()
    # ECEMFFigures(scenario=building_scenario_murcia_m, scenario_name="Buildings_weak_policy").create_figures()



