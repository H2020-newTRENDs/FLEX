from pathlib import Path
import pandas as pd
import numpy as np
import random
from typing import List
import plotly.express as px

from flex.db import DB
from flex.config import Config
from flex_operation.constants import OperationTable, OperationScenarioComponent


class ECEMFDataPreparation:
    def __init__(self):
        self.project_folder = Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects\ECEMF_T4.3_Murcia")

    def calculate_peak_demand_for_all_scenarios(self, force_new_file: bool = False):
        """
        If the csv exists this function will be jumped except if force_new_file = True
        :return: csv file where the peak demand is saved for each scenario
        """
        file_name = self.project_folder / ""
        # check if file exists:

    pass


def find_net_peak_demand_day():
    pass


def find_net_peak_generation_day():
    pass


class ECEMFPostProcess:
    def __init__(self, region: str):
        self.region = region
        self.path_to_osm = Path(r"C:\Users\mascherbauer\PycharmProjects\OSM")
        self.path_to_project = Path(r"C:\Users\mascherbauer\PycharmProjects\FLEX\projects") / f"ECEMF_T4.3_{region}"
        self.original_building_ids = self.load_original_building_ids()
        self.clustered_building_df = self.load_clustered_building_df()
        self.original_building_df = self.load_original_building_df()
        # add the cluster ID to the original building df
        self.add_cluster_id_to_original_building_df()
        self.db = DB(config=Config(project_name="ECEMF_T4.3_Murcia"))
        self.scenario_table = self.db.read_dataframe(OperationTable.Scenarios)

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

        pass

    def percentage_of_total_buildings(self):
        """
        Percentage of total buildings that belong to this type at the studied location in this scenario
        :return:
        """

    @staticmethod
    def calculate_number_of_buildings_from_percentage(percentage: float, total_number: int) -> (int, int):
        denominator = 1 / percentage
        number_pro_tech, rest = divmod(total_number, denominator)
        rest_pro_tech, rest_anti_tech = np.ceil(rest / 2), np.floor(rest / 2)
        number_pro_tech += rest_pro_tech
        number_anti_tech = total_number - number_pro_tech
        return int(number_pro_tech), int(number_anti_tech)

    @staticmethod
    def assign_id(list_of_percentages: List[float],) -> int:
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
        choice = random.choices(choices, weights=weights)[0]
        return choice

    def scenario_generator(self,
                           pv_installation_percentage: float,
                           dhw_storage_percentage: float,
                           buffer_storage_percentage: float,
                           heating_element_percentage: float,
                           air_hp_percentage: float,
                           ground_hp_percentage: float,
                           direct_electric_heating_percentage: float,
                           ac_percentage: float,
                           battery_percentage: float,
                           ):
        """
        generates a scenario based on the percentage values that are provided for each installation. The values have
        to be between 0 and 1. 0 = no buildings are equipped with tech, 1 = all buildings are equipped.
        :param pv_installation_percentage: percentage of buildings having a PV installed (SFH = 5kWp, MFH = 15kWp)
        :param dhw_storage_percentage: percentage of buildings having a domestic hot water storage (SFH=300l, MFH=700l)
        :param buffer_storage_percentage: percentage of buildings having a heating buffer storage (SFH=700l, MFH=1500)
        :param heating_element_percentage: percentage of buildings having a heating element (max power 5kWh SFH with
                PV or 15kWh MFH with PV)
        :param air_hp_percentage: percentage of buildings having an air sourced heat pump
        :param ground_hp_percentage: percentage of buildings having a ground sourced heat pump
        :param direct_electric_heating_percentage: percentage of buildings having a direct electric heating system
        :param ac_percentage: percentage of buildings having an air conditioner for cooling (COP=4)
        :return:
        """
        assert 0 <= pv_installation_percentage <= 1
        assert 0 <= dhw_storage_percentage <= 1
        assert 0 <= buffer_storage_percentage <= 1
        assert 0 <= heating_element_percentage <= 1
        assert 0 <= air_hp_percentage <= 1
        assert 0 <= ground_hp_percentage <= 1
        assert 0 <= direct_electric_heating_percentage <= 1
        assert 0 <= ac_percentage <= 1
        assert 0 <= battery_percentage <= 1
        assert air_hp_percentage + ground_hp_percentage + direct_electric_heating_percentage <= 1
        gases_percentage = 1 - (air_hp_percentage + ground_hp_percentage + direct_electric_heating_percentage)

        dict_of_inputs = {
            "ID_PV": [1-pv_installation_percentage, pv_installation_percentage * 0.5, pv_installation_percentage * 0.25, pv_installation_percentage * 0.25],
            "ID_HotWaterTank": [dhw_storage_percentage],
            "ID_SpaceHeatingTank": [buffer_storage_percentage],
            "ID_HeatingElement": [heating_element_percentage],
            "ID_Boiler": [air_hp_percentage, ground_hp_percentage, direct_electric_heating_percentage, gases_percentage],
            "ID_SpaceCoolingTechnology": [ac_percentage]

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
        total_scenarios = []
        for _, row in self.clustered_building_df.iterrows():
            number_buildings = row["number_of_buildings"]
            building_id = row["ID_Building"]
            building_scenario = self.scenario_table.query(f"ID_Building == {building_id}")
            if building_id != 39:
                continue
            for n_building in range(number_buildings):
                chosen_building_attributes = {}
                for attribute_name, perc_list in dict_of_inputs.items():
                    chosen_id = self.assign_id(list_of_percentages=perc_list)
                    # translate the chosen ID to an ID in the scenario table:
                    # get unique labels for the specific building
                    possible_labels = building_scenario[attribute_name].unique().tolist()
                    print(possible_labels, translation_2_id[attribute_name][chosen_id])
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
                    if self.assign_id([battery_percentage]) == 1:
                        id_battery = list(set(possible_bat_ids).intersection([2, 3]))[0]
                    else:
                        id_battery = 1
                else:
                    id_battery = 1
                if len(possible_spacetank_ids) > 1:
                    if self.assign_id(possible_spacetank_ids) > 1:
                        id_buffer = list(set(possible_spacetank_ids).intersection([2, 3]))[0]
                    else:
                        id_buffer = 1
                else:
                    id_buffer = 1
                if len(possible_heatelement_ids) > 1:
                    if self.assign_id(possible_heatelement_ids) > 1:
                        id_heatelement = list(set(possible_heatelement_ids).intersection([2, 3]))[0]
                    else:
                        id_heatelement = 1
                else:
                    id_heatelement = 1
                full_query = f"{query} and ID_Battery == {id_battery} " \
                             f"and ID_SpaceHeatingTank == {id_buffer} " \
                             f"and ID_HeatingELement == {id_heatelement}"

                scenario = building_scenario.query(full_query)["ID_Scenario"].values.tolist()
                total_scenarios.append(scenario)










if __name__ == "__main__":
    ECEMFPostProcess(region="Murcia").scenario_generator(
        pv_installation_percentage=0.5,
        dhw_storage_percentage=0.5,
        buffer_storage_percentage=0.5,
        heating_element_percentage=0.5,
        air_hp_percentage=0.3,
        ground_hp_percentage=0.3,
        direct_electric_heating_percentage=0.2,
        ac_percentage=0.5,
        battery_percentage=0.5
    )

# TODO zu jedem einzelnen Gebäude im original df die geclusterten dazufügen + Ergebnis und dann den
#  heat demand vergeleichen, Außerdem die Abweichung in Floor area plotten! (wegen clustering)
#  ein shapefile erstellen bei dem die Gebäude zugeordnet sind
