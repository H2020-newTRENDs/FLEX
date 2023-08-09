from pathlib import Path
import pandas as pd
import numpy as np
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
        self.original_building_ids = self.load_original_building_ids()
        self.clustered_building_df = self.load_clustered_building_df()
        self.original_building_df = self.load_original_building_df()
        # add the cluster ID to the original building df
        self.add_cluster_id_to_original_building_df()

    def load_original_building_ids(self):
        return pd.read_excel(self.path_to_osm / f"Original_Building_IDs_to_clusters_{self.region}.xlsx")

    def load_clustered_building_df(self):
        return pd.read_excel(self.path_to_osm / f"OperationScenario_Component_Building_{self.region}.xlsx")

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

    def scenario_generator(self,
                           pv_installation_percentage: float,
                           dhw_storage_percentage: float,
                           buffer_storage_percentage: float,
                           heating_element_percentage: float,
                           air_hp_percentage: float,
                           ground_hp_percentage: float,
                           direct_electric_heating_percentage: float,
                           ac_percentage: float,
                           ):
        """
        generates a scenario based on the percentage values that are provided for each installation. The values have
        to be between 0 and 1.
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



if __name__ == "__main__":
    ECEMFPostProcess(region="Murcia")


# TODO zu jedem einzelnen Gebäude im original df die geclusterten dazufügen + Ergebnis und dann den
#  heat demand vergeleichen, Außerdem die Abweichung in Floor area plotten! (wegen clustering)
#  ein shapefile erstellen bei dem die Gebäude zugeordnet sind









