from pathlib import Path
import logging
import multiprocessing

import numpy as np
import pandas as pd
import os
from typing import List
from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable

logger = logging.getLogger("PNRImporter")
logger.setLevel(level=logging.INFO)


class PRNImporter:
    def __init__(self, project_name):

        self.project_name = project_name
        self.main_path = Path(get_config(self.project_name).project_root / Path(f"projects/{project_name}/input"))
        self.building_names = {
            1: "EZFH_5_B",
            2: "EZFH_5_S",
            3: "EZFH_9_B",
            4: "EZFH_1_B",
            5: "EZFH_1_S",
            6: "MFH_5_B",
            7: "MFH_5_S",
            8: "MFH_1_B",
            9: "MFH_1_S"
        }
        self.scenario_table = DB(get_config(self.project_name)).read_dataframe(
            table_name=OperationTable.Scenarios.value)

    def scenario_id_2_building_name(self, id_scenario: int) -> str:
        building_id = int(
            self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id_scenario, "ID_Building"])
        return self.building_names[building_id]

    def get_folder_names(self, path: Path) -> list:
        dicts = []
        for child in path.iterdir():
            if child.is_dir():
                dicts.append(child.name)
        return dicts

    def grab_scenario_ids_for_price(self, id_price: int, cooling: int) -> list:
        ids = self.scenario_table.loc[self.scenario_table.loc[:, "ID_EnergyPrice"] == id_price &
                                      self.scenario_table.loc[:, "ID_SpaceCoolingTechnology"] == cooling, "ID_Scenario"]
        return list(ids)

    def prn_to_csv(self, path) -> None:
        file_names = ["HEAT_BALANCE.prn", "TEMPERATURES.prn"]
        for name in file_names:
            table = pd.read_table(Path(path) / Path(name))
            column_name = list(table.columns)[0].replace("#", "").split(" ")
            column_names = [word for word in column_name if word != ""]

            # create new dataframe:
            df = pd.DataFrame(columns=column_names)

            all_values = table.iloc[:, 0]
            for i, entry in enumerate(all_values):
                # split entries
                row = [float(number) for number in entry.split(" ") if number != ""]
                df = df.append(pd.DataFrame([row], columns=column_names), ignore_index=True)

            # df to csv:
            df.to_csv(Path(path) / Path(name.replace("prn", "csv")), sep=";", index=False)

    def load_csv_heat_balance_file(self, path) -> (np.array, np.array):
        file_name = "HEAT_BALANCE.csv"
        table = pd.read_csv(Path(path) / Path(file_name), sep=";")
        table = table[~table.loc[:, "time"].duplicated(keep='first')]
        qhc2zone = table.loc[:, "qhc2zone"].to_numpy()[1:]  # drop first hour because daniel has 8761
        heating = qhc2zone.copy()
        heating[heating < 0] = 0
        cooling = qhc2zone.copy()
        cooling[cooling > 0] = 0
        return heating, cooling

    def load_csv_temperature_file(self, path):
        print(path)
        file_name = "TEMPERATURES.csv"
        table = pd.read_csv(Path(path) / Path(file_name), sep=";")
        table = table[~table.loc[:, "time"].duplicated(keep='first')]
        indoor_temp = table.loc[:, "tairmean"].to_numpy()[1:]
        return indoor_temp

    def iterate_through_folders(self, folder_path) -> List[str]:
        list_subfolders_paths = [f.path for f in os.scandir(folder_path) if f.is_dir()
                                 and f.path.split("\\")[-1] != "heating" and f.path.split("\\")[-1] != "cooling"]
        return list_subfolders_paths

    def create_csvs(self, system: str, price_scen: str):
        logging.info(f"loading {price_scen}, {system}")
        folders = self.iterate_through_folders(self.main_path / system / price_scen)
        heating_demand = {}
        cooling_demand = {}
        indoor_temp = {}
        for folder in folders:
            # load folders of zones:
            zone_paths = self.iterate_through_folders(Path(folder))
            house_heat_load = np.zeros((8760,))
            house_cool_load = np.zeros((8760,))

            number_of_zones = 0
            air_temp = np.zeros((8760,))
            # get heat load from each zone:
            for zone in zone_paths:
                number_of_zones += 1
                # create the csv files:
                self.prn_to_csv(zone)
                # load the csv file:
                zone_heat_load, zone_cool_load = self.load_csv_heat_balance_file(zone)
                house_heat_load += zone_heat_load
                house_cool_load += zone_cool_load

                air_temp_zone = self.load_csv_temperature_file(zone)
                air_temp += air_temp_zone
            # divide air temp by number of zones to get mean
            air_temp_mean = air_temp / number_of_zones

            # house name:
            house_name = folder.split("\\")[-1]
            heating_demand[house_name] = house_heat_load
            cooling_demand[house_name] = house_cool_load
            indoor_temp[house_name] = air_temp_mean

        # heating demand to csv for later analysis:
        heating_demand_df = pd.DataFrame(heating_demand)
        heating_demand_df.to_csv(self.main_path / Path(f"heating_demand_daniel_{system}_{price_scen}.csv"), sep=";",
                                 index=False)
        # cooling demand to csv:
        cooling_demand_df = pd.DataFrame(cooling_demand)
        cooling_demand_df.to_csv(self.main_path / f"cooling_demand_daniel_{system}_{price_scen}.csv", sep=";",
                                 index=False)
        # air temp to csv
        indoor_temp_df = pd.DataFrame(indoor_temp)
        indoor_temp_df.to_csv(self.main_path / Path(f"indoor_temp_daniel_{system}_{price_scen}.csv"), sep=";",
                              index=False)

    def main(self):
        heating_systems = ["ideal", "floor_heating"]
        multi_list = []
        for system in heating_systems:
            strategies = self.get_folder_names(self.main_path / system)
            for strategy in strategies:
                multi_list.append((system, strategy))

        with multiprocessing.Pool(processes=6) as pool:
            pool.map(self.create_csvs, multi_list)

    def read_heat_demand(self, table_name: str, prize_scenario: str, cooling: int):
        scenario_id = self.grab_scenario_ids_for_price(int(prize_scenario.replace("_cooling", "")[-1]), cooling)
        demand = DB(get_config(self.project_name)).read_dataframe(table_name=table_name,
                                        column_names=["ID_Scenario", "Q_HeatingTank_bypass", "Hour"],
                                        )
        # split the demand into columns for every Scenario ID:
        df = demand.pivot_table(index="Hour", columns="ID_Scenario")
        # drop columns which are not in this price
        df = df.iloc[:, [scen_id - 1 for scen_id in scenario_id]]
        df.columns = [self.scenario_id_2_building_name(scen_id) for scen_id in scenario_id]
        return df

    def modify_heat_demand(self):

        # Solution: Whenever the heat demand in the opt scenario = 0, the heat demand of the IDA ICE opt will be set to
        # the same heat demand the IDA ICE model has in the reference scenario

        # load heat demand from first price scenario
        filename = f"heating_demand_daniel_ideal_price1.csv"
        ref_demand_df = pd.read_csv(self.main_path / Path(filename), sep=";")
        heating_systems = ["ideal", "floor_heating"]
        for system in heating_systems:
            prices = self.get_folder_names(self.main_path / system)
            for price in prices:
                if "cooling" in price:
                    cooling = 2
                else:
                    cooling = 1
                # load heat demand from other price scenarios:
                filename = f"heating_demand_daniel_{system}_{price}.csv"
                opt_demand_IDA = pd.read_csv(self.main_path / Path(filename), sep=";")

                # load heat demand from 5R1C model optimized:
                opt_demand_5R1C = self.read_heat_demand(OperationTable.ResultOptHour.value, price, cooling)

                # order the columns of both dfs:
                column_order = list(opt_demand_IDA.columns)
                opt_demand_5R1C = opt_demand_5R1C[column_order].reset_index(drop=True)

                # check each hour if the heat demand is 0 in the 5R1C model and above 0 in the IDA. If this is the case
                # the heat demand in the IDA model will be exchanged with the heat demand of the IDA model without
                # optimization:
                for column_name, series in opt_demand_IDA.iteritems():
                    for index, value in enumerate(series):
                        if opt_demand_5R1C.loc[index, column_name] == 0 and value > 0:
                            # exchange with the ref demand from IDA:
                            opt_demand_IDA.loc[index, column_name] = ref_demand_df.loc[index, column_name]

                # save opt_demand_IDA as csv under the old name
                opt_demand_IDA.to_csv(self.main_path / Path(f"heating_demand_daniel_{price}.csv"), sep=";", index=False)

    def clean_up(self):
        """ delete the csv files that only contain zeros (cooling demand for scenarios without cooling)"""
        for file in self.main_path.glob('*.csv'):
            if file.name.lower().startswith("cooling"):
                if file.name.lower().count("cooling") == 1:
                    file.unlink()


if __name__ == "__main__":
    prn_importer = PRNImporter("5R1C_validation")
    prn_importer.main()
    prn_importer.clean_up()
    # prn_importer.modify_heat_demand()
