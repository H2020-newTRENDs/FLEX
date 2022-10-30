from pathlib import Path

import numpy as np
import pandas as pd
import os
from typing import List
from basics.db import DB
from config import config, get_config
from models.operation.enums import OperationTable



class PRNImporter:
    def __init__(self, project_name):

        self.project_name = project_name
        self.main_path = Path(get_config(self.project_name).project_root / Path(f"projects/{project_name}/input"))
        self.building_names = {
            1: "EZFH_5_B",
            2: "EZFH_5_S",
            3: "EZFH_9_B",
            4: "EZFH_1_B",
            5: "EZFH_1_S"
        }
        self.scenario_table = DB(get_config(self.project_name)).read_dataframe(table_name=OperationTable.Scenarios.value)

    def scenario_id_2_building_name(self, id_scenario: int) -> str:
        building_id = int(
            self.scenario_table.loc[self.scenario_table.loc[:, "ID_Scenario"] == id_scenario, "ID_Building"])
        return self.building_names[building_id]

    def grab_scenario_ids_for_price(self, id_price: int) -> list:
        ids = self.scenario_table.loc[self.scenario_table.loc[:, "ID_EnergyPrice"] == id_price, "ID_Scenario"]
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

    def load_csv_heat_balance_file(self, path):
        file_name = "HEAT_BALANCE.csv"
        table = pd.read_csv(Path(path) / Path(file_name), sep=";")
        table = table[~table.loc[:, "time"].duplicated(keep='first')]
        heating = table.loc[:, "qhc2zone"].to_numpy()[1:]  # drop first hour because daniel has 8761
        return heating

    def load_csv_temperature_file(self, path):
        print(path)
        file_name = "TEMPERATURES.csv"
        table = pd.read_csv(Path(path) / Path(file_name), sep=";")
        table = table[~table.loc[:, "time"].duplicated(keep='first')]
        indoor_temp = table.loc[:, "tairmean"].to_numpy()[1:]
        return indoor_temp

    def iterate_through_folders(self, folder_path) -> List[str]:
        list_subfolders_paths = [f.path for f in os.scandir(folder_path) if f.is_dir() and f.path.split("\\")[-1]!="heating"]
        return list_subfolders_paths

    def main(self):
        strategies = ["price1", "price2", "price3", "price4"]
        for strat in strategies:
            folders = self.iterate_through_folders(self.main_path / Path(strat))
            heating_demand = {}
            indoor_temp = {}
            for folder in folders:
                # load folders of zones:
                zone_paths = self.iterate_through_folders(Path(folder))
                house_heat_load = np.zeros((8760,))

                number_of_zones = 0
                air_temp = np.zeros((8760,))
                # get heat load from each zone:
                for zone in zone_paths:
                    number_of_zones += 1
                    # create the csv files:
                    self.prn_to_csv(zone)
                    # load the csv file:
                    zone_heat_load = self.load_csv_heat_balance_file(zone)
                    house_heat_load += zone_heat_load

                    air_temp_zone = self.load_csv_temperature_file(zone)
                    air_temp += air_temp_zone
                # divide air temp by number of zones to get mean
                air_temp_mean = air_temp / number_of_zones

                # house name:
                house_name = folder.split("\\")[-1]
                heating_demand[house_name] = house_heat_load
                indoor_temp[house_name] = air_temp_mean

            # heating demand to csv for later analysis:
            heating_demand_df = pd.DataFrame(heating_demand)
            heating_demand_df.to_csv(self.main_path / Path(f"heating_demand_daniel_{strat}.csv"), sep=";", index=False)
            # air temp to csv
            indoor_temp_df = pd.DataFrame(indoor_temp)
            indoor_temp_df.to_csv(self.main_path / Path(f"indoor_temp_daniel_{strat}.csv"), sep=";", index=False)

        # self.modify_heat_demand()

    def read_heat_demand(self, table_name: str, prize_scenario: str):
        scenario_id = self.grab_scenario_ids_for_price(int(prize_scenario[-1]))
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
        # TODO for the optimized profiles, if the heat demand in the 5R1C model is zero, it has to be zero for the IDA ICE
        #  model as well. The indoor temp in the 5R1C model is input to the IDA ICE. After the summer, the temperature in
        #  the 5R1C is much higher, causing IDA ICE to heat even though it is not nessecary.

        # Solution: Whenever the heat demand in the opt scenario = 0, the heat demand of the IDA ICE opt will be set to
        # the same heat demand the IDA ICE model has in the reference scenario

        # load heat demand from first price scenario
        filename = f"heating_demand_daniel_price1.csv"
        ref_demand_df = pd.read_csv(self.main_path / Path(filename), sep=";")

        for price in ["price2", "price3", "price4"]:
            # load heat demand from other price scenarios:
            filename = f"heating_demand_daniel_{price}.csv"
            opt_demand_IDA = pd.read_csv(self.main_path / Path(filename), sep=";")

            # load heat demand from 5R1C model optimized:
            opt_demand_5R1C = self.read_heat_demand(OperationTable.ResultOptHour.value, price)

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


if __name__ == "__main__":
    prn_importer = PRNImporter("5R1C_validation")
    prn_importer.main()
    prn_importer.modify_heat_demand()
