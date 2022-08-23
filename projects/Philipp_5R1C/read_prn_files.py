from pathlib import Path

import numpy as np
import pandas as pd
import os
from typing import List


class PRNImporter:
    def __init__(self):
        self.main_path = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data")

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
        list_subfolders_paths = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        return list_subfolders_paths

    def main(self):
        strategies = ["price_1", "price_2", "price_3", "price_4"]
        for strat in strategies:
            folders = self.iterate_through_folders(self.main_path / Path(strat))
            heating_demand = {}
            indoor_temp = {}
            for folder in folders:
                # load folders of zones:
                zone_paths = self.iterate_through_folders(Path(folder))
                house_heat_load = np.zeros((8760, ))

                number_of_zones = 0
                air_temp = np.zeros((8760, ))
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
            heating_demand_df.to_csv(self.main_path / Path(f"heating_demand_daniel_{strat}.csv"), sep=";")
            # air temp to csv
            indoor_temp_df = pd.DataFrame(indoor_temp)
            indoor_temp_df.to_csv(self.main_path / Path(f"indoor_temp_daniel_{strat}.csv"), sep=";")


if __name__ == "__main__":
    PRNImporter().prn_to_csv(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data\optimized\EZFH_5_B\zone")
    # PRNImporter().main()