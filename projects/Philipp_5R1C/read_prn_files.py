from pathlib import Path

import numpy as np
import pandas as pd
import os
from typing import List


class PRNImporter:
    def __init__(self):
        self.main_path = Path(r"C:\Users\mascherbauer\PycharmProjects\NewTrends\Prosumager\projects\Philipp_5R1C\input_data")


    def load_csv_heat_balance_file(self, path):
        file_name = "HEAT_BALANCE.csv"
        table = pd.read_csv(Path(path) / Path(file_name), sep=";")
        heating = table.loc[:, "qhc2zone"].to_numpy()[1:]  # drop first hour because daniel has 8761
        return heating

    def iterate_through_folders(self, folder_path) -> List[str]:
        list_subfolders_paths = [f.path for f in os.scandir(folder_path) if f.is_dir()]
        return list_subfolders_paths

    def main(self):
        folders = self.iterate_through_folders(self.main_path)
        heating_demand = {}
        for folder in folders:
            # load folders of zones:
            zone_paths = self.iterate_through_folders(Path(folder))
            house_heat_load = np.zeros((8760, ))
            # get heat load from each zone:
            for zone in zone_paths:
                # load the csv file:
                zone_heat_load = self.load_csv_heat_balance_file(zone)
                house_heat_load += zone_heat_load

            # house name:
            house_name = folder.split("\\")[-1]
            heating_demand[house_name] = house_heat_load

        # heating demand to csv for later analysis:
        heating_demand_df = pd.DataFrame(heating_demand)
        heating_demand_df.to_csv(self.main_path / Path("heating_demand_daniel.csv"), sep=";")



if __name__ == "__main__":
    PRNImporter().main()