from pathlib import Path

import pandas as pd

from flex.config import Config
from flex.db import create_db_conn
from flex.plotter import Plotter
from flex_behavior.constants import BehaviorTable


class BehaviorScenario:

    def __init__(self, scenario_id: int, config: "Config"):
        self.scenario_id = scenario_id
        self.config = config
        self.db = create_db_conn(self.config)
        self.plotter = Plotter(config)

    def import_dataframe(self, file_name: str):
        if not self.db.if_exists(file_name):
            df = pd.read_excel(self.config.input_behavior / Path(file_name + ".xlsx"), engine="openpyxl")
            self.db.write_dataframe(file_name, df, if_exists='replace')
        df = self.db.read_dataframe(file_name)
        return df

    def import_scenario_data(self):
        self.tou_profile = self.import_dataframe(BehaviorTable.ToUProfile)



