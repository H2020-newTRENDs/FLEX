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
        self.person_num: int = None

    def import_behavior_dataframe(self, file_name: str):
        if not self.db.if_exists(file_name):
            df = pd.read_excel(self.config.input_behavior / Path(file_name + ".xlsx"), engine="openpyxl")
            self.db.write_dataframe(file_name, df, if_exists='replace')
        df = self.db.read_dataframe(file_name)
        return df

    def setup_scenario_params(self):
        df = self.import_behavior_dataframe(BehaviorTable.Scenarios)
        params_dict = df.loc[df["ID_Scenario"] == self.scenario_id].iloc[0].to_dict()
        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    def import_scenario_data(self):
        self.activity_profile = self.import_behavior_dataframe(BehaviorTable.ActivityProfile)

    def generate_activity_transition_prob_matrix(self):
        ...

