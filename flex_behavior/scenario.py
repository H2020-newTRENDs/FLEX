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

    def setup(self):
        self.setup_scenario_params()
        self.import_scenario_data()
        self.generate_activity_transition_matrix()

    def setup_scenario_params(self):
        behavior_scenarios = self.db.read_dataframe(BehaviorTable.Scenarios)
        params_dict = behavior_scenarios.loc[behavior_scenarios["ID_Scenario"] == self.scenario_id].iloc[0].to_dict()
        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    def import_scenario_data(self):
        self.activity_profile = self.db.read_dataframe(BehaviorTable.ActivityProfile)

    def generate_activity_transition_matrix(self):
        ...

