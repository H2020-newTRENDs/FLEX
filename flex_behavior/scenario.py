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

    def setup_scenario_params(self):
        behavior_scenarios = self.db.read_dataframe(BehaviorTable.Scenarios)
        params_dict = behavior_scenarios.loc[behavior_scenarios["ID_Scenario"] == self.scenario_id].iloc[0].to_dict()
        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    def import_scenario_data(self):
        self.id_activity = self.db.read_dataframe(BehaviorTable.ID_Activity)
        self.id_technology = self.db.read_dataframe(BehaviorTable.ID_Technology)
        self.activity_profile = self.db.read_dataframe(BehaviorTable.ActivityProfile)
        self.technology_trigger_prob = self.db.read_dataframe(BehaviorTable.TechnologyTriggerProbability)
        self.technology_ownership_rate = self.db.read_dataframe(BehaviorTable.TechnologyOwnershipRate)
        self.technology_power_active = self.db.read_dataframe(BehaviorTable.TechnologyPowerActive)
        self.technology_power_standby = self.db.read_dataframe(BehaviorTable.TechnologyPowerStandby)






















