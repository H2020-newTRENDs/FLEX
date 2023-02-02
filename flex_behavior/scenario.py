from typing import Optional

from flex.config import Config
from flex.db import create_db_conn
from flex.plotter import Plotter
from flex_behavior.constants import BehaviorTable
from flex import kit


class BehaviorScenario:

    def __init__(self, scenario_id: int, config: "Config"):
        self.scenario_id = scenario_id
        self.config = config
        self.db = create_db_conn(self.config)
        self.plotter = Plotter(config)
        self.period_num = 8760
        self.id_household_type: Optional[int] = None

    def setup(self):
        self.setup_scenario_params()
        self.import_scenario_data()

    def setup_scenario_params(self):
        df = self.db.read_dataframe(BehaviorTable.Scenarios)
        df.set_index("ID_Scenario", inplace=True)
        params_dict = df.loc[self.scenario_id].to_dict()
        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    def import_scenario_data(self):
        self.household_composition = self.db.read_dataframe(BehaviorTable.HouseholdComposition)
        self.activity_profile = self.db.read_dataframe(BehaviorTable.ActivityProfile)
        self.technology_trigger_prob = self.db.read_dataframe(BehaviorTable.TechnologyTriggerProbability)
        self.technology_power_active = self.db.read_dataframe(BehaviorTable.TechnologyPowerActive)
        self.technology_power_standby = self.db.read_dataframe(BehaviorTable.TechnologyPowerStandby)

    def get_household_composition(self, id_household_type: int):
        df = self.household_composition.loc[self.household_composition["ID_HouseholdType"] == id_household_type]
        d = {}
        for _, row in df.iterrows():
            d[row["ID_PersonType"]] = row["value"]
        return d

    def get_activity_technology(self, id_activity: int):
        df = self.technology_power_standby.loc[self.technology_power_standby["ID_Activity"] == id_activity]
        techs = {}
        for _, row in df.iterrows():
            techs[row["ID_Technology"]] = row["value"]
        return kit.dict_sample(techs)




















