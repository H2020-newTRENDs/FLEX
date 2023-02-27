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
        # imported from excel input: BehaviorScenario
        self.id_household_type: Optional[int] = None
        self.setup()

    def setup(self):
        self.setup_scenario_params()
        self.import_scenario_data()
        self.setup_day_type()

    def setup_scenario_params(self):
        df = self.db.read_dataframe(BehaviorTable.Scenarios)
        df.set_index("ID_Scenario", inplace=True)
        params_dict = df.loc[self.scenario_id].to_dict()
        for key, value in params_dict.items():
            if key in self.__dict__.keys():
                self.__setattr__(key, value)

    def import_scenario_data(self):
        self.household_composition = self.db.read_dataframe(BehaviorTable.HouseholdComposition)
        self.activity_change_prob = self.db.read_dataframe(BehaviorTable.ActivityChangeProb)
        self.activity_duration_prob = self.db.read_dataframe(BehaviorTable.ActivityDurationProb)
        self.technology_trigger_prob = self.db.read_dataframe(BehaviorTable.TechnologyTriggerProbability)
        self.technology_power_active = self.db.read_dataframe(BehaviorTable.TechnologyPowerActive)
        self.technology_power_standby = self.db.read_dataframe(BehaviorTable.TechnologyPowerStandby)

    def setup_day_type(self):
        self.day_type = {
            1: 1,  # Monday
            2: 1,  # Tuesday
            3: 1,  # Wednesday
            4: 1,  # Thursday
            5: 2,  # Friday
            6: 3,  # Saturday
            0: 4,  # Sunday --> weekday % 7 = 0
        }

    @staticmethod
    def filter_dataframe_smart(df, columns, values):
        while len(df[df[columns] == values]) == 0 and len(columns) > 1:
            columns = columns[:-1]
            values = values[:-1]
        filtered_df = df[df[columns] == values]
        print()
        return filtered_df

    def get_activity_duration(self, id_person_type: int, id_day_type: int, id_activity: int, timeslot: int):
        columns = ["ID_Activity", "t", "ID_PersonType", "ID_DayType"]
        values = [id_activity, timeslot, id_person_type, id_day_type]
        df = self.filter_dataframe_smart(self.activity_duration_prob.copy(), columns, values)
        d = {}
        for index, row in df.iterrows():
            d[row["duration"]] = row["probability"]
        return int(kit.dict_sample(d))

    def get_activity_now(self, id_person_type: int, id_day_type: int, id_activity_before: int, timeslot: int):
        columns = ["ID_ActivityBefore", "t", "ID_PersonType", "ID_DayType"]
        values = [id_activity_before, timeslot, id_person_type, id_day_type]
        df = self.filter_dataframe_smart(self.activity_change_prob.copy(), columns, values)
        d = {}
        for index, row in df.iterrows():
            d[row["ID_ActivityNow"]] = row["probability"]
        return int(kit.dict_sample(d))

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




















