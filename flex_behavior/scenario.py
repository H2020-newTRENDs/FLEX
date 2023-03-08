from typing import Optional
import random

from flex.config import Config
from flex.db import create_db_conn
from flex.plotter import Plotter
from flex_behavior.constants import BehaviorTable
from flex import kit
from collections import OrderedDict


class BehaviorScenario:

    def __init__(self, scenario_id: int, config: "Config"):
        self.scenario_id = scenario_id
        self.config = config
        self.db = create_db_conn(self.config)
        self.plotter = Plotter(config)
        self.period_num = 8760
        # imported from excel input: BehaviorScenario
        self.id_household_type: Optional[int] = None
        self.wfh_share: Optional[float] = None
        self.setup()

    def setup(self):
        self.setup_scenario_params()
        self.import_scenario_data()
        self.setup_day_type()
        self.setup_activity_location()

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
        self.technology_power_active = self.db.read_dataframe(BehaviorTable.TechnologyPower)
        self.technology_type = self.db.read_dataframe(BehaviorTable.TechnologyType)
        self.technology_duration = self.db.read_dataframe(BehaviorTable.TechnologyDuration)

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

    def setup_activity_location(self):
        df = self.db.read_dataframe(BehaviorTable.ActivityLocation)
        self.activity_location = {}
        for i, row in df.iterrows():
            self.activity_location[row["ID_Activity"]] = row["ID_Location"]

    @staticmethod
    def filter_dataframe_dynamic(df, od_filter: "OrderedDict"):
        while len(kit.filter_df(df, od_filter)) == 0 and len(od_filter) > 1:
            od_filter.popitem()
        filtered_df = kit.filter_df(df, od_filter)
        return filtered_df

    def get_activity_duration(self, id_person_type: int, id_day_type: int, id_activity: int, timeslot: int):
        od = OrderedDict([
            ("ID_Activity", id_activity),
            ("t", timeslot),
            ("ID_DayType", id_day_type),
            ("ID_PersonType", id_person_type),
        ])  # these items are ranked from most to least important
        df = self.filter_dataframe_dynamic(self.activity_duration_prob.copy(), od)
        d = {}
        for index, row in df.iterrows():
            d[row["duration"]] = row["probability"]
        return int(kit.dict_sample(d))

    def get_activity_now(self, id_person_type: int, id_day_type: int, id_activity_before: int, timeslot: int):
        od = OrderedDict([
            ("ID_ActivityBefore", id_activity_before),
            ("t", timeslot),
            ("ID_DayType", id_day_type),
            ("ID_PersonType", id_person_type),
        ])  # these items are ranked from most to least important
        df = self.filter_dataframe_dynamic(self.activity_change_prob.copy(), od)
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
        df = self.technology_trigger_prob.loc[self.technology_trigger_prob["ID_Activity"] == id_activity]
        d = {}
        for index, row in df.iterrows():
            d[row["ID_Technology"]] = row["value"]
        return int(kit.dict_sample(d))

    def get_technology_power(self, id_technology: int):
        power = self.technology_power_active.loc[
            self.technology_power_active["ID_Technology"] == id_technology, ['value']]
        return power.iloc[0, 0]

    def get_technology_type(self, id_technology: int):
        tec_type = self.technology_type.loc[
            self.technology_type["ID_Technology"] == id_technology, ['ID_TechnologyType']]
        return tec_type.iloc[0, 0]

    def get_technology_duration(self, id_technology: int):
        duration = self.technology_duration.loc[
            self.technology_duration["ID_Technology"] == id_technology, ['value']]
        return duration.iloc[0, 0]

    def get_building_occupancy_by_hourly_activity(
            self,
            activities: list[int],
            work_from_home: int  # home = 1, outside = 0
    ):
        self.setup_activity_location()
        locations = [self.activity_location[activity] for activity in activities]  # translate activities to locations - 1: outside, 2: home, 3: home or outside, 4: commuting
        locations = [0 if loc == 4 else loc - 1 for loc in locations]  # commute is also not at home = 0, home = 1
        for idx, loc in enumerate(locations):
            if loc == 2:
                if activities[idx] == 2:  # eating and drinking, 50% probability to be from home/outside TODO keep this assumption?
                    rand = random.uniform(0, 1)
                    locations[idx] = 0 if rand < 0.5 else 1
                else:
                    locations[idx] = work_from_home  # working/education and breaks
        return locations
