from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario

from flex.config import Config
from flex_behavior.scenario import BehaviorScenario


class Person:
    def __init__(self, scenario: "BehaviorScenario", id_person_type):
        self.scenario = scenario
        self.timeslot_num = 144  # each timeslot represents 10min, in total 144 timeslots in a day
        self.id_person_type = id_person_type
        self.activity_profile = []
        self.technology_ids = []
        self.electricity_demand = []
        self.hot_water_demand = []

    def setup(self):
        self.setup_activity_profile()
        self.setup_electricity_and_hotwater_demand_profile()

    def setup_activity_profile(self):
        for day in range(2, 367):  # The year of 2019 started with Tuesday
            print(f'day = {day}')
            id_day_type = self.scenario.day_type[day % 7]
            sleep_duration = self.scenario.get_activity_duration(self.id_person_type, id_day_type, 1, 120)
            day_activities = [1] * sleep_duration  # sleeping, at the beginning of the day (4:00 am)
            while len(day_activities) <= self.timeslot_num:
                timeslot = len(day_activities) + 1
                id_activity_before = day_activities[-1]
                id_activity_now = self.scenario.get_activity_now(
                    self.id_person_type,
                    id_day_type,
                    id_activity_before,
                    timeslot
                )
                duration = self.scenario.get_activity_duration(
                    self.id_person_type,
                    id_day_type,
                    id_activity_now,
                    timeslot
                )
                day_activities.extend([id_activity_now] * duration)
            day_activities = day_activities[0: 144]
            day_activities = day_activities[-24:] + day_activities[:-24]
            self.activity_profile.extend(day_activities)

    def setup_electricity_and_hotwater_demand_profile(self):
        self.electricity_demand = [0] * len(self.activity_profile)
        self.hot_water_demand = [0] * len(self.activity_profile)

        for timeslot, id_activity in enumerate(self.activity_profile):
            id_technology = self.scenario.get_activity_technology(id_activity)
            self.technology_ids.append(id_technology)
            technology_power = self.scenario.get_technology_power(id_technology)

            tec_duration = self.scenario.get_technology_duration(id_technology)
            if timeslot + tec_duration > len(self.activity_profile):
                tec_duration = len(self.activity_profile) - timeslot  # duration of technology ends with end of day (if longer)

            for idx in range(tec_duration):
                if id_technology == 25:  # hot water was triggered
                    self.hot_water_demand[timeslot + idx] += technology_power
                else:
                    self.electricity_demand[timeslot + idx] += technology_power

if __name__ == "__main__":
    cfg = Config(project_name="FLEX_Behavior")
    scenario = BehaviorScenario(scenario_id=1, config=cfg)
    person = Person(scenario, 1)
    person.setup_electricity_and_hotwater_demand_profile()
