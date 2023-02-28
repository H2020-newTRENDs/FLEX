from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario


class Person:
    def __init__(self, scenario: "BehaviorScenario", id_person_type):
        self.scenario = scenario
        self.timeslot_num = 144  # each timeslot represents 10min, in total 144 timeslots in a day
        self.id_person_type = id_person_type
        self.activity_profile = []
        self.electricity_demand = []
        self.hot_water_demand = []

    def setup(self):
        self.setup_activity_profile()
        self.setup_electricity_demand_profile()
        self.setup_hot_water_demand_profile()

    def setup_activity_profile(self):
        for day in range(2, 367):  # The year of 2019 started with Tuesday
            print(f'day = {day}')
            id_day_type = self.scenario.day_type[day % 7]
            sleep_duration = self.scenario.get_activity_duration(self.id_person_type, id_day_type, 1, 1)
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

    def setup_electricity_demand_profile(self):
        for id_activity in self.activity_profile:
            # id_technology = self.scenario.get_activity_technology(id_activity)
            # technology
            self.electricity_demand.append(1)

    def setup_hot_water_demand_profile(self):
        for id_activity in self.activity_profile:
            # id_technology = self.scenario.get_activity_technology(id_activity)
            # technology
            self.hot_water_demand.append(1)
