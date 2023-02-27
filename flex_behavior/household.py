from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario


class Person:
    def __init__(self, scenario: "BehaviorScenario", id_person_type):
        self.scenario = scenario
        self.timeslot_num = 144  # each timeslot represents 10min, in total 144 timeslots in a day
        self.id_person_type = id_person_type

    def setup(self):
        self.setup_activity_profile()
        self.setup_building_occupancy_profile()
        self.setup_electricity_demand_profile()
        self.setup_hot_water_demand_profile()
        self.setup_driving_profile()

    def setup_activity_profile(self):
        self.activity_profile = []
        for day in range(1, 366):
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

    def setup_building_occupancy_profile(self):
        self.building_occupancy = np.ones(self.scenario.period_num)

    def setup_electricity_demand_profile(self):
        ...

    def setup_base_electricity_demand_profile(self):
        ...

    def setup_activity_electricity_demand_profile(self):
        self.electricity_demand = np.zeros(self.scenario.period_num)
        for id_activity in self.activity_profile:
            id_technology = self.scenario.get_activity_technology(id_activity)
            # technology

    def setup_hot_water_demand_profile(self):
        self.hot_water_demand = np.zeros(self.scenario.period_num)

    def setup_driving_profile(self):
        self.driving_profile = np.zeros(self.scenario.period_num)


class Household:
    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario
        self.id_household_type = scenario.id_household_type

    def setup(self):
        self.setup_household_persons()
        self.setup_building_occupancy_profile()
        self.setup_electricity_demand_profile()
        self.setup_hot_water_demand_profile()
        self.setup_driving_profile()

    def setup_household_persons(self):
        self.persons = []
        composition = self.scenario.get_household_composition(self.id_household_type)
        for id_person_type, person_num in composition.items():
            for i in range(0, person_num):
                person = Person(self.scenario, id_person_type)
                person.setup()
                self.persons.append(person)

    def setup_building_occupancy_profile(self):
        self.building_occupancy = np.zeros(self.scenario.period_num)
        for hour in range(0, self.scenario.period_num):
            for person in self.persons:
                if person.building_occupancy[hour] == 1:
                    self.building_occupancy[hour] = 1
                    break

    def setup_electricity_demand_profile(self):
        self.electricity_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.electricity_demand += person.electricity_demand

    def setup_hot_water_demand_profile(self):
        self.hot_water_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.hot_water_demand += person.hot_water_demand

    def setup_driving_profile(self):
        # logic to be decided
        self.driving_profile = np.zeros(self.scenario.period_num)





