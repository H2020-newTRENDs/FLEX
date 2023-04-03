from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario

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
        day = 2  # The year of 2019 started with Tuesday
        id_day_type = self.scenario.day_type[day % 7]
        init_timeslot = 121  # 0:00 am
        sleep_duration = self.scenario.get_activity_duration(self.id_person_type, id_day_type, 1, init_timeslot)
        activities = [1] * sleep_duration  # sleeping, at the beginning of the day (0:00 am)
        day = (len(activities) // self.timeslot_num) + 2  # + 1 to get days from 1 to 365, +1 bc yaer starts at day 2
        prev_day = day
        while day < 367:
            if day != prev_day:
                print('day:', day)
                prev_day = day
            id_day_type = self.scenario.day_type[day % 7]
            timeslot = (init_timeslot + len(activities)) % self.timeslot_num + 1
            id_activity_before = activities[-1]
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
            activities.extend([id_activity_now] * duration)
            day = (len(activities) // self.timeslot_num) + 2 # + 1 to get days from 1 to 365, +1 bc year starts at day 2
        activities = activities[:(365 * self.timeslot_num)]
        self.activity_profile = activities

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

            if tec_duration < 1:
                technology_power = technology_power * tec_duration
                tec_duration = 1

            for idx in range(int(tec_duration)):
                if id_technology == 25:  # hot water was triggered
                    self.hot_water_demand[timeslot + idx] += technology_power
                else:
                    self.electricity_demand[timeslot + idx] += technology_power

