import random
from typing import List, Optional
from typing import TYPE_CHECKING

import numpy as np

from flex_behavior.constants import BehaviorTable
from flex_behavior.scenario import BehaviorScenario

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario


class HouseholdPerson:
    def __init__(self, scenario: "BehaviorScenario", id_person_type):
        self.scenario = scenario
        self.id_person_type = id_person_type
        self.activity_ids_10min: Optional[List[float]] = None
        self.building_occupancy_profile_10min: Optional[List[float]] = None
        self.technology_ids_10min: Optional[List[float]] = None
        self.electricity_profile_10min: Optional[List[float]] = None
        self.hot_water_profile_10min: Optional[List[float]] = None
        self.electricity_profile: Optional[List[float]] = None
        self.hot_water_profile: Optional[List[float]] = None
        self.building_occupancy_profile: Optional[List[int]] = None

    def setup_household_person(self):
        self.read_profiles()
        self.setup_building_occupancy_profile()
        self.setup_electricity_and_hot_water_profile()

    def read_profiles(self):
        df = self.scenario.db.read_dataframe(BehaviorTable.PersonProfiles)
        sample = 0  # TODO generate randomly
        self.activity_ids_10min = df[f"activity_p{self.id_person_type}s{sample}"].tolist()
        self.technology_ids_10min = df[f"id_technology_p{self.id_person_type}s{sample}"].tolist()
        self.electricity_profile_10min = df[f"electricity_p{self.id_person_type}s{sample}"].tolist()
        self.hot_water_profile_10min = df[f"hotwater_p{self.id_person_type}s{sample}"].tolist()

    def setup_building_occupancy_profile(self):
        # sets building_occupancy_profile with list of 1 and 0: 1 meaning at home, 0 meaning outside
        # do this based on WFH assumptions, which are parameters of scenario
        # we also need the length and starting point of the working/education time window
        # in the working/education time window are possible outside, which is decided by WFH assumption. For the other activities, we decided based on the table (if home_or_outside, 0.5-0.5).
        # aggregate to hourly resolution (based on assumption --> more than 30min in an hour, it is counted as at-home or outside; equaling to 30min, we use 0.5-0.5.

        def generate_wfh(whf_probability):
            rand = random.uniform(0, 1)
            wfh = 0 if rand < whf_probability else 1
            return wfh

        min_activity = self.activity_ids_10min
        min_occupancy = []
        hour_occupancy = []
        work_outside_prob = 1 - self.scenario.wfh_share

        for i in range(int(len(min_activity) / 6)):
            if i % 24 == 0:  # start of new day
                wfh = generate_wfh(work_outside_prob)
            hour_activity = min_activity[i*6:i*6 + 6]
            min_occupancy_in_hour = self.scenario.get_building_occupancy_by_hourly_activity(hour_activity, wfh)
            min_occupancy.extend(min_occupancy_in_hour)
            sum_occupancy = sum(min_occupancy_in_hour)/6
            if sum_occupancy == 0.5:  # 30 min home, 30 min outside
                rand = random.uniform(0, 1)
                occupancy = 0 if rand < 0.5 else 1  # occupancy based on where person was most in this hour
            else:
                occupancy = round(sum_occupancy)
            hour_occupancy.append(occupancy)
        self.building_occupancy_profile_10min = min_occupancy
        self.building_occupancy_profile = hour_occupancy

    def setup_electricity_and_hot_water_profile(self):
        # lighting is added to electricity demand, if person at home and not sleeping
        self.add_electricity_demand_for_lighting()

        min_electricity = self.electricity_profile_10min
        hour_electricity = []

        min_hot_water = self.hot_water_profile_10min
        hour_hot_water = []

        # aggregate to hourly resolution
        for i in range(int(len(min_electricity) / 6)):
            hour_electricity.append(sum(min_electricity[i*6:i*6 + 6])/6)
            hour_hot_water.append(sum(min_hot_water[i*6:i*6+6])/6)

        self.electricity_profile = hour_electricity
        self.hot_water_profile = hour_hot_water

    def add_electricity_demand_for_lighting(self):
        idx = [self.activity_ids_10min[i] != 1 and self.building_occupancy_profile_10min[i] == 1 for i in range(len(self.activity_ids_10min))]
        power = self.scenario.get_technology_power(36)  # electricity demand of lightning
        lightning = [power if i else 0 for i in idx]  # if person at home and not sleeping -> use the light
        self.electricity_profile_10min = [x + y for x, y in zip(self.electricity_profile_10min, lightning)]


class Household:
    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario
        self.id_household_type = scenario.id_household_type
        # to be set up below
        self.persons: List["HouseholdPerson"] = []

    def setup(self):
        self.setup_household_persons()
        self.setup_building_occupancy_profile()
        self.setup_electricity_demand_profile()
        self.setup_hot_water_demand_profile()

    def setup_household_persons(self):
        composition = self.scenario.get_household_composition(self.id_household_type)
        for id_person_type, person_num in composition.items():
            for i in range(0, person_num):
                person = HouseholdPerson(self.scenario, id_person_type)
                person.setup_household_person()
                self.persons.append(person)

    def setup_building_occupancy_profile(self):
        self.building_occupancy = np.zeros(self.scenario.period_num)
        for hour in range(0, self.scenario.period_num):
            for person in self.persons:
                if person.building_occupancy_profile[hour] == 1:
                    self.building_occupancy[hour] = 1
                    break

    def setup_household_base_electricity_demand_profile(self):
        # Add base load - 37, 39 - always on
        power_modem = self.scenario.get_technology_power(35)  # electricity demand of internet_modem
        power_refrigerator_freezer = self.scenario.get_technology_power(37)  # electricity demand of refrigerator_freezer_combi
        self.base_electricity_demand = (power_modem + power_refrigerator_freezer) * np.ones(self.scenario.period_num)

    def setup_electricity_demand_profile(self):
        self.setup_household_base_electricity_demand_profile()
        self.electricity_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.electricity_demand += person.electricity_profile
        self.electricity_demand += self.base_electricity_demand

    def setup_hot_water_demand_profile(self):
        self.hot_water_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.hot_water_demand += person.hot_water_profile

