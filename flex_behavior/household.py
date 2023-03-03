from typing import List, Optional
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from flex_behavior.scenario import BehaviorScenario


class HouseholdPerson:
    def __init__(self, scenario: "BehaviorScenario", id_person_type):
        self.scenario = scenario
        self.id_person_type = id_person_type
        self.electricity_profile: Optional[List[float]] = None
        self.hot_water_profile: Optional[List[float]] = None
        self.building_occupancy_profile: Optional[List[int]] = None

    def setup_electricity_and_hot_water_profile(self):
        # read from databases and initialize electricity_profile and hot_water_profile
        # aggregate to hourly resolution
        ...

    def setup_building_occupancy_profile(self, scenario: "BehaviorScenario"):
        # sets building_occupancy_profile with list of 1 and 0: 1 meaning at home, 0 meaning outside
        # do this based on WFH assumptions, which are parameters of scenario
        # we also need the length and starting point of the working/education time window
        # in the working/education time window are possible outside, which is decided by WFH assumption. For the other activities, we decided based on the table (if home_or_outside, 0.5-0.5).
        # aggregate to hourly resolution (based on assumption --> more than 30min in an hour, it is counted as at-home or outside; equaling to 30min, we use 0.5-0.5.
        ...


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
                person.setup_electricity_and_hot_water_profile()
                person.setup_building_occupancy_profile(self.scenario)
                self.persons.append(person)

    def setup_building_occupancy_profile(self):
        self.building_occupancy = np.zeros(self.scenario.period_num)
        for hour in range(0, self.scenario.period_num):
            for person in self.persons:
                if person.building_occupancy_profile[hour] == 1:
                    self.building_occupancy[hour] = 1
                    break

    def setup_household_base_electricity_demand_profile(self):
        ...

    def setup_electricity_demand_profile(self):
        self.setup_household_base_electricity_demand_profile()
        self.electricity_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.electricity_demand += person.electricity_profile

    def setup_hot_water_demand_profile(self):
        self.hot_water_demand = np.zeros(self.scenario.period_num)
        for person in self.persons:
            self.hot_water_demand += person.hot_water_profile





