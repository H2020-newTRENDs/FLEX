import pandas
from .person import Person

class Household:

    def __init__(self, params: pandas.Series):

        self.id_household_type = params["id_household_type"]
        self.members = self.setup_household_members()
        self.appliance_adoption = params["appliance_adoption"]
        self.appliance_power = params["appliance_power"]
        self.car_adoption = params["car_adoption"]
        self.target_temperature_high = None
        self.target_temperature_low = None
        self.appliance_electricity_demand = None
        self.hot_water_demand = None
        self.car_demand = None

    def setup_household_members(self) -> list:
        # append persons into self.members based on params
        # return members
        pass

    def calc_household_life_plan(self, time_structure, annual_holiday):

        # 1. for member in self.members: member.calc_person_at_home_profile(time_structure)
        # 2. the household make holiday decision together and update the at_home_profile of each member.
        # 3. for each member, according to their at_home_profile:
        #    member.calc_person_appliance_using_behavior()
        #    member.calc_person_hot_water_using_behavior()
        #    member.calc_person_car_using_behavior()
        # 4. calculate important results for the household:
        #    self.target_temperature_high
        #    self.target_temperature_low
        #    self.appliance_electricity_demand
        #    self.hot_water_demand
        #    self.car_demand --> 0 means at home, positive means running outside (unit: km)

        pass

    def save_household_life_plan(self):
        pass

