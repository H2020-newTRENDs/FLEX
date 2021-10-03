import pandas

class Person:

    def __init__(self, params: pandas.Series):

        self.id_person_type = params["id_person_type"]
        self.firm_work_day_per_week = params["firm_work_day_per_week"]
        self.target_temperature_daytime_low = params["target_temperature_daytime_low"]
        self.target_temperature_daytime_high = params["target_temperature_daytime_high"]
        self.target_temperature_night_low = params["target_temperature_night_low"]
        self.target_temperature_night_high = params["target_temperature_night_high"]
        self.target_temperature_empty_low = params["target_temperature_empty_low"]
        self.target_temperature_empty_high = params["target_temperature_empty_high"]
        self.at_home_profile = None

    def calc_person_at_home_profile(self):
        # according to the (1) work-from-home assumption and (2) time structure
        # generate hourly self.at_home_profile
        pass

    def calc_person_appliance_using_behavior(self):
        pass

    def calc_person_hot_water_using_behavior(self):
        pass

    def calc_person_car_using_behavior(self):
        # assume this person has his own car
        # in household.calc_household_life_plan(), we know if there is a car that he can use
        pass




