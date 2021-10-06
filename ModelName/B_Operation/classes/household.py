import pandas

class Household:

    def __init__(self, params: pandas.Series):

        self.id_household = params["id_household"]
        self.id_country = params["id_country"]
        self.appliance_load = self.setup_appliance_load()
        self.hot_water_demand = self.setup_hot_water_demand()
        self.target_temperature_high = self.setup_target_temperature_high()
        self.target_temperature_low = self.setup_target_temperature_low()

    def setup_appliance_load(self):
        # read from database. The result is produced by A_HouseholdBehavior.
        pass

    def setup_hot_water_demand(self):
        # read from database. The result is produced by A_HouseholdBehavior.
        pass

    def setup_target_temperature_high(self):
        # read from database. The result is produced by A_HouseholdBehavior.
        pass

    def setup_target_temperature_low(self):
        # read from database. The result is produced by A_HouseholdBehavior.
        pass
