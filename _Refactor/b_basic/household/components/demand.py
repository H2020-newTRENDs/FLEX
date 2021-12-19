
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy
from ..household import Household

"""
setup interface
"""
class DemandDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Demand:

    def setup(self, household: Household, strategy: 'Type[SetupStrategy]' = DemandDatabaseStrategy):
        self.household = household
        self.set_indoor_temperature_profile()
        self.set_driving_load_profile()
        self.set_appliance_load_profile()
        self.set_hot_water_load_profile()

    def set_indoor_temperature_profile(self):
        # self.indoor_temperature_min =
        # self.indoor_temperature_max =
        pass

    def set_driving_load_profile(self):
        # self.driving_load_profile =
        pass

    def set_appliance_load_profile(self):
        # self.appliance_load_profile =
        pass

    def set_hot_water_load_profile(self):
        # self.hot_water_load_profile =
        pass

