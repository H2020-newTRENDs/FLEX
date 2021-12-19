
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy
from ..household import Household

"""
setup interface
"""
class BehaviorDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Behavior:

    def setup(self, household: Household, strategy: 'Type[SetupStrategy]' = BehaviorDatabaseStrategy):
        self.household = household
        self.set_indoor_hours()
        self.set_driving_behavior()
        self.set_appliance_behavior()
        self.set_hot_water_behavior()

    def set_indoor_hours(self):
        # self.indoor_hours =
        pass

    def set_driving_behavior(self):
        # self.driving_behavior =
        pass

    def set_appliance_behavior(self):
        # self.appliance_behavior =
        pass

    def set_hot_water_behavior(self):
        # self.hot_water_behavior =
        pass

