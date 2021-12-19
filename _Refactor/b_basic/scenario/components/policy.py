
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy
from _Refactor.b_basic.household.household import Household

"""
setup interface
"""
class PolicyDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Policy:

    def setup(self, household: Household, strategy: 'Type[SetupStrategy]' = PolicyDatabaseStrategy):
        self.household = household
        self.set_electricity_price()
        self.set_feedin_tariff()

    def set_electricity_price(self):
        # self.electricity_price =
        pass

    def set_feedin_tariff(self):
        # self.feedin_tariff =
        pass