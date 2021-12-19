
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class BatteryDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Battery:

    def setup(self, strategy: 'Type[SetupStrategy]' = BatteryDatabaseStrategy):
        pass
