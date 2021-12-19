
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class AirConditionerDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class AirConditioner:

    def setup(self, strategy: 'Type[SetupStrategy]' = AirConditionerDatabaseStrategy):
        pass
