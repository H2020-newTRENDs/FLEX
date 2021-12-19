
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class BuildingDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Building:

    def setup(self, strategy: 'Type[SetupStrategy]' = BuildingDatabaseStrategy):
        pass
