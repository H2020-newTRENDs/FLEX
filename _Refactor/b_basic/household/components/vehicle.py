
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class VehicleDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Vehicle:

    def setup(self, strategy: 'Type[SetupStrategy]' = VehicleDatabaseStrategy):
        pass
