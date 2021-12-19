
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class TankDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Tank:

    def setup(self, strategy: 'Type[SetupStrategy]' = TankDatabaseStrategy):
        pass
