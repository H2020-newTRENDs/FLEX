
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class BoilerDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Boiler:

    def setup(self, strategy: 'Type[SetupStrategy]' = BoilerDatabaseStrategy):
        pass
