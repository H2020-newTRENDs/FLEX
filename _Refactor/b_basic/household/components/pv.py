
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class PVDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class PV:

    def setup(self, strategy: 'Type[SetupStrategy]' = PVDatabaseStrategy):
        pass
