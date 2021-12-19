
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class ApplianceListDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Appliance:
    pass

class ApplianceList:

    def __init__(self):
        self.items = []

    def setup(self, strategy: 'Type[SetupStrategy]' = ApplianceListDatabaseStrategy):
        pass
