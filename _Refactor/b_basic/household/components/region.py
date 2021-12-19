
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class RegionDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass

"""
components abstract class
"""
class Region:

    def setup(self, strategy: 'Type[SetupStrategy]' = RegionDatabaseStrategy):
        self.setup_region_id()
        self.setup_temperature()
        self.setup_radiation()

    def setup_region_id(self):
        # self.id =
        pass

    def setup_temperature(self):
        # self.temperature =
        pass

    def setup_radiation(self):
        # self.radiation =
        pass