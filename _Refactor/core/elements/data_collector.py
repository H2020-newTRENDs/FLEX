
from abc import ABC, abstractmethod
from typing import Dict, Any

class DataCollector(ABC):

    def __init__(self, object):
        self.object = object
        self.variables_to_collect = {}
        self.setup()

    def add_var(self, var_name, var_data_type):
        self.variables_to_collect[var_name] = var_data_type

    def collect(self):
        for key, value in self.variables_to_collect:
            if key in self.object.__dict__.keys():
                if result_value != None:
                    # save object.key

    def setup(self):
        pass


