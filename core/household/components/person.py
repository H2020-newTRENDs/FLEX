
from typing import Dict, Any

from core.elements.component import Component

class Person(Component):

    def __init__(self, params_dict: Dict[str, Any]):
        self.var_1 = None
        self.var_2 = None
        self.set_parameters(params_dict, )


class PersonList:

    def __init__(self):
        self.persons = []

    def setup(self):
        pass