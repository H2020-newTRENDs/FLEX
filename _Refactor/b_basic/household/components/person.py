
from typing import Type

from _Refactor.b_basic.interface.setup import SetupStrategy

"""
setup interface
"""
class PersonListDatabaseStrategy(SetupStrategy):

    def setup(self):
        pass


"""
components abstract class
"""
class Person:
    pass

class PersonList:

    def __init__(self):
        self.persons = []

    def setup(self, strategy: 'Type[SetupStrategy]' = PersonListDatabaseStrategy):
        pass