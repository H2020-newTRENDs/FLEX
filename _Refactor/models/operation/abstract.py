
from typing import Type
from abc import ABC, abstractmethod

from _Refactor.core.environment.abstract_environment import AbstractEnvironment
from _Refactor.core.household.abstract_household import AbstractHousehold

"""
abstract operation model
"""
class AbstractOperationModel(ABC):

    def __init__(self,
                 household: 'AbstractHousehold',
                 environment: 'AbstractEnvironment'):

        self.household = household
        self.environment = environment

    # @abstractmethod
    # def run(self):
    #     pass

