
from abc import ABC, abstractmethod

from _Refactor.core.scenario.abstract_scenario import AbstractScenario

class Pillar(ABC):

    def __init__(self, scenario: 'AbstractScenario'):
        self.scenario = scenario
        self.setup()

    @abstractmethod
    def add_components(self):
        pass

    @abstractmethod
    def add_component_classes(self):
        pass

    def setup(self):
        self.add_components()
        self.add_component_classes()
        for component, component_params in self.scenario.component_params_dict.items():
            if component in self.__dict__.keys():
                setattr(self, component, self.__dict__[component + "_class"](component_params))

