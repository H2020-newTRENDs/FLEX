
from typing import Type, Optional, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from _Refactor.core.elements.pillar import Pillar
from _Refactor.core.scenario.abstract_scenario import AbstractScenario
import _Refactor.core.household.components as components

class AbstractHousehold:

    def __init__(self, scenario: 'AbstractScenario'):
        self.scenario = scenario
        self.add_component_classes()

    def add_component_classes(self):
        # iterate through the scenario dict which hols all IDs for each component:
        for component, component_id in self.scenario.__dict__.items():
            for key in components.__dict__.keys():
                if component == key.lower():  # check if they class exists and use its name
                    component_name = key
            # get the class
            print(component_name)
            class_filled = getattr(components, component_name)(component_id=component_id)
            # create self variable of the class with the filled class
            setattr(self, component + "_class", class_filled)

            pass

    # def add_components(self):
    #     self.region = None
    #     self.person_list = None
    #     self.building = None
    #     self.appliance_list = None
    #     self.boiler = None
    #     self.space_heating_tank = None
    #     self.air_conditioner = None
    #     self.hot_water_tank = None
    #     self.pv = None
    #     self.battery = None
    #     self.vehicle = None
    #     self.behavior = None
    #     self.hot_water_demand = None
    #     self.electricity_demand = None



if __name__ == "__main__":
    scenario = AbstractScenario(scenario_id=0)
    AbstractHousehold(scenario).add_component_classes()

