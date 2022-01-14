
from abc import ABC, abstractmethod
from typing import Type, Dict, Any

from _Refactor.basic.reg import Table
from _Refactor.core.elements.component import Component
from _Refactor.core.elements.component_setup import GeneralComponent


class AbstractScenario(Component):

    def __init__(self, scenario_id: int):
        # self.scenario_id = scenario_id
        # self.setup()
        self.add_component_ids(scenario_id)

    def setup(self):
        # self.add_component_ids()
        # self.set_component_ids()
        # self.add_component_with_params()
        pass

    def add_component_ids(self, scenario_id):
        """adds all the component ids in the scenario table with their respective index"""
        # get all scenario ids from the scenarios table:
        all_scenario_ids = GeneralComponent(
            table_name=Table().scenarios,
            row_id=scenario_id
        ).get_complete_data()
        # set the components with their ids to self variables
        for component_name, component_index in all_scenario_ids.items():
            name = component_name.lower().replace("id_", "")
            setattr(self, name, component_index)


    def add_component(self, component_name: str, component_params_dict: Dict[str, Any]):
        self.component_params_dict[component_name] = component_params_dict

    # @abstractmethod
    def add_component_with_params(self):
        pass





if __name__=="__main__":
    AbstractScenario(scenario_id=0).add_component_ids()


