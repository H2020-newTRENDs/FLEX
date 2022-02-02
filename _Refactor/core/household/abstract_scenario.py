
from typing import Type, Optional, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod

import _Refactor.core.household.components as components
from _Refactor.core.elements.component_setup import GeneralComponent
from _Refactor.basic.reg import Table


class AbstractScenario:

    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        self.region_class = None
        self.person_class = None
        self.personlist_class = None
        self.building_class = None
        self.appliance_class = None
        self.appliancelist_class = None
        self.boiler_class = None
        self.spaceheatingtank_class = None
        self.airconditioner_class = None
        self.hotwatertank_class = None
        self.pv_class = None
        self.battery_class = None
        self.vehicle_class = None
        self.behavior_class = None
        self.hotwaterdemand_class = None
        self.electricitydemand_class = None
        self.electricityprice_class = None
        self.feedintariff_class = None

        # get all scenario ids from the scenarios table:
        self.scenario_ids = GeneralComponent(
            table_name=Table().scenarios,
            row_id=scenario_id
        ).get_complete_data()  # is a dictionary  with the ids
        self.add_component_classes()

    def add_component_classes(self):
        # iterate through the scenario dict which hols all IDs for each component:
        for component_name, component_id in self.scenario_ids.items():
            name = component_name.lower().replace("id_", "")
            for key in components.__dict__.keys():
                if name == key.lower():  # check if they class exists and use its name
                    class_name = key
                    # get the class
                    print(f"{class_name} exists")
            class_filled = getattr(components, class_name)(component_id=component_id)  # all classes take the component_id as parameter to initialize their values
            # create self variable of the class with the filled class
            setattr(self, name + "_class", class_filled)




if __name__ == "__main__":
    household_class = AbstractScenario(scenario_id=0)
    household_class.add_component_classes()

