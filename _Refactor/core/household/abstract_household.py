
from typing import Type, Optional, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from _Refactor.core.elements.pillar import Pillar
import _Refactor.core.household.components as components
from _Refactor.core.elements.component_setup import GeneralComponent
from _Refactor.basic.reg import Table



class AbstractHousehold:

    def __init__(self, scenario_id: int):
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
        for component, component_id in self.scenario_ids.items():
            for key in components.__dict__.keys():
                if component == key.lower():  # check if they class exists and use its name
                    component_name = key
            # get the class
            print(f"{component_name} exists")
            class_filled = getattr(components, component_name)(component_id=component_id)
            # create self variable of the class with the filled class
            setattr(self, component + "_class", class_filled)

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
    AbstractHousehold(scenario_id=0).add_component_classes()

