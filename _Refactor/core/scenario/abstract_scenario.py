
from abc import ABC, abstractmethod
from typing import Type, Dict, Any

from _Refactor.basic.reg import Table
from _Refactor.core.elements.component import Component
from _Refactor.core.elements.component_setup import GeneralComponent

class AbstractScenario(Component):

    def __init__(self, scenario_id: int):
        self.scenario_id = scenario_id
        self.setup()

    def setup(self):
        self.add_component_ids()
        self.set_component_ids()
        self.component_params_dict = {}
        self.add_component_with_params()

    def add_component_ids(self):
        self.region_id: int = None
        self.person_list_id: int = None
        self.building_id: int = None
        self.appliance_list_id: int = None
        self.boiler_id: int = None
        self.space_heating_tank_id: int = None
        self.air_conditioner_id: int = None
        self.hot_water_tank_id: int = None
        self.pv_id: int = None
        self.battery_id: int = None
        self.vehicle_id: int = None
        self.behavior_id: int = None
        self.demand_id: int = None
        self.electricity_price_id: int = None
        self.feedin_tariff_id: int = None

    def set_component_ids(self):
        self.set_params(GeneralComponent(Table().scenarios, self.scenario_id).get_params_dict())

    def add_component(self, component_name: str, component_params_dict: Dict[str, Any]):
        self.component_params_dict[component_name] = component_params_dict

    @abstractmethod
    def add_component_with_params(self):
        pass








