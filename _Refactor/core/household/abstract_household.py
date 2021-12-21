
from typing import Type, Optional, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from _Refactor.core.elements.pillar import Pillar
from _Refactor.core.scenario.abstract_scenario import AbstractScenario
from _Refactor.core.household.components import Region, PersonList, Building, ApplianceList, Boiler, Tank,\
                                                AirConditioner, PV, Battery, Vehicle, Behavior, Demand


class AbstractHousehold(Pillar):

    def add_components(self):
        self.region = None
        self.person_list = None
        self.building = None
        self.appliance_list = None
        self.boiler = None
        self.tank = None
        self.air_conditioner = None
        self.pv = None
        self.battery = None
        self.vehicle = None
        self.behavior = None
        self.demand = None

    def add_component_classes(self):
        self.region_class: ClassVar['Region'] = Region
        self.person_list_class: ClassVar['PersonList'] = PersonList
        self.building_class: ClassVar['Building'] = Building
        self.appliance_list_class: ClassVar['ApplianceList'] = ApplianceList
        self.boiler_class: ClassVar['Boiler'] = Boiler
        self.tank_class: ClassVar['Tank'] = Tank
        self.air_conditioner_class: ClassVar['AirConditioner'] = AirConditioner
        self.pv_class: ClassVar['PV'] = PV
        self.battery_class: ClassVar['Battery'] = Battery
        self.vehicle_class: ClassVar['Vehicle'] = Vehicle
        self.behavior_class: ClassVar['Behavior'] = Behavior
        self.demand_class: ClassVar['Demand'] = Demand




