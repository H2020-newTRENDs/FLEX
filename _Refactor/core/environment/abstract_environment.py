
from typing import Type, Optional, ClassVar, TYPE_CHECKING
from abc import ABC, abstractmethod

from _Refactor.core.elements.pillar import Pillar
from _Refactor.core.scenario.abstract_scenario import AbstractScenario
from _Refactor.core.environment.components import ElectricityPrice, FeedinTariff

class AbstractEnvironment(Pillar):

    def add_components(self):
        self.electricity_price = None
        self.feedin_tariff = None

    def add_component_classes(self):
        self.electricity_price_class: ClassVar['ElectricityPrice'] = ElectricityPrice
        self.feedin_tariff_class: ClassVar['FeedinTariff'] = FeedinTariff





