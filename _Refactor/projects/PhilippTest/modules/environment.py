
from _Refactor.core.environment.abstract_environment import AbstractEnvironment
from _Refactor.core.environment.components import ElectricityPrice, FeedinTariff

class PhilippTestEnvironment(AbstractEnvironment):

    def add_components(self):
        self.electricity_price = None
        self.feedin_tariff = None

    def add_component_classes(self):
        self.electricity_price_class = ElectricityPrice
        self.feedin_tariff_class = FeedinTariff



