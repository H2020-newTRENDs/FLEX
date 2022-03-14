
from core.elements.component import Component


class HotWaterDemand(Component):
    def __init__(self, component_id):
        self.ID_HotWaterDemand = None
        self.name = None
        self.hot_water_demand = None

        self.set_parameters(component_id)


class ElectricityDemand(Component):
    def __init__(self, component_id):
        self.ID_Electricity_Demand = None
        self.name = None
        self.electricity_demand = None

        self.set_parameters(component_id)
