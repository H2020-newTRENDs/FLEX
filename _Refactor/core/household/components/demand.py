from _Refactor.core.elements.component import Component


class HotWaterDemand(Component):
    def __init__(self):
        self.ID_HotWaterDemand = None
        self.name = None
        self.hot_water_demand = None


class ElectricityDemand(Component):
    def __init__(self):
        self.ID_Electricity_Demand = None
        self.name = None
        self.electricity_demand = None
