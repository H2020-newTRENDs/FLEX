
from _Refactor.core.household.abstract_household import AbstractHousehold
from _Refactor.core.household.components import Region, PersonList, Building, ApplianceList, Boiler, SpaceHeatingTank,\
                                                AirConditioner, HotWaterTank, PV, Battery, Vehicle, Behavior, HotWaterDemand

class PhilippTestHousehold(AbstractHousehold):

    def add_components(self):
        self.space_heating_tank = None
        self.hot_water_tank = None
        self.pv = None
        self.battery = None

    def add_component_classes(self):
        self.boiler_class = Boiler
        self.air_conditioner_class = AirConditioner
        self.pv_class = PV
        self.battery_class = Battery
        self.demand_class = HotWaterDemand
        self.building_class = Building
        self.space_heating_tank_class = SpaceHeatingTank
        self.hot_water_tank_class = HotWaterTank

