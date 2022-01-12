
from _Refactor.core.household.abstract_household import AbstractHousehold
from _Refactor.core.household.components import Region, PersonList, Building, ApplianceList, Boiler, SpaceHeatingTank,\
                                                AirConditioner, HotWaterTank, PV, Battery, Vehicle, Behavior, HotWaterDemand

class SongminTestHousehold(AbstractHousehold):

    def add_components(self):
        self.space_heating_tank = None
        self.hot_water_tank = None

    def add_component_classes(self):
        self.space_heating_tank_class = SpaceHeatingTank
        self.hot_water_tank_class = HotWaterTank

