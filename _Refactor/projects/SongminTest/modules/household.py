
from _Refactor.core.household.abstract_household import AbstractHousehold
from _Refactor.core.household.components import Region, PersonList, Building, ApplianceList, Boiler, Tank,\
                                                AirConditioner, PV, Battery, Vehicle, Behavior, Demand

class SongminTestHousehold(AbstractHousehold):

    def add_components(self):
        self.tank = None

    def add_component_classes(self):
        self.tank_class = Tank

