from typing import Dict, Any

from _Refactor.basic.reg import TABLE
from _Refactor.core.elements.component_setup import GeneralComponent as GC
from _Refactor.core.scenario.abstract_scenario import AbstractScenario

class SongminTestScenario(AbstractScenario):

    def add_component_with_params(self):
        table = TABLE()
        # self.add_component_params("region", GC(table.region, self.region_id).get_params_dict())
        # self.add_component_params("person_list", GC(table.person_list, self.person_list_id).get_params_dict())
        # self.add_component_params("building", GC(table.building, self.building_id).get_params_dict())
        # self.add_component_params("appliance_list", GC(table.appliance_list, self.appliance_list_id).get_params_dict())
        # self.add_component_params("boiler", GC(table.boiler, self.boiler_id).get_params_dict())
        self.add_component("tank", GC(table.tank, self.tank_id).get_params_dict())
        # self.add_component_params("air_conditioner", GC(table.air_conditioner, self.air_conditioner_id).get_params_dict())
        # self.add_component_params("pv", GC(table.pv, self.pv_id).get_params_dict())
        # self.add_component_params("battery", GC(table.battery, self.battery_id).get_params_dict())
        # self.add_component_params("vehicle", GC(table.vehicle, self.vehicle_id).get_params_dict())
        # self.add_component_params("behavior", GC(table.behavior, self.behavior_id).get_params_dict())
        # self.add_component_params("demand", GC(table.demand, self.demand_id).get_params_dict())
        # self.add_component_params("electricity_price", GC(table.electricity_price, self.electricity_price_id).get_params_dict())
        # self.add_component_params("feedin_tariff", GC(table.feedin_tariff, self.feedin_tariff_id).get_params_dict())

