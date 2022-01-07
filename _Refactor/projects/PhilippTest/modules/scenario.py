from typing import Dict, Any

from _Refactor.basic.reg import Table
from _Refactor.core.elements.component_setup import GeneralComponent as GC
from _Refactor.core.scenario.abstract_scenario import AbstractScenario

class PhilippTestScenario(AbstractScenario):

    def add_component_with_params(self):
        table = Table()
        # self.add_component_params("region", GC(table.region, self.region_id).get_params_dict())
        # self.add_component_params("person_list", GC(table.person_list, self.person_list_id).get_params_dict())
        self.add_component(
            "building", GC(
                table_name=table.building,
                row_id=self.building_id,
            ).get_params_dict()
        )
        # self.add_component_params("appliance_list", GC(table.appliance_list, self.appliance_list_id).get_params_dict())
        # self.add_component_params("boiler", GC(table.boiler, self.boiler_id).get_params_dict())
        self.add_component(
            "space_heating_tank", GC(
                table_name=table.space_heating_tank,
                row_id=self.space_heating_tank_id,
                db_name="ProsumagerUpdated_Philipp"
            ).get_params_dict()
        )
        self.add_component(
            "air_conditioner", GC(
                table.air_conditioner, self.air_conditioner_id, "ProsumagerUpdated_Philipp"
            ).get_params_dict()
        )
        self.add_component("hot_water_tank", GC(table.hot_water_tank, self.hot_water_tank_id, "ProsumagerUpdated_Philipp").get_params_dict())
        self.add_component(
            "pv", GC(
                table.pv, self.pv_id, "ProsumagerUpdated_Philipp"
            ).get_params_dict()
        )
        self.add_component(
            "battery", GC(
                table.battery, self.battery_id, "ProsumagerUpdated_Philipp"
            ).get_params_dict()
        )
        # self.add_component_params("vehicle", GC(table.vehicle, self.vehicle_id).get_params_dict())
        # self.add_component_params("behavior", GC(table.behavior, self.behavior_id).get_params_dict())
        # self.add_component_params("demand", GC(table.demand, self.demand_id).get_params_dict())
        self.add_component(
            "electricity_price", GC(
                table.electricity_price, self.electricity_price_id, "ProsumagerUpdated_Philipp"
            ).get_params_dict()
        )
        self.add_component(
            "feedin_tariff", GC(table.feedin_tariff, self.feedin_tariff_id, "ProsumagerUpdated_Philipp"
                                ).get_params_dict()
        )

