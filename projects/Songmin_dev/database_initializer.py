from config import config
from flex.flex_operation.component import Region, Building, Boiler, SpaceHeatingTank, HotWaterTank, \
    SpaceCoolingTechnology, PV, Battery, Vehicle, SEMS, EnergyPrice, Behavior
from flex.flex_operation.database_initializer import OperationDatabaseInitializer


class ProjectOperationDatabaseInitializer(OperationDatabaseInitializer):

    def load_source_table(self):
        self.load_table("behavior")
        self.load_table("region_pv_gis")

    def setup_component_scenario(self):
        self.add_component(
            component_cls=Region,
            component_scenarios={
                'code': ["DE"],  # only countries are implemented
                'year': [2019]  # weather and pv_generation data are downloaded from pv_gis only for 2019
            }
        )

        self.add_component(
            component_cls=Building,
            component_scenarios={
                'Af': [163.28],
                'Hop': [159.101],
                'Htr_w': [71.254],
                'Hve': [46.189],
                'CM_factor': [213505.516],
                'Am_factor': [3],
                'internal_gains': [3.4],
                'effective_window_area_west_east': [8.819],
                'effective_window_area_south': [4.724],
                'effective_window_area_north': [2.205],
                'grid_power_max':[64_000]
            })

        self.add_component(
            component_cls=Boiler,
            component_scenarios={
                'type': ["heat_pump"],
                'thermal_power_max': [15_000],
                'thermal_power_max_unit': ['W'],
                'heating_element_power': [7_500],
                'heating_element_power_unit': ['W'],
                'carnot_efficiency_factor': [0.35],
                'heating_supply_temperature': [35],
                'hot_water_supply_temperature': [55],
                'db_name': ['Air_HP']           #Air_HP pr Ground_HP
            })

        self.add_component(
            component_cls=SpaceHeatingTank,
            component_scenarios={
                'size': [750],
                'size_unit': ['L'],
                'surface_area': [4.57],
                'surface_area_unit': ['m2'],
                'loss': [0.2],
                'loss_unit': ['W/m2K'],
                'temperature_start': [28],
                'temperature_max': [45],
                'temperature_min': [28],
                'temperature_surrounding': [20],
                'temperature_unit': ['°C']
            })

        self.add_component(
            component_cls=HotWaterTank,
            component_scenarios={
                'size': [400],
                'size_unit': ['L'],
                'surface_area': [3.01],
                'surface_area_unit': ['m2'],
                'loss': [0.2],
                'loss_unit': ['W/m2K'],
                'temperature_start': [28],
                'temperature_max': [55],
                'temperature_min': [28],
                'temperature_surrounding': [20],
                'temperature_unit': ['°C']
            })

        self.add_component(
            component_cls=SpaceCoolingTechnology,
            component_scenarios={
                'efficiency': [3],
                'power': [10_000],
                'power_unit': ['W']
            })

        self.add_component(
            component_cls=PV,
            component_scenarios={
                'size': [5],
                'size_unit': ['kW_peak'],
            })

        self.add_component(
            component_cls=Battery,
            component_scenarios={
                'capacity': [7_000],
                'capacity_unit': ['Wh'],
                'charge_efficiency': [0.95],
                'charge_power_max': [4_500],
                'charge_power_max_unit': ['W'],
                'discharge_efficiency': [0.95],
                'discharge_power_max': [4_500],
                'discharge_power_max_unit': ['W'],
            })

        self.add_component(
            component_cls=Vehicle,
            component_scenarios={
                'type': ['electricity'],
                'capacity': [50_000],
                'capacity_unit': ['Wh'],
                'consumption_rate': [100],
                'consumption_rate_unit': ['Wh/km'],
                'charge_efficiency': [0.95],
                'charge_power_max': [11_000],
                'charge_power_max_unit': ['W'],
                'discharge_efficiency': [0.95],
                'discharge_power_max': [11_000],
                'discharge_power_max_unit': ['W'],
            })

        self.add_component(
            component_cls=SEMS,
            component_scenarios={
                'adoption': [1, 0],
            })

        self.add_component(
            component_cls=EnergyPrice,
            component_scenarios={
                'electricity_consumption': [0.032],
                'electricity_feed_in': [0.007],
                'district_heating': [0.02],
                'natural_gas': [0.02],
                'heating_oil': [0.02],
                'biomass': [0.02],
                'gasoline': [0.02],
                'price_unit': ['cent/Wh']
            })

        self.add_component(
            component_cls=Behavior,
            component_scenarios={
                'target_temperature_at_home_max': [45],
                'target_temperature_at_home_min': [10],
                'target_temperature_not_at_home_max': [45],
                'target_temperature_not_at_home_min': [10],
                'target_temperature_unit': ['°C'],
                'vehicle_distance_annual': [27],
                'vehicle_distance_unit': ['km'],
                'hot_water_demand_annual': [1000_000],
                'hot_water_demand_unit': ['Wh'],
                'appliance_electricity_demand_annual': [1000_000],
                'appliance_electricity_demand_unit': ['Wh'],
            })


if __name__ == "__main__":
    op_init = ProjectOperationDatabaseInitializer(config)
    op_init.run()
