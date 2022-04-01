from flex.core.config import Config
from flex.models.test_operation.component import Region, Year, Building, Boiler, SpaceHeatingTank, HotWaterTank, \
    SpaceCoolingTechnology, PV, Battery, Vehicle, SEMS
from flex.models.test_operation.database_initializer import OperationDatabaseInitializer

project_config = Config(project_name="SongminDev")


class ProjectOperationDatabaseInitializer(OperationDatabaseInitializer):

    def setup_component_scenarios(self):
        self.add_component(
            component_cls=Region,
            component_scenarios={
                'nuts': ["DE"]
            }
        )

        self.add_component(
            component_cls=Year,
            component_scenarios={
                'year': [2019]
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
                'effective_window_area_north': [2.205]
            })

        self.add_component(
            component_cls=Boiler,
            component_scenarios={
                'type': ["gas", "heat_pump"],
                'thermal_power_max': [15_000, 15_000],
                'thermal_power_max_unit': ['W', 'W'],
                'heating_element_power': [7_500, 7_500],
                'heating_element_power_unit': ['W', 'W'],
                'carnot_efficiency_factor': [0.35, 0.35],
                'heating_supply_temperature': [35, 35],
                'hot_water_supply_temperature': [55, 55]
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
                'power': [10000],
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
                'charge_power_max': [4500],
                'charge_power_max_unit': ['W'],
                'discharge_efficiency': [0.95],
                'discharge_power_max': [4500],
                'discharge_power_max_unit': ['W'],
            })

        self.add_component(
            component_cls=Vehicle,
            component_scenarios={
                'type': ['electricity'],
                'capacity': [15_000],
                'capacity_unit': ['Wh'],
                'consumption_rate': [100],
                'consumption_rate_unit': ['Wh/km'],
                'charge_efficiency': [0.95],
                'charge_power_max': [4500],
                'charge_power_max_unit': ['W'],
                'discharge_efficiency': [0.95],
                'discharge_power_max': [4500],
                'discharge_power_max_unit': ['W'],
            })

        self.add_component(
            component_cls=SEMS,
            component_scenarios={
                'adoption': [1, 0],
            })


if __name__ == "__main__":
    op_init = ProjectOperationDatabaseInitializer(project_config)
    op_init.run()
