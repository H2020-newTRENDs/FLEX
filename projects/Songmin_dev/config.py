from flex.core.config import Config
from flex.core.component import Battery, Boiler


class DevConfig(Config):

    def setup_component_scenarios(self):
        self.add_component(
            component_cls=Battery,
            component_scenarios={
                'capacity': [0, 7_000],
                'capacity_unit': ['Wh'],
                'charge_efficiency': [0.95],
                'discharge_efficiency': [0.95],
                'charge_power_max': [4500],
                'charge_power_max_unit': ['W'],
                'discharge_power_max': [4500],
                'discharge_power_max_unit': ['W'],
            })
        self.add_component(
            component_cls=Boiler,
            component_scenarios={
                'name': ["gas", "heat_pump"],
                'thermal_power_max': [15_000],
                'thermal_power_max_unit': ['W'],
                'heating_element_power': [7_500],
                'heating_element_power_unit': ['W'],
                'carnot_efficiency_factor': [0.35],
                'heating_supply_temperature': [35],
                'hot_water_supply_temperature': [55]
            })

config = DevConfig("SongminDev")