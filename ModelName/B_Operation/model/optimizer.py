
from ..classes.environment import Environment
from ..classes.household import Household
from ..classes.building import Building

class Optimizer:

    def __init__(self, environment: Environment, household: Household, building: Building):

        self.environment = environment
        self.household = household
        self.building = building

        pass

    def setup_building_COP(self):
        self.building.setup_heating_COP(self.environment.temperature)
        self.building.setup_cooling_COP(self.environment.temperature)
        self.building.setup_hot_water_COP(self.environment.temperature)
        self.building.setup_pv_generation(self.environment.radiation)

    def run_optimization(self):
        # run optimization for only heating/cooling, pv, battery, ev
        # then post-adjustment: add appliance and hot water demand back to the optimization result
        # return the post-adjusted result
        pass
