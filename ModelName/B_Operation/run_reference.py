
from ModelName.B_Operation.classes.environment import Environment
from ModelName.B_Operation.classes.household import Household
from ModelName.B_Operation.classes.building import Building
from ModelName.B_Operation.model.operation_optimization import OperationOptimization

class RunReference:

    def __init__(self, id_household, id_building_class, id_building_technology, id_environment):

        # household_params = --> read from database, the result table from A_HouseholdBehavior
        # self.household = Household(household_params)

        # read one row from BuildingClass table (id_building_class, type, age class, efficiency status)
        # then read one row from BuildingData table, by country, type, age class, and efficiency status --> building_params
        # read one row from BuildingTechnology table --> building_technology_params
        # self.building = Building(building_params, building_technology_params)

        # read one row from Environment table --> environment_params
        # self.environment = Environment(self.household.id_country, environment_params)

        pass

    def run(self):

        # result = OperationOptimization(self.household, self.building, self.environment).run_optimization()
        # save result

        pass