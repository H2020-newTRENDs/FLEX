
import os

from config import project_config
from modules.model_opt import PhilippTestOptOperationModel
from modules.environment import PhilippTestEnvironment
from modules.household import PhilippTestHousehold
from modules.scenario import PhilippTestScenario


scenario = PhilippTestScenario(scenario_id=0)
household = PhilippTestHousehold(scenario)
environment = PhilippTestEnvironment(scenario)
model = PhilippTestOptOperationModel(household, environment)

print(model.household.space_heating_tank.loss)
print(model.household.hot_water_tank.loss)

