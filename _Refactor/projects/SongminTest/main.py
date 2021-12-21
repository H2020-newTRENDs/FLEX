
import os

from config import project_config
from modules.model import SongminTestOptOperationModel
from modules.environment import SongminTestEnvironment
from modules.household import SongminTestHousehold
from modules.scenario import SongminTestScenario

scenario = SongminTestScenario(scenario_id=0)
household = SongminTestHousehold(scenario)
environment = SongminTestEnvironment(scenario)
model = SongminTestOptOperationModel(household, environment)

print(model.household.tank.loss)
