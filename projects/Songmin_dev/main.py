from flex.flex_operation.scenario import OperationScenario
from config import config


scenario = OperationScenario(scenario_id=4, config=config)
print(scenario.behavior.vehicle_distance)
