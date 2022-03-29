from flex.models.test_operation.scenario import OperationScenario
from config import project_config


scenario = OperationScenario(scenario_id=4, config=project_config)
print(scenario.sems.adoption)
