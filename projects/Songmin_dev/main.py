from flex.core.config import Config
from models.test_operation.scenario import OperationScenario
from config import project_config


scenario = OperationScenario(scenario_id=1, config=project_config)
print(scenario.boiler.thermal_power_max)
