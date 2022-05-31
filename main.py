from models.operation.scenario import OperationScenario
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.data_collector import OptDataCollector, RefDataCollector
from config import config
from basics.kit import get_logger

logger = get_logger(__name__)

scenario_ids = list(range(1, 193))
for scenario_id in scenario_ids:
    logger.info(f'Scenario = {scenario_id}')
    scenario = OperationScenario(scenario_id=scenario_id, config=config)
    ref_model = RefOperationModel(scenario).run()
    RefDataCollector(ref_model, scenario.scenario_id, config).run()
    opt_model = OptOperationModel(scenario).run()
    OptDataCollector(opt_model, scenario.scenario_id, config).run()


