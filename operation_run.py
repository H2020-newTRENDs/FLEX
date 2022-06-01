from models.operation.scenario import OperationScenario
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.data_collector import OptDataCollector, RefDataCollector
from config import config
from basics.kit import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    scenario_ids = list(range(1, 433))
    for scenario_id in scenario_ids:
        logger.info(f'Scenario = {scenario_id}')
        scenario = OperationScenario(scenario_id=scenario_id, config=config)
        ref_model = RefOperationModel(scenario).run()
        opt_model = OptOperationModel(scenario).run()
        RefDataCollector(ref_model, scenario.scenario_id, config).run()
        OptDataCollector(opt_model, scenario.scenario_id, config).run()


