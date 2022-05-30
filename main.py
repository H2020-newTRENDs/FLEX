from models.operation.scenario import OperationScenario
from models.operation.opt import OptOperationModel
from models.operation.ref import RefOperationModel
from models.operation.data_collector import OptDataCollector, RefDataCollector
from config import config
from basics.kit import get_logger

logger = get_logger(__name__)


for scenario_id in range(1, 7):
    logger.info(f'Scenario = {scenario_id}')
    scenario = OperationScenario(scenario_id=scenario_id, config=config)

    opt_model = OptOperationModel(scenario).run()
    OptDataCollector(opt_model, scenario.scenario_id, config).run()

    ref_model = RefOperationModel(scenario).run()
    RefDataCollector(ref_model, scenario.scenario_id, config).run()
