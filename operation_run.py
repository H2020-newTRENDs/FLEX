from basics.kit import get_logger
from config import config
from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import FuelBoilerOptInstance
from models.operation.model_opt import FuelBoilerOptOperationModel
from models.operation.model_opt import OptInstance
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario

logger = get_logger(__name__)

if __name__ == "__main__":
    hp_instance = OptInstance().create_instance()
    # fb_instance = FuelBoilerOptInstance().create_instance()
    scenario_ids = [1, 2]
    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")
        scenario = OperationScenario(scenario_id=scenario_id, config=config)
        ref_model = RefOperationModel(scenario).solve()

        opt_model = OptOperationModel(scenario).solve(hp_instance)

        RefDataCollector(ref_model, scenario.scenario_id, config).run()
        OptDataCollector(opt_model, scenario.scenario_id, config).run()

