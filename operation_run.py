from models.operation.scenario import OperationScenario
from models.operation.model_opt import OptModelFramework, SolveHeatPumpOptimization
from models.operation.model_ref import RefOperationModel
from models.operation.data_collector import OptDataCollector, RefDataCollector
from config import config
from basics.kit import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    # TODO call Optoperation and create the base instance
    base_instance = OptModelFramework().create_base_instance()
    scenario_ids = [1, 2, 3, 4, 5]
    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")
        scenario = OperationScenario(scenario_id=scenario_id, config=config)
        ref_model = RefOperationModel(scenario).run()
        opt_model = SolveHeatPumpOptimization(scenario).solve_model(base_instance)
        RefDataCollector(ref_model, scenario.scenario_id, config).run()
        OptDataCollector(opt_model, scenario.scenario_id, config).run()
