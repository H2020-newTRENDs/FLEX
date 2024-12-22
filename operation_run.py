import numpy as np

from basics.kit import get_logger
from config import config
from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import OptInstance
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario, MotherOperationScenario

logger = get_logger(__name__)


def run_ref_operation(scenario: "OperationScenario"):
    ref_model = RefOperationModel(scenario).solve()
    RefDataCollector(ref_model, scenario.scenario_id, config, save_hour_results=True).run()


def run_opt_scenario(scenario: "OperationScenario", opt_instance):
    opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
    if solve_status:
        OptDataCollector(opt_model, scenario.scenario_id, config, save_hour_results=True).run()


def run_scenarios(scenario_ids):
    opt_instance = OptInstance().create_instance()
    mother_operation = MotherOperationScenario(config=config)
    config.input_operation

    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")

        scenario = OperationScenario(scenario_id=scenario_id, config=config, tables=mother_operation)
        run_ref_operation(scenario)
        run_opt_scenario(scenario, opt_instance)


if __name__ == "__main__":

    SCENARIO_IDS = np.arange(1, 37)
    run_scenarios(SCENARIO_IDS)



