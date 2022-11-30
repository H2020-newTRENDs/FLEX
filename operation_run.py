import numpy as np

from basics.kit import get_logger
from config import config
from models.operation.data_collector import OptDataCollector
from models.operation.data_collector import RefDataCollector
from models.operation.model_opt import OptInstance
from models.operation.model_opt import OptOperationModel
from models.operation.model_ref import RefOperationModel
from models.operation.scenario import OperationScenario

logger = get_logger(__name__)


def run_ref_operation(scenario: "OperationScenario"):
    ref_model = RefOperationModel(scenario).solve()
    RefDataCollector(ref_model, scenario.scenario_id, config, save_hour_results=True).run()


def run_opt_scenario(scenario: "OperationScenario", opt_instance):
    try:
        opt_model = OptOperationModel(scenario).solve(opt_instance)
        OptDataCollector(opt_model, scenario.scenario_id, config, save_hour_results=True).run()
    except ValueError:
        print(f'Infeasible --> ID_Scenario = {scenario.scenario_id}')


def run_scenarios(scenario_ids):
    opt_instance = OptInstance().create_instance()
    for scenario_id in scenario_ids:
        logger.info(f"FlexOperation --> Scenario = {scenario_id}.")

        scenario = OperationScenario(scenario_id=scenario_id, config=config)
        run_ref_operation(scenario)
        run_opt_scenario(scenario, opt_instance)


if __name__ == "__main__":

    SCENARIO_IDS = np.arange(1, 73)
    run_scenarios(SCENARIO_IDS)



