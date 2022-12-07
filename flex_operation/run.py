from typing import List, Tuple

from flex.config import Config
from flex.db import create_db_conn
from flex.kit import get_logger
from flex_operation.analyzer import OperationAnalyzer
from flex_operation.constants import OperationTable
from flex_operation.data_collector import OptDataCollector
from flex_operation.data_collector import RefDataCollector
from flex_operation.model_opt import OptInstance
from flex_operation.model_opt import OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.scenario import OperationScenario

logger = get_logger(__name__)


def run_operation_model(operation_scenario_ids: List[int], config: "Config"):
    opt_instance = OptInstance().create_instance()

    for id_operation_scenario in operation_scenario_ids:
        logger.info(f"FlexOperation Model --> Scenario = {id_operation_scenario}.")
        scenario = OperationScenario(scenario_id=id_operation_scenario, config=config)
        # run ref model
        ref_model = RefOperationModel(scenario).solve()
        RefDataCollector(ref_model, scenario.scenario_id, config, save_hour_results=False).run()
        # run opt model

        opt_model, solve_status = OptOperationModel(scenario).solve(opt_instance)
        if solve_status:
            OptDataCollector(opt_model, scenario.scenario_id, config, save_hour_results=False).run()


def run_operation_analyzer(
    component_changes: List[Tuple[str, int, int]],
    components: List[Tuple[str, int]],
    config: "Config"
):
    ana = OperationAnalyzer(config)
    ana.create_operation_energy_cost_table()
    ana.plot_operation_energy_cost_curve()
    ana.create_component_energy_cost_change_tables(component_changes)
    ana.plot_operation_energy_cost_change_curve(component_changes)
    ana.plot_scenario_electricity_balance(scenario_id=1)
    # # component interaction
    component_change = ("ID_PV", 2, 1)
    other_components = list(filter(lambda item: item[0] != component_change[0], components))
    ana.plot_component_interaction_full(component_change, other_components)
    ana.plot_component_interaction_specific(component_change, ("ID_Battery", 2))
    ana.plot_component_interaction_heatmap(component_change, ("ID_Boiler", 2), ("ID_SpaceHeatingTank", 2), components)


def find_infeasible_scenarios(config: "Config"):
    db = create_db_conn(config)
    opt = db.read_dataframe(OperationTable.ResultOptYear)
    ref = db.read_dataframe(OperationTable.ResultRefYear)
    opt_scenarios = list(opt["ID_Scenario"].unique())
    ref_scenarios = list(ref["ID_Scenario"].unique())
    infeasible_scenarios = set(ref_scenarios) - set(opt_scenarios)
    print(f'Infeasible Scenarios: {infeasible_scenarios}.')
