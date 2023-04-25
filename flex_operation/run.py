from typing import List, Tuple

from flex.config import Config
from flex.kit import get_logger
from flex_operation.analyzer import OperationAnalyzer
from flex_operation.data_collector import OptDataCollector
from flex_operation.data_collector import RefDataCollector
from flex_operation.model_opt import OptInstance
from flex_operation.model_opt import OptOperationModel
from flex_operation.model_ref import RefOperationModel
from flex_operation.scenario import OperationScenario

logger = get_logger(__name__)


def run_operation_model(operation_scenario_ids: List[int], config: "Config"):
    for id_operation_scenario in operation_scenario_ids:
        logger.info(f"FlexOperation Model --> Scenario = {id_operation_scenario}.")
        scenario = OperationScenario(scenario_id=id_operation_scenario, config=config)

        """
        run ref model
        """
        ref_model = RefOperationModel(scenario).solve()
        RefDataCollector(ref_model, scenario.scenario_id, config, save_hour_results=True).run()

        """
        run opt model
        """
        # opt_instance = OptInstance().create_instance()
        # try:
        #     opt_model = OptOperationModel(scenario).solve(opt_instance)
        #     OptDataCollector(opt_model, scenario.scenario_id, config, save_hour_results=True).run()
        # except ValueError:
        #     print(f'Infeasible --> ID_Scenario = {scenario.scenario_id}')


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


def run_operation_analyzer_kevan(config: "Config", scenario_id: int):
    ana = OperationAnalyzer(config)
    # ana.plot_electricity_balance_demand(scenario_id=scenario_id, model="ref")
    # ana.plot_scenario_energy_demand(scenario_id=scenario_id)

    # ana.plot_scenario_electricity_balance(scenario_id=scenario_id)
    ana.plot_scenario_energy_demand_mean_seasonal(scenario_id=scenario_id)
    ana.plot_scenario_energy_demand_mean_yearly(scenario_id=scenario_id)
