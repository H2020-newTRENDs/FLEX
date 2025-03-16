from typing import List

from flex.config import Config
from flex.db import create_db_conn
from flex_community.aggregator import Aggregator
from flex_community.analyzer import CommunityAnalyzer
from flex_community.data_collector import CommunityDataCollector
from flex_community.model import CommunityModel
from flex_community.scenario import CommunityScenario
from flex_operation.constants import OperationTable


def run_community_model(community_scenario_ids, operation_scenario_ids, config: "Config"):
    db = create_db_conn(config)
    logger.info(f"Importing the FLEX-Operation results...")
    operation_scenario = db.read_dataframe(OperationTable.Scenarios)
    ref_operation_result_hour = db.read_dataframe(OperationTable.ResultRefHour)
    ref_operation_result_year = db.read_dataframe(OperationTable.ResultRefYear)
    for id_community_scenario in community_scenario_ids:
        logger.info(f"FLEX-Community Model --> ID_Scenario = {id_community_scenario}.")
        scenario = CommunityScenario(id_community_scenario, config)
        scenario.operation_scenario_ids = operation_scenario_ids
        scenario.operation_scenario = operation_scenario
        scenario.ref_operation_result_hour = ref_operation_result_hour
        scenario.ref_operation_result_year = ref_operation_result_year
        scenario.setup()
        opt_instance = Aggregator().create_opt_instance()
        try:
            community_model = CommunityModel(scenario)
            community_model.solve_aggregator_p2p_trading()
            opt_instance = community_model.solve_aggregator_optimization(opt_instance)
            CommunityDataCollector(
                model=community_model,
                opt_instance=opt_instance,
                scenario_id=scenario.scenario_id,
                config=config
            ).run()
        except ValueError:
            print(f'FLEX-Community Infeasible --> ID_Scenario = {scenario.scenario_id}')


def run_community_analyzer(community_scenario_ids: List[int], config: "Config"):
    ana = CommunityAnalyzer(config)
    for id_community_scenario in community_scenario_ids:
        logger.info(f"FLEX-Community Analyzer --> ID_Scenario = {id_community_scenario}.")
        ana.plot_aggregator_profit_against_battery_size()
        ana.plot_p2p_trading_amount_by_month(id_scenario=id_community_scenario)
        ana.plot_battery_operation_by_month(id_scenario=id_community_scenario)
