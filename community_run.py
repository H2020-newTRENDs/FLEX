from basics.db import create_db_conn
from basics.kit import get_logger
from config import config
from models.community.aggregator import Aggregator
from models.community.data_collector import CommunityDataCollector
from models.community.model import CommunityModel
from models.community.scenario import CommunityScenario
from models.operation.enums import OperationTable

logger = get_logger(__name__)


def run_community_model(scenario: "CommunityScenario", opt_instance):
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
        print(f'Infeasible --> ID_CommunityScenario = {scenario.scenario_id}')


def run_scenarios(community_scenario_ids, operation_scenario_ids):
    db = create_db_conn(config)
    logger.info(f"Importing the OperationModel results...")
    operation_scenario = db.read_dataframe(OperationTable.Scenarios.value)
    ref_operation_result_hour = db.read_dataframe(OperationTable.ResultRefHour.value)
    ref_operation_result_year = db.read_dataframe(OperationTable.ResultRefYear.value)
    for id_community_scenario in community_scenario_ids:
        logger.info(f"FlexCommunity --> ID_CommunityScenario = {id_community_scenario}.")
        scenario = CommunityScenario(id_community_scenario, config)
        scenario.operation_scenario_ids = operation_scenario_ids
        scenario.operation_scenario = operation_scenario
        scenario.ref_operation_result_hour = ref_operation_result_hour
        scenario.ref_operation_result_year = ref_operation_result_year
        scenario.setup()
        opt_instance = Aggregator().create_opt_instance()
        run_community_model(scenario, opt_instance)


if __name__ == "__main__":
    COMMUNITY_SCENARIO_IDS = [id_scenario for id_scenario in range(1, 31)]
    OPERATION_SCENARIO_IDS = [id_scenario for id_scenario in range(1, 193)]
    run_scenarios(COMMUNITY_SCENARIO_IDS, OPERATION_SCENARIO_IDS)



