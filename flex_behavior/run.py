import logging
from typing import List

from flex.config import Config
from flex_behavior.model import BehaviorModel
from flex_behavior.scenario import BehaviorScenario
from flex_behavior.analyzer import BehaviorAnalyzer


def run_behavior_model(behavior_scenario_ids: List[int], config: "Config"):
    logger = logging.getLogger(config.project_name)
    for id_behavior_scenario in behavior_scenario_ids:
        logger.info(f"FLEX-Behavior Model --> ID_Scenario = {id_behavior_scenario}.")
        behavior_scenario = BehaviorScenario(scenario_id=id_behavior_scenario, config=config)
        behavior_scenario.setup_scenario_params()
        behavior_scenario.import_scenario_data()
        model = BehaviorModel(behavior_scenario)
        model.run()


def run_behavior_analyzer(behavior_scenario_ids: List[int], config: "Config"):
    logger = logging.getLogger(config.project_name)
    for id_behavior_scenario in behavior_scenario_ids:
        logger.info(f"FLEX-Behavior Analyzer --> ID_Scenario = {id_behavior_scenario}.")
        ana = BehaviorAnalyzer(config)
        ana.run()
