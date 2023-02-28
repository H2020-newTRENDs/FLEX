from typing import List

from flex.config import Config
from flex.kit import get_logger
from flex_behavior.model import BehaviorModel
from flex_behavior.scenario import BehaviorScenario
from flex_behavior.analyzer import BehaviorAnalyzer
from flex_behavior.person import Person
from flex_behavior.data_collector import BehaviorDataCollector
from flex_behavior.constants import BehaviorTable
import pandas as pd
logger = get_logger(__name__)


def gen_person_profiles(config: "Config", sample_size: int = 1):
    scenario = BehaviorScenario(scenario_id=1, config=config)
    data_collector = BehaviorDataCollector(scenario.scenario_id, config)
    person_types = scenario.db.read_dataframe(BehaviorTable.ID_PersonType)["ID_PersonType"].tolist()
    d = {}
    for sample in range(0, sample_size):
        # for id_person_type in person_types:
        for id_person_type in [1]:
            person = Person(scenario, id_person_type)
            person.setup()
            d[f"activity_p{id_person_type}s{sample}"] = person.activity_profile
            d[f"electricity_p{id_person_type}s{sample}"] = person.electricity_demand
            d[f"hotwater_p{id_person_type}s{sample}"] = person.hot_water_demand
    df = pd.DataFrame(d)
    data_collector.db.write_dataframe(
        table_name=BehaviorTable.PersonProfiles,
        data_frame=df,
        if_exists="replace"
    )


def run_behavior_model(behavior_scenario_ids: List[int], config: "Config"):
    for id_behavior_scenario in behavior_scenario_ids:
        logger.info(f"FLEX-Behavior Model --> ID_Scenario = {id_behavior_scenario}.")
        behavior_scenario = BehaviorScenario(scenario_id=id_behavior_scenario, config=config)
        behavior_scenario.setup_scenario_params()
        behavior_scenario.import_scenario_data()
        model = BehaviorModel(behavior_scenario)
        model.run()


def run_behavior_analyzer(behavior_scenario_ids: List[int], config: "Config"):
    for id_behavior_scenario in behavior_scenario_ids:
        logger.info(f"FLEX-Behavior Analyzer --> ID_Scenario = {id_behavior_scenario}.")
        ana = BehaviorAnalyzer(config)
        ana.run()


if __name__ == "__main__":
    cfg = Config(project_name="FLEX_Behavior")
    gen_person_profiles(config=cfg)
