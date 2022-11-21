from config import config
from models.behavior.model import BehaviorModel
from models.behavior.scenario import BehaviorScenario


def get_behavior_scenario(scenario_id: int):
    behavior_scenario = BehaviorScenario(scenario_id=scenario_id, config=config)
    behavior_scenario.setup()
    return behavior_scenario


if __name__ == "__main__":
    scenario = get_behavior_scenario(scenario_id=1)
    model = BehaviorModel(scenario)
    model.run()
