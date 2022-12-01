from flex_behavior.scenario import BehaviorScenario


class BehaviorModel:

    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario

    def run(self):
        print(f"FLEX-Behavior --> ID_Scenario = {self.scenario.scenario_id}.")
