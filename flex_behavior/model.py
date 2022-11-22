from flex_behavior.scenario import BehaviorScenario


class BehaviorModel:

    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario

    def run(self):
        print(self.scenario.tou_profile)
