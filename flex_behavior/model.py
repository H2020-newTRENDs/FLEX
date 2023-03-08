from flex_behavior.scenario import BehaviorScenario
from flex_behavior.household import Household


class BehaviorModel:

    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario
        self.household = self.setup_household(scenario)

    @staticmethod
    def setup_household(scenario: "BehaviorScenario"):
        household = Household(scenario)
        household.setup()
        return household

    def save_household_profiles(self):
        ...

    def run(self):
        print("")
        print(f"FLEX-Behavior --> ID_Scenario = {self.scenario.scenario_id}.")
