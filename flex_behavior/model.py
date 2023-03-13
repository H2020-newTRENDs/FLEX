from typing import Optional

from flex_behavior.household import Household
from flex_behavior.scenario import BehaviorScenario


class BehaviorModel:

    def __init__(self, scenario: "BehaviorScenario"):
        self.scenario = scenario
        self.household: Optional["Household"] = None

    def setup_household(self):
        self.household = Household(self.scenario)
        self.household.setup()

    def run(self):
        self.setup_household()
