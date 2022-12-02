from dataclasses import dataclass


@dataclass
class BehaviorTable:
    Scenarios = "BehaviorScenario"
    ID_Activity = "BehaviorScenario_ActivityInfo"
    ID_Technology = "BehaviorScenario_TechnologyInfo"
    ActivityProfile = "BehaviorScenario_ActivityProfile"
