from dataclasses import dataclass


@dataclass
class BehaviorTable:
    Scenarios = "BehaviorScenario"
    ID_HouseholdType = "BehaviorScenario_Info_HouseholdType"
    ID_PersonType = "BehaviorScenario_Info_PersonType"
    ID_Activity = "BehaviorScenario_Info_Activity"
    ID_Technology = "BehaviorScenario_Info_Technology"
    ID_TechnologyType = "BehaviorScenario_Info_TechnologyType"
    HouseholdComposition = "BehaviorScenario_HouseholdComposition"
    ActivityProfile = "BehaviorScenario_ActivityProfile"
    TechnologyType = "BehaviorScenario_TechnologyType"
    TechnologyTriggerProbability = "BehaviorScenario_TechnologyTriggerProbability"
    TechnologyOwnershipRate = "BehaviorScenario_TechnologyOwnershipRate"
    TechnologyPowerActive = "BehaviorScenario_TechnologyPowerActive"
    TechnologyPowerStandby = "BehaviorScenario_TechnologyPowerStandby"

