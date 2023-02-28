from dataclasses import dataclass


@dataclass
class BehaviorTable:
    Scenarios = "BehaviorScenario"
    ID_HouseholdType = "BehaviorScenario_Info_HouseholdType"
    ID_PersonType = "BehaviorScenario_Info_PersonType"
    ID_Activity = "BehaviorScenario_Info_Activity"
    ID_Location = "BehaviorScenario_Info_Location"
    ID_Technology = "BehaviorScenario_Info_Technology"
    ID_TechnologyType = "BehaviorScenario_Info_TechnologyType"
    HouseholdComposition = "BehaviorScenario_HouseholdComposition"
    ActivityChangeProb = "BehaviorScenario_Activity_ChangeProb"
    ActivityDurationProb = "BehaviorScenario_Activity_DurationProb"
    ActivityLocation = "BehaviorScenario_Activity_Location"
    TechnologyType = "BehaviorScenario_Technology_Type"
    TechnologyTriggerProbability = "BehaviorScenario_Technology_TriggerProbability"
    TechnologyPower = "BehaviorScenario_Technology_Power"
    PersonProfiles = "BehaviorResult_PersonProfiles"

