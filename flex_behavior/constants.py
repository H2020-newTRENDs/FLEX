from dataclasses import dataclass


@dataclass
class BehaviorTable:
    Scenarios = "BehaviorScenario"
    ID_HouseholdType = "BehaviorScenario_Info_HouseholdType"
    ID_PersonType = "BehaviorScenario_Info_PersonType"
    ID_Activity = "BehaviorScenario_Info_Activity"
    ID_Location = "BehaviorScenario_Info_Location"
    ID_Technology = "BehaviorScenario_Info_Technology"
    HouseholdComposition = "BehaviorScenario_HouseholdComposition"
    ActivityChangeProb = "BehaviorScenario_Activity_ChangeProb"
    ActivityDurationProb = "BehaviorScenario_Activity_DurationProb"
    ActivityLocation = "BehaviorScenario_Activity_Location"
    TechnologyTriggerProbability = "BehaviorScenario_Technology_TriggerProbability"
    TechnologyPower = "BehaviorScenario_Technology_Power"
    TechnologyDuration = "BehaviorScenario_Technology_Duration"
    PersonProfiles = "BehaviorResult_PersonProfiles"
    HouseholdProfiles = "BehaviorResult_HouseholdProfiles"
