
class HouseholdBehaviorTables:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.Object = "OBJ_"
        self.Scenario = "Sce_"
        self.GeneratedData = "Gen_"
        self.Result = "Res_"

        # Exogenous tables: ID and parameter
        self.ID_Country = self.ID + "Country"

        # Generated tables: objects
        self.Gen_OBJ_ID_Household = self.GeneratedData + self.Object + self.ID + "Household"

        # Result tables
        self.Res_HouseholdBehavior = self.Result + "HouseholdBehavior"