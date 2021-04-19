
class REG:

    def __init__(self):

        # Prefix
        self.ID = "ID_"
        self.ExogenousData = "Exo_"
        self.GeneratedData = "Gen_"

        # ID_Region
        self.ID_Country = self.ID + "Country"
        self.ID_BaseYearTimeStructure = self.ID + "BaseYearTimeStructure" # base year: 2010
        self.ID_BuildingType = self.ID + "BuildingType"
        self.ID_BuildingAgeClass = self.ID + "BuildingAgeClass"

        # Exogenous Non-scenario-dependent Table
        self.Exo_BaseYearIrradiation = self.ExogenousData + "BaseYearIrradiation" # from helioclim dataset for year 2010
        self.Exo_BaseYearTemperature = self.ExogenousData + "BaseYearTemperature" # from MERRA2 dataset for year 2010

        # Generated Table









