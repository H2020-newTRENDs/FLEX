
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class Environment:

    """
    Data type:
    (1) self.ElectricityPrice: numpy real array (len = 8760)
    (2) self.FeedinTariff: numpy real array (len = 8760)
    """

    def __init__(self, household, para_series, conn):
        self.Household = household
        self.Conn = conn

        # ElectricityPrice
        self.ElectricityPrice = DB().read_DataFrame(REG().Exo_HourlyElectricityPrice, self.Conn,
                                                    ID_ElectricityPriceScenario=para_series["ID_ElectricityPriceScenario"],
                                                    ID_Country=self.Household.ID_Country)["Value"].values

        # FeedinTariff
        self.FeedinTariff = DB().read_DataFrame(REG().Exo_HourlyFeedinTariff, self.Conn,
                                                    ID_FeedinTariffScenario=para_series["ID_FeedinTariffScenario"],
                                                    ID_Country=self.Household.ID_Country)["Value"].values

        # Temperature
        self.Temperature = DB().read_DataFrame(REG().Exo_BaseYearTemperature, self.Conn,
                                                    ID_Country=self.Household.ID_Country)["Value"].values

        # Irradiation
        self.Irradiation = DB().read_DataFrame(REG().Exo_BaseYearIrradiation, self.Conn,
                                               ID_Country=self.Household.ID_Country)["Value"].values





















