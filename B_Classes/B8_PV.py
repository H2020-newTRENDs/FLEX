
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class PV:

    """
    Data type:
    (1)
    """

    def __init__(self, household, para_series, conn):
        self.Household = household
        self.Conn = conn


    def calc_PVGeneration(self):

        # move the generation calculation method in the tool here
        Irradiation = self.Household.Environment.Irradiation
        Generation = Irradiation * 0.2 * self.PVSize * self.ff

        return Generation
