
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from B_Classes.B1_Household import Household

class OptimizationCore:

    """
    Intro

    Optimize the prosumaging behavior of all representative "household - environment" combinations.
    Read all big fundamental tables at the beginning and keep them in the memory.
    (1) Appliances parameter table
    (2) Energy price
    Go over all the aspects and summarize the flexibilities to be optimized.
    Formulate the final optimization problem.
    """

    def __init__(self, conn):
        self.Conn = conn
        self.ID_Household = DB().read_DataFrame(REG().Gen_OBJ_ID_Household, self.Conn)
        # self.ID_Environment = DB().read_DataFrame(REG().Gen_Sec_ID_Environment, self.Conn)

    def gen_Household(self, para_series):
        HouseholdAgent = Household(para_series, self.Conn)
        print(HouseholdAgent.SpaceHeating.TankSize)
        pass

    def run(self):
        self.gen_Household(self.ID_Household.iloc[0])
        pass



