
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class OptimizationCore:

    """
    Optimize the prosumaging behavior of all representative "household - environment" combinations.

    Energy price is imported and used here.
    Go over all the aspects and summarize the flexibilities to be optimized.
    Formulate the final optimization problem.
    """

    def __init__(self, conn):
        self.Conn = conn

    def run(self):
        print(DB().read_DataFrame(REG().Gen_ID_OBJ_Environment, self.Conn).iloc[0])
        pass

    pass


