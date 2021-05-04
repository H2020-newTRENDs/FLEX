
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class OptimizationCore:

    """
    Optimize the prosumaging behavior of all representative "household - environment" combinations.

    Read all big fundamental tables at the beginning and keep them in the memory.
    (1) Appliances parameter table
    (2) Energy price
    Go over all the aspects and summarize the flexibilities to be optimized.
    Formulate the final optimization problem.
    """

    def __init__(self, conn):
        self.Conn = conn





    def run(self):

        pass



