
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB

class TableGenerator:

    """
    based on the "ID_" and "Exo_" tables, generate all the "Gen_" tables.
    """

    def __init__(self, conn):
        self.Conn = conn

    def run(self):
        ParameterValue = DB().read_GlobalParameterValue("PumpConsumptionForUnitSpaceHeating", self.Conn)
        print(ParameterValue)
        pass