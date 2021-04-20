
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A1_Config.b_Register import REG
from A_Infrastructure.A2_ToolKits.a_DB import DB

class TableGenerator:

    """
    based on the "ID_" and "Exo_" tables, generate all the "Gen_" tables.
    """

    def __init__(self, conn):
        self.Conn = conn

    def run(self):
        Duration = DB().read_DataFrame(REG().Exo_WashingMachineDuration, self.Conn).iloc[0]["Value"]
        print(Duration)
        pass

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    TableGenerator(CONN).run()



