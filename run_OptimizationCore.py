
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A2_ToolKits.a_DB import DB

class OptimizationCore:

    """
    optimize the prosumaging behavior of all representative "household - environment" combinations.
    """

    def __init__(self, conn):
        self.Conn = conn

    def run(self):
        print(bool(1))
        pass

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    OptimizationCore(CONN).run()




