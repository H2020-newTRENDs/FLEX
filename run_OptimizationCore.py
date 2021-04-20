
from A_Infrastructure.A1_Config.a_Constants import CONS
from A_Infrastructure.A2_ToolKits.a_DB import DB
from C_Model.C2_OptimizationCore.OptimizationCore import OptimizationCore


if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    OptimizationCore(CONN).run()




