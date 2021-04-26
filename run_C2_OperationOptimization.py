
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from C_Model.C2_OperationOptimization.C21_OperationOptimization import OptimizationCore

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    OptimizationCore(CONN).run()




