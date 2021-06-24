
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from C_Model.C1_TableGenerator.C11_Ope_TableGenerator import Ope_TableGenerator
from C_Model.C2_OperationOptimization.C21_OperationOptimization import OperationOptimization
from B_Classes.B2_Building import Building, HeatingCooling_noDR

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # Ope_TableGenerator(CONN).run()
    OperationOptimization(CONN).run()














