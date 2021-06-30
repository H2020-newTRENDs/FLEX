
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C3_TableGenerator import TableGenerator
from C_Model_Operation.C4_OperationOptimization import OperationOptimization
from C_Model_Operation.C5_Visualization import Visualization


if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # TableGenerator(CONN).run()
    # OperationOptimization(CONN).run()
    Visualization(CONN).run()



