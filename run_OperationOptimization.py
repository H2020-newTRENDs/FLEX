
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C4_OperationOptimization import OperationOptimization

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # Ope_TableGenerator(CONN).run()
    OperationOptimization(CONN).run()














