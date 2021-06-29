
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A3_DB import DB
from C_Model.C2_OperationOptimization.C21_OperationOptimization import OperationOptimization

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # Ope_TableGenerator(CONN).run()
    OperationOptimization(CONN).run()














