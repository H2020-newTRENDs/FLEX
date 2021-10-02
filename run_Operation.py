
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table, REG_Var
from C_Model_Operation.C3_TableGenerator import TableGenerator


#from C_Model_Operation.C4_RQ2_Reference_OperationOptimization import OperationOptimization
#from C_Model_Operation.C4_RQ3_Battery_and_Optimization import OperationOptimization
from C_Model_Operation.C4_RQ4_OperationOptimization_EV import OperationOptimization

#from C_Model_Operation.C4_OperationOptimization import OperationOptimization

from C_Model_Operation.C5_Visualization import Visualization


if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    #TableGenerator(CONN).run()
    #OperationOptimization(CONN).run()
    Visualization(CONN).run()


# test

