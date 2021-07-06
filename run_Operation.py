
from A_Infrastructure.A1_CONS import CONS
from A_Infrastructure.A2_DB import DB
from C_Model_Operation.C1_REG import REG_Table, REG_Var
from C_Model_Operation.C3_TableGenerator import TableGenerator
from C_Model_Operation.C4_OperationOptimization import OperationOptimization
from C_Model_Operation.C5_Visualization import Visualization


if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # TableGenerator(CONN).run()
    # OperationOptimization(CONN).run()
    # Visualization(CONN).run()

    df = DB().read_DataFrame(REG_Table().Res_SystemOperationHour, CONN)
    array1 = df.loc[df["ID_Environment"] == 1][REG_Var().BuildingMassTemperature]
    array2 = df.loc[df["ID_Environment"] == 2][REG_Var().BuildingMassTemperature]
    difference = array1.to_numpy()-array2.to_numpy()
    for t in range(0, 8760):
        print("t = " + str(t + 1) + ", diff = " + str(round(difference[t], 4)))


