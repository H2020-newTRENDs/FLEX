
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from C_Model.C1_TableGenerator.C11_Ope_TableGenerator import Ope_TableGenerator
from C_Model.C2_OperationOptimization.C21_OperationOptimization import OperationOptimization

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    # Ope_TableGenerator(CONN).run()
    OperationOptimization(CONN).run()





    # table_name = REG().Sce_ID_TimeStructure
    # table = DB().read_DataFrame(table_name, CONN, ID_Day=5)
    # table_select = table.iloc[10]["ID_Day"]
    # table_select = table.loc[table["ID_Day"] == 5]
    # print(table_select)
    # print(table)






