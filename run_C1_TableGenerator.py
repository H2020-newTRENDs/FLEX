
from A_Infrastructure.A1_Config.A11_Constants import CONS
from A_Infrastructure.A1_Config.A12_Register import REG
from A_Infrastructure.A2_ToolKits.A21_DB import DB
from C_Model.C1_TableGenerator.C11_TableGenerator import TableGenerator

if __name__ == "__main__":

    CONN = DB().create_Connection(CONS().RootDB)
    TableGenerator(CONN).run()



